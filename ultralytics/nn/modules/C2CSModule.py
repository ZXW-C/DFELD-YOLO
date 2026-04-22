import torch
import torch.nn as nn
from .conv import Conv

__all__ = ['C2CSModule']

class CrossShapedStripeAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert num_heads % 2 == 0, "num_heads must be even for parallel grouping"
        self.horiz_heads = num_heads // 2
        self.vert_heads = num_heads // 2
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5

        # qkv
        self.qkv = Conv(
            dim,
            dim +
            self.horiz_heads * self.key_dim +
            self.horiz_heads * self.head_dim +
            self.vert_heads * self.key_dim +
            self.vert_heads * self.head_dim,
            1, act=False
        )
        self.proj = Conv(dim, dim, 1, act=False)

    def dynamic_stripe_width(self, H, W):
        sw_candidates = [8, 4, 2, 1]
        sw = 1
        for cand in sw_candidates:
            if H % cand == 0 and W % cand == 0:
                sw = cand
                break
        if H >= 64 or W >= 64:
            sw = min(sw, 2)
        elif H >= 32 or W >= 32:
            sw = min(sw, 4)
        else:
            sw = min(sw, 8)
        return sw

    def stripe_attention(self, q, k, v, stripe_type, sw):
        B, heads, dim, H, W = q.shape
        head_dim = self.head_dim
        if stripe_type == 'horizontal':
            q = q.contiguous().reshape(B, heads, dim, H // sw, sw, W)
            k = k.contiguous().reshape(B, heads, dim, H // sw, sw, W)
            v = v.contiguous().reshape(B, heads, head_dim, H // sw, sw, W)

            q = q.permute(0, 1, 3, 4, 5, 2).contiguous().reshape(B * heads * (H // sw), sw * W, dim)
            k = k.permute(0, 1, 3, 4, 5, 2).contiguous().reshape(B * heads * (H // sw), sw * W, dim)
            v = v.permute(0, 1, 3, 4, 5, 2).contiguous().reshape(B * heads * (H // sw), sw * W, head_dim)
        else:  # vertical
            q = q.contiguous().reshape(B, heads, dim, H, W // sw, sw)
            k = k.contiguous().reshape(B, heads, dim, H, W // sw, sw)
            v = v.contiguous().reshape(B, heads, head_dim, H, W // sw, sw)

            q = q.permute(0, 1, 4, 2, 5, 3).contiguous().reshape(B * heads * (W // sw), H * sw, dim)
            k = k.permute(0, 1, 4, 2, 5, 3).contiguous().reshape(B * heads * (W // sw), H * sw, dim)
            v = v.permute(0, 1, 4, 2, 5, 3).contiguous().reshape(B * heads * (W // sw), H * sw, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1, dtype=q.dtype)
        out = attn @ v

        if stripe_type == 'horizontal':
            out = out.contiguous().reshape(B, heads, H // sw, sw, W, head_dim)
            out = out.permute(0, 1, 5, 2, 3, 4).contiguous().reshape(B, heads, head_dim, H, W)
        else:
            out = out.contiguous().reshape(B, heads, W // sw, H, sw, head_dim)
            out = out.permute(0, 1, 5, 3, 2, 4).contiguous().reshape(B, heads, head_dim, H, W)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        sw = self.dynamic_stripe_width(H, W)

        qkv = self.qkv(x).contiguous()
        q_global = qkv[:, :self.dim, :, :].contiguous().reshape(B, self.num_heads, self.head_dim, H, W)

        k_horiz_ch = self.horiz_heads * self.key_dim
        v_horiz_ch = self.horiz_heads * self.head_dim
        k_vert_ch = self.vert_heads * self.key_dim
        v_vert_ch = self.vert_heads * self.head_dim
        k_horiz, v_horiz, k_vert, v_vert = torch.split(
            qkv[:, self.dim:, :, :].contiguous(),
            [k_horiz_ch, v_horiz_ch, k_vert_ch, v_vert_ch],
            dim=1
        )

        q_horiz = q_global[:, :self.horiz_heads, :self.key_dim, :, :].contiguous()
        k_horiz = k_horiz.contiguous().reshape(B, self.horiz_heads, self.key_dim, H, W)
        v_horiz = v_horiz.contiguous().reshape(B, self.horiz_heads, self.head_dim, H, W)

        q_vert = q_global[:, self.horiz_heads:, :self.key_dim, :, :].contiguous()
        k_vert = k_vert.contiguous().reshape(B, self.vert_heads, self.key_dim, H, W)
        v_vert = v_vert.contiguous().reshape(B, self.vert_heads, self.head_dim, H, W)

        out_horiz = self.stripe_attention(q_horiz, k_horiz, v_horiz, 'horizontal', sw)
        out_vert = self.stripe_attention(q_vert, k_vert, v_vert, 'vertical', sw)

        out = torch.cat([out_horiz, out_vert], dim=1).contiguous()
        out = out.permute(0, 2, 3, 4, 1).contiguous().reshape(B, C, H, W)

        return self.proj(out)


class CSSABlock(nn.Module):
    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        super().__init__()
        self.attn = CrossShapedStripeAttention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2CSModule(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(CSSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
