import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['GhostSEC3']

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = int(in_chs * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class CBR(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU() if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU() if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostSEBottleneck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, se_ratio=0.):
        super(GhostSEBottleneck, self).__init__()
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)
        self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        x = self.se(x)
        x = self.ghost2(x)
        x = x + residual
        return x


class GhostSEC3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, se_ratio=0.25):
        super(GhostSEC3, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.n = n
        self.shortcut = shortcut
        mid_chs = int(c2 * e)

        self.conv1 = CBR(c1, mid_chs, 1)
        self.conv2 = CBR(c1, mid_chs, 1)
        self.bottlenecks = nn.Sequential(
            *[GhostSEBottleneck(mid_chs, mid_chs * 2, mid_chs, se_ratio=se_ratio) for _ in range(n)]
        )
        self.conv_out = CBR(mid_chs * 2, c2, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.bottlenecks(x2)
        x_cat = torch.cat([x1, x2], dim=1)
        out = self.conv_out(x_cat)
        if self.shortcut and self.c1 == self.c2:
            out = out + x
        return out

