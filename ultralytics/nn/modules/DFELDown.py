import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("DFELDown",)

class DALL(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.in_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
        )
        self.global_context = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
        self.channel_sigmoid = nn.Sigmoid()


        self.spatial_convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            dw_conv = nn.Sequential(
                nn.Conv2d(2, 2, k, padding=padding, groups=2, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True)
            )
            pw_conv = nn.Conv2d(2, 1, 1, bias=False)
            self.spatial_convs.append(nn.Sequential(dw_conv, pw_conv))
        self.spatial_weights = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))

        self.line_enhancer = nn.Sequential(
            nn.Conv2d(1, 1, (5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.local_focus = nn.Sequential(
            nn.Conv2d(1, 1, 7, padding=3, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 5, padding=4, dilation=2, bias=False),
            nn.Sigmoid()
        )
        self.spatial_sigmoid = nn.Sigmoid()

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, 1, 1),
            nn.Sigmoid()
        )
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 3, padding=1, groups=in_channels // 4),
            nn.Sigmoid()
        )

        self.ctx_weight = nn.Parameter(torch.tensor(-0.8473), requires_grad=True)
        self.spatial_fusion_weights03 = nn.Parameter(torch.tensor(-0.8473), requires_grad=True)
        self.spatial_fusion_weights04 = nn.Parameter(torch.tensor(-0.4055), requires_grad=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        avg_out = self.channel_fc(self.avg_pool(x))
        max_out = self.channel_fc(self.max_pool(x))
        channel_att = self.channel_sigmoid(avg_out + max_out)
        global_ctx = self.global_context(x)
        ctx_weight_learnable = torch.sigmoid(self.ctx_weight)
        channel_out = x * (channel_att + ctx_weight_learnable * global_ctx)

        avg_spatial = torch.mean(channel_out, dim=1, keepdim=True)
        max_spatial, _ = torch.max(channel_out, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_outputs = []
        for conv in self.spatial_convs:
            spatial_outputs.append(conv(spatial_cat))
        weights = F.softmax(self.spatial_weights, dim=0)
        combined_spatial = sum(w * out for w, out in zip(weights, spatial_outputs))
        line_enhanced = self.line_enhancer(combined_spatial)
        local_focused = self.local_focus(combined_spatial)

        weight_learnable03 = torch.sigmoid(self.spatial_fusion_weights03)
        weight_learnable04 = torch.sigmoid(self.spatial_fusion_weights04)
        spatial_att = self.spatial_sigmoid(combined_spatial + weight_learnable04 * line_enhanced + weight_learnable03 * local_focused)
        spatial_out = channel_out * spatial_att

        fusion_weight = self.fusion_gate(x)
        multi_scale_weight = self.multi_scale_fusion(spatial_out)
        output = identity + fusion_weight * (spatial_out * multi_scale_weight)
        return output


class LSDD(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(in_channels * (scale ** 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=1, groups=out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        if h % self.scale != 0 or w % self.scale != 0:
            new_h = (h // self.scale) * self.scale
            new_w = (w // self.scale) * self.scale
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            h, w = new_h, new_w
        out_h, out_w = h // self.scale, w // self.scale
        out_c = c * (self.scale ** 2)

        # Cw-SPD operation
        x = x.view(b, c, out_h, self.scale, out_w, self.scale)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, out_c, out_h, out_w)

        x = self.channel_adjust(x)
        attention = self.multi_scale_fusion(x)
        x = x * attention
        return x


class DFELDown(nn.Module):
    def __init__(self, c1, c2, reduction_ratio=16, spd_scale=2):
        super().__init__()
        self.dall = DALL(c1, reduction_ratio)
        self.lsdd = LSDD(c1, c2, scale=spd_scale)
        self.damage_confidence = nn.Sequential(
            nn.Conv2d(c2, c2 // 4, 3, padding=1, groups=c2 // 4),
            nn.ReLU(),
            nn.Conv2d(c2 // 4, c2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enhanced = self.dall(x)
        downsampled = self.lsdd(enhanced)
        confidence = self.damage_confidence(downsampled)
        output = downsampled * confidence
        return output
