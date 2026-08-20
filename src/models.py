"""Model architectures."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SurgicalCoAttention(nn.Module):
    """Co-attention module combining instrument and eye-region attention."""

    def __init__(self, in_ch):
        super().__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.inst_net = nn.Conv2d(64, 1, kernel_size=1)
        self.eye_net = nn.Conv2d(64, 1, kernel_size=1)
        self.fusion = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.alpha = nn.Parameter(torch.ones(1))

        # Spatial gating branch
        self.gate_conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.gate_relu = nn.ReLU(inplace=True)
        self.gate_conv2 = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        shared = self.shared_conv(x)
        inst_attn = torch.sigmoid(self.inst_net(shared))
        eye_attn = torch.sigmoid(self.eye_net(shared))

        w = self.fusion(torch.cat([inst_attn, eye_attn], dim=1))
        w = w * (1 + self.alpha)

        g = torch.cat([inst_attn, eye_attn], dim=1)
        g = self.gate_relu(self.gate_conv1(g))
        g = torch.sigmoid(self.gate_conv2(g))

        w = w * g
        return x * w + x


class SurgicalNet(nn.Module):
    """4-channel surgical phase classifier with co-attention."""

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            'regnety_008',
            in_chans=4,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3)
        )
        ch = self.backbone.feature_info.channels()
        self.co1 = SurgicalCoAttention(ch[0])
        self.co2 = SurgicalCoAttention(ch[1])
        self.co3 = SurgicalCoAttention(ch[2])

        self.decoder = nn.Sequential(
            nn.Conv2d(sum(ch), 512, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f1, f2, f3 = self.backbone(x)
        f1, f2, f3 = self.co1(f1), self.co2(f2), self.co3(f3)
        sz = f3.shape[2:]
        f1a = F.interpolate(f1, size=sz, mode='nearest')
        f2a = F.interpolate(f2, size=sz, mode='nearest')
        fused = torch.cat([f1a, f2a, f3], dim=1)
        dec = self.decoder(fused)
        return self.classifier(dec)
