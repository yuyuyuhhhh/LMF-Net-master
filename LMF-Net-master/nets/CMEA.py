import torch
import torch.nn as nn
from nets.eca import ECAAttention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(y))

class ImprovedCMEA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        C = in_channels

        self.dw3 = self._make_dsconv(C,kernel_size=3)
        self.dw5 = self._make_dsconv(C,kernel_size=5)
        self.dw7 = self._make_dsconv(C, kernel_size=7)

        self.fuse_conv = nn.Conv2d(C*3, C, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(C)

        self.eca = ECAAttention(16)
        self.sa = SpatialAttention(kernel_size=7)

        self.relu = nn.ReLU(inplace=True)

    def _make_dsconv(self, C, kernel_size):
        return nn.Sequential(
            nn.Conv2d(C, C, kernel_size, padding=kernel_size//2, groups=C, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        identity = x

        f3 = self.dw3(x)
        f5 = self.dw5(x)
        f7 = self.dw7(x)

        fused = torch.cat([f3, f5, f7], dim=1)
        fused = self.fuse_conv(fused)
        fused = self.bn_fuse(fused)
        fused = self.relu(fused + identity)

        out = self.eca(fused)
        out = self.sa(out)

        return out
