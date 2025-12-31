import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

class BoundaryFusionNet(nn.Module):
    def __init__(self, in_channels=3, num_blocks=6, feat_channels=64):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, feat_channels, 3, padding=1))
        for _ in range(num_blocks):
            layers.append(ResBlock(feat_channels))
        layers.append(nn.Conv2d(feat_channels, in_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
