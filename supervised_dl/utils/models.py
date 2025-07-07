
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple network
class SimpleReconNet(nn.Module):
    def __init__(self):
        super(SimpleReconNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)
    

# UNet
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2):
        super(UNet, self).__init__()
        self.enc1 = UNetBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(64, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.middle = UNetBlock(64, 64)

        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec3 = UNetBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec2 = UNetBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        m = self.middle(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out_conv(d1)
        return out



