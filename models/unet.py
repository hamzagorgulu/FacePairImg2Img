import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class LightweightUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder layers
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.enc5 = DoubleConv(256, 512)
        self.enc6 = DoubleConv(512, 1024)  # Added deeper layers
        self.enc7 = DoubleConv(1024, 2048)  # Deepest layer
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Decoder layers
        self.dec7 = DoubleConv(2048 + 1024, 1024)
        self.dec6 = DoubleConv(1024 + 512, 512)
        self.dec5 = DoubleConv(512 + 256, 256)
        self.dec4 = DoubleConv(256 + 128, 128)
        self.dec3 = DoubleConv(128 + 64, 64)
        self.dec2 = DoubleConv(64 + 32, 32)
        self.dec1 = nn.Conv2d(32, 3, kernel_size=1)
        
        self.final_act = nn.Tanh()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))
        e7 = self.enc7(self.pool(e6))  # Deepest encoder level
        
        # Decoder
        d7 = self.dec7(torch.cat([self.upsample(e7), e6], dim=1))
        d6 = self.dec6(torch.cat([self.upsample(d7), e5], dim=1))
        d5 = self.dec5(torch.cat([self.upsample(d6), e4], dim=1))
        d4 = self.dec4(torch.cat([self.upsample(d5), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], dim=1))
        
        return self.final_act(self.dec1(d2))
