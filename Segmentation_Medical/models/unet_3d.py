"""
U-Net 3D architecture implementation
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """3D Convolution Block: Conv3D -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric medical image segmentation
    
    Architecture:
    - Encoder: 4 downsampling blocks
    - Bottleneck: Dense feature extraction
    - Decoder: 4 upsampling blocks with skip connections
    
    Input: (B, C, D, H, W) where D=H=W=128
    Output: (B, 1, D, H, W) probability map
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: list = [32, 64, 128, 256]):
        super().__init__()
        self.features = features
        
        # Encoder
        self.encoder1 = ConvBlock3D(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(2)
        
        self.encoder2 = ConvBlock3D(features[0], features[1])
        self.pool2 = nn.MaxPool3d(2)
        
        self.encoder3 = ConvBlock3D(features[1], features[2])
        self.pool3 = nn.MaxPool3d(2)
        
        self.encoder4 = ConvBlock3D(features[2], features[3])
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(features[3], features[3] * 2)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose3d(features[3] * 2, features[3],
                                          kernel_size=2, stride=2)
        self.decoder4 = ConvBlock3D(features[3] * 2, features[3])
        
        self.upconv3 = nn.ConvTranspose3d(features[3], features[2],
                                          kernel_size=2, stride=2)
        self.decoder3 = ConvBlock3D(features[2] * 2, features[2])
        
        self.upconv2 = nn.ConvTranspose3d(features[2], features[1],
                                          kernel_size=2, stride=2)
        self.decoder2 = ConvBlock3D(features[1] * 2, features[1])
        
        self.upconv1 = nn.ConvTranspose3d(features[1], features[0],
                                          kernel_size=2, stride=2)
        self.decoder1 = ConvBlock3D(features[0] * 2, features[0])
        
        # Output layer
        self.final = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        
        enc3 = self.encoder3(x)
        x = self.pool3(enc3)
        
        enc4 = self.encoder4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.decoder4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)
        
        # Output
        x = self.final(x)
        x = self.sigmoid(x)
        
        return x


if __name__ == "__main__":
    # Test the model
    model = UNet3D(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 128, 128, 128)  # Batch, Channels, Depth, Height, Width
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
