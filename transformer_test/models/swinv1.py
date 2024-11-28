import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer_v2 import SwinTransformerV2, _cfg

class SwinUNet(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=(256, 256), patch_size=4, num_classes=1):
        super(SwinUNet, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Swin Transformer backbone
        self.backbone = SwinTransformerV2(img_size=img_size,
                                          patch_size=patch_size,
                                          in_chans=in_channels,
                                          num_classes=num_classes,
                                          **_cfg('swin_tiny_patch4_window7_224'))
        
        # Decoder
        self.decoder = nn.ModuleList([
            DecoderBlock(1024, 512),
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64)
        ])
        
        # Final conv layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder (Swin Transformer backbone)
        x = self.backbone(x)
        
        # Decoder
        for decoder_block in self.decoder:
            x = decoder_block(x)
        
        # Final conv layer
        x = self.final_conv(x)
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.up(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# Example usage:
if __name__ == "__main__":
    # Create SwinUNet instance
    model = SwinUNet(in_channels=1, out_channels=1)
    
    # Test with random input
    input_tensor = torch.randn(1, 1, 512, 512)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)
