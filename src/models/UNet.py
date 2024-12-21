import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, p=0.2):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p)
        )
        
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.double_conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = self.conv(x)  # reduce channels before concatenation
        
        # Ensure sizes match for concatenation
        if x.shape != skip_connection.shape:
            x = F.interpolate(x, size=skip_connection.shape[2:], 
                            mode='bilinear', align_corners=True)
            
        concat_skip = torch.cat((skip_connection, x), dim=1)
        return self.double_conv(concat_skip)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        # Upsampling
        for feature in features[::-1]:
            self.ups.append(UpBlock(feature*2, feature))
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        
        for idx in range(len(self.ups)):
            skip_connection = skip_connections[idx]
            x = self.ups[idx](x, skip_connection)
            
        return self.final(x)
    
def test():
    x = torch.randn((1, 3, 161, 161))
    model = UNet(in_channels=3, out_channels=6)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    assert preds.shape[2:] == x.shape[2:]
    
if __name__ == "__main__":
    test()