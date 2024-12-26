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

class NestedUpBlock(nn.Module):
    def __init__(self, prev_channels, skip_channels, out_channels):
        super(NestedUpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        total_in_channels = prev_channels + skip_channels
        self.conv = DoubleConv(total_in_channels, out_channels)
        
    def forward(self, prev_feature, skip_features):
        up_feature = self.up(prev_feature)
        
        # Ensure all features have the same spatial dimensions
        for i in range(len(skip_features)):
            if up_feature.shape[2:] != skip_features[i].shape[2:]:
                up_feature = F.interpolate(
                    up_feature, 
                    size=skip_features[i].shape[2:],
                    mode='bilinear', 
                    align_corners=True
                )
        
        concat_features = torch.cat([up_feature] + skip_features, dim=1)
        return self.conv(concat_features)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], deep_supervision=True):
        super(UNetPlusPlus, self).__init__()
        self.features = features
        self.deep_supervision = deep_supervision
        
        # Initial path
        self.encoder_blocks = nn.ModuleList()
        current_channels = in_channels
        
        # Create encoder blocks with correct channel dimensions
        for feature in features:
            self.encoder_blocks.append(DoubleConv(current_channels, feature))
            current_channels = feature
        
        # Nested convolution paths
        self.nested_convs = nn.ModuleDict()
        
        # Create nested paths
        for i in range(1, len(features)):  # depth
            for j in range(len(features) - i):  # level
                skip_channels = features[j] * i
                prev_channels = features[j + 1]
                
                self.nested_convs[f'up_{i}_{j}'] = NestedUpBlock(
                    prev_channels=prev_channels,
                    skip_channels=skip_channels,
                    out_channels=features[j]
                )
        
        # Final convolution layers for deep supervision
        self.final_convs = nn.ModuleList([
            nn.Conv2d(features[0], out_channels, kernel_size=1)
            for _ in range(len(features))
        ])
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Store feature maps
        feature_maps = {}
        outputs = []
        
        # Encoder path
        current_input = x
        for i, encoder in enumerate(self.encoder_blocks):
            current_output = encoder(current_input)
            feature_maps[f'0_{i}'] = current_output
            if i < len(self.encoder_blocks) - 1:  # Don't pool after last encoder block
                current_input = self.pool(current_output)
        
        # Nested paths
        for i in range(1, len(self.features)):  # depth
            for j in range(len(self.features) - i):  # level
                # Collect skip connections
                skip_features = [feature_maps[f'{k}_{j}'] for k in range(i)]
                prev_feature = feature_maps[f'{i-1}_{j+1}']
                
                # Compute current node
                curr_feature = self.nested_convs[f'up_{i}_{j}'](prev_feature, skip_features)
                feature_maps[f'{i}_{j}'] = curr_feature
                
                # Add deep supervision output
                if j == 0 and self.deep_supervision:
                    outputs.append(self.final_convs[i](curr_feature))
        
        # Final output
        final_output = self.final_convs[0](feature_maps[f'{len(self.features)-1}_0'])
        
        if self.deep_supervision and self.training:
            outputs.append(final_output)
            return outputs
        return final_output

def test():
    x = torch.randn((1, 3, 160, 160))
    model = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=False)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    if isinstance(preds, list):
        print("Deep supervision outputs:")
        for i, pred in enumerate(preds):
            print(f"Output {i+1} shape: {pred.shape}")
    else:
        print(f"Output shape: {preds.shape}")
    
if __name__ == "__main__":
    test()