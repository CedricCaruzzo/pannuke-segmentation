import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=False)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        return self.relu(x)

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, backbone=resnet50, aspp_out_channels=256):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = backbone(pretrained=True)

        # Extractor layers (low-level features)
        self.low_level_features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1
        )

        # High-level features (encoder output)
        self.encoder = nn.Sequential(
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )

        # ASPP module
        self.aspp = ASPP(in_channels=2048, out_channels=aspp_out_channels)

        # Decoder
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_out_channels + 48, aspp_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_out_channels),
            nn.ReLU(),
            nn.Conv2d(aspp_out_channels, aspp_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_out_channels),
            nn.ReLU(),
            nn.Conv2d(aspp_out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        low_level_features = self.low_level_features(x)
        x = self.encoder(low_level_features)
        x = self.aspp(x)
        
        low_level_features = self.low_level_projection(low_level_features)
        x = F.interpolate(x, size=low_level_features.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)

        x = self.decoder(x)
        x = F.interpolate(x, size=(x.shape[2] * 4, x.shape[3] * 4), mode='bilinear', align_corners=True)
        return x

# Test function
def test():
    x = torch.randn((1, 3, 256, 256))
    model = DeepLabV3Plus(in_channels=3, out_channels=1).eval()
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    assert preds.shape[2:] == x.shape[2:]

if __name__ == "__main__":
    test()
