import torch
import torch.nn as nn
import torch.nn.functional as F

# Downsampling Block for ResNet
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DownBlock, self).__init__()
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.resblock = ResBlock(in_channels, out_channels, stride, downsample)

    def forward(self, x):
        return self.resblock(x)

# Define a Basic Convolution Block (Conv3x3 -> BN -> ReLU)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Define a Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out += identity
        return self.relu(out)

# Define the Up Block for UNet Decoder
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        if x.shape != skip_connection.shape:
            x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x, skip_connection), dim=1)
        return self.double_conv(x)

# Define the UNet with ResNet50 Encoder
class ResNetUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(ResNetUNet, self).__init__()

        # Initial Convolution
        self.initial_conv = ConvBlock(in_channels, features[0])

        # Encoder
        self.encoders = nn.ModuleList()
        for idx, feature in enumerate(features):
            if idx == 0:
                self.encoders.append(ResBlock(features[idx], features[idx]))
            else:
                self.encoders.append(DownBlock(features[idx - 1], features[idx]))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(features[-1], features[-1] * 2),
            ConvBlock(features[-1] * 2, features[-1] * 2)
        )

        # Decoder
        self.decoders = nn.ModuleList()
        reversed_features = features[::-1]
        for idx, feature in enumerate(reversed_features):
            if idx == 0:
                self.decoders.append(UpBlock(features[-1] * 2, feature))
            else:
                self.decoders.append(UpBlock(reversed_features[idx - 1], feature))

        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        x = self.initial_conv(x)
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[idx])

        return self.final_conv(x)

# Testing the Model
def test():
    x = torch.randn((1, 3, 256, 256))  # Input image
    model = ResNetUNet(in_channels=3, out_channels=1)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    assert preds.shape[2:] == x.shape[2:], "Output shape must match input shape."

    print(model)

if __name__ == "__main__":
    test()

