import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetUNet_pt(nn.Module):
    def __init__(self, out_channels=1):
        super(ResNetUNet_pt, self).__init__()
        
        # Load pretrained ResNet50 with updated weights parameter
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )  # 64 channels
        self.pool = nn.Identity()
        
        self.encoder2 = resnet.layer1  # 256 channels
        self.encoder3 = resnet.layer2  # 512 channels
        self.encoder4 = resnet.layer3  # 1024 channels
        self.encoder5 = resnet.layer4  # 2048 channels
        
        # Decoder path with correct channel dimensions
        self.upconv5 = UpBlock(2048, 1024)            # Input: 2048, Skip: 1024 -> Output: 1024
        self.upconv4 = UpBlock(1024, 512)             # Input: 1024, Skip: 512 -> Output: 512
        self.upconv3 = UpBlock(512, 256)              # Input: 512, Skip: 256 -> Output: 256
        self.upconv2 = UpBlock(256, 64)               # Input: 256, Skip: 64 -> Output: 64
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in [self.upconv5, self.upconv4, self.upconv3, self.upconv2, self.final_conv]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder path
        x1 = self.encoder1(x)        
        x = self.pool(x1)            
        
        x2 = self.encoder2(x)        
        x3 = self.encoder3(x2)       
        x4 = self.encoder4(x3)       
        x5 = self.encoder5(x4)       
        
        # Decoder path
        d5 = self.upconv5(x5, x4)    
        d4 = self.upconv4(d5, x3)    
        d3 = self.upconv3(d4, x2)    
        d2 = self.upconv2(d3, x1)    
        
        output = self.final_conv(d2)

        output = F.interpolate(output, size=input_size, 
                          mode='bilinear', align_corners=True)
        
        return output

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super(UpBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # After concatenation, input channels will be out_channels * 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p)
        )
        
    def forward(self, x, skip_connection):
        # Upsample and reduce channels
        x = self.upsample(x)
        x = self.conv1x1(x)
        
        # Handle different spatial dimensions
        if x.shape[-2:] != skip_connection.shape[-2:]:
            x = F.interpolate(x, size=skip_connection.shape[-2:], 
                            mode='bilinear', align_corners=True)
        
        # Concatenate and process
        x = torch.cat([skip_connection, x], dim=1)
        return self.double_conv(x)

# Training configuration
def get_model_and_optimizer(device, out_channels=1, learning_rate=1e-4):
    model = ResNetUNet_pt(out_channels).to(device)
    
    # Freeze encoder initially
    for param in model.encoder1.parameters():
        param.requires_grad = False
    for param in model.encoder2.parameters():
        param.requires_grad = False
    for param in model.encoder3.parameters():
        param.requires_grad = False
    for param in model.encoder4.parameters():
        param.requires_grad = False
    for param in model.encoder5.parameters():
        param.requires_grad = False
    
    # Separate parameter groups for encoder and decoder
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': learning_rate/10},  # Lower LR for encoder
        {'params': decoder_params, 'lr': learning_rate}      # Higher LR for decoder
    ], weight_decay=0.01)
    
    return model, optimizer

def unfreeze_encoder(model, current_epoch, unfreeze_epoch=3):
    """Unfreeze encoder after specified number of epochs"""
    if current_epoch == unfreeze_epoch:
        for param in model.parameters():
            param.requires_grad = True
        print("Unfreezing encoder layers...")
        return True
    return False

def test():
    x = torch.randn((1, 3, 161, 161))
    model = ResNetUNet_pt(out_channels=1)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    assert preds.shape[2:] == x.shape[2:]

if __name__ == "__main__":
    test()