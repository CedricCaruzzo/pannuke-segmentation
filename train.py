import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from src.models.UNet import UNet
# from src.models.UNetpp import UNetPlusPlus  # Change this line to use UNet++ instead
from src.models.ResNetUNet_pt import get_model_and_optimizer, unfreeze_encoder
from src.models.ResNetUNet import ResNetUNet

from src.datasets.dataset import PanNukeDataset
from src.utils.utils import save_checkpoint, load_checkpoint, check_accuracy, save_predictions_as_imgs
from torch.utils.data import DataLoader

class DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, predictions, targets):
        # Handle both single predictions and deep supervision outputs
        if isinstance(predictions, list):
            total_loss = 0
            # Calculate loss for each deep supervision output
            for pred in predictions:
                pred_probs = F.softmax(pred, dim=1)
                intersection = (pred_probs * targets).sum(dim=(0, 2, 3))
                union = pred_probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
                dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
                if self.weight is not None:
                    dice_score = dice_score * self.weight.to(dice_score.device)
                total_loss += (1 - dice_score.mean())
            return total_loss / len(predictions)
        else:
            # Original single prediction logic
            pred_probs = F.softmax(predictions, dim=1)
            intersection = (pred_probs * targets).sum(dim=(0, 2, 3))
            union = pred_probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
            dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
            if self.weight is not None:
                dice_score = dice_score * self.weight.to(dice_score.device)
            return 1 - dice_score.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.BCE = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(weight=weight)

    def forward(self, predictions, targets):
        if isinstance(predictions, list):
            # Handle deep supervision outputs
            bce_loss = sum(self.BCE(pred, targets) for pred in predictions) / len(predictions)
            dice_loss = self.dice(predictions, targets)
        else:
            # Handle single prediction
            bce_loss = self.BCE(predictions, targets)
            dice_loss = self.dice(predictions, targets)
        
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
NUM_EPOCHS = 10
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IN_CHANNELS = 3
OUT_CHANNELS = 1
FEATURES = [64, 128, 256, 512]
LOAD_MODEL = False
ROOT_DIR = 'data/raw/folds'
TRAIN_FOLD = 1
VAL_FOLD = 2
DEEP_SUPERVISION = True  # parameter for UNet++

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, mask) in enumerate(loop):
        data = data.to(DEVICE)
        mask = mask.float().to(DEVICE)
        
        with torch.cuda.amp.autocast():
            predictions = model(data)  # May return list of predictions with deep supervision
            loss = loss_fn(predictions, mask)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Rotate(limit=360, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
        ]
    )
    
    if LOAD_MODEL:
        model = ResNetUNet(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            features=FEATURES,
            # deep_supervision=DEEP_SUPERVISION # Change this line to use UNet++ instead
        ).to(DEVICE)
        # model, optimizer = get_model_and_optimizer(device=DEVICE, out_channels=OUT_CHANNELS, learning_rate=LEARNING_RATE) # Change this line to use ResNetUNet instead
        load_checkpoint('checkpoints/UNet/checkpoint.pth', model)
    else:
        model = ResNetUNet(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            features=FEATURES,
            # deep_supervision=DEEP_SUPERVISION # Change this line to use UNet++ instead
        ).to(DEVICE)
        # model, optimizer = get_model_and_optimizer(device=DEVICE, out_channels=OUT_CHANNELS, learning_rate=LEARNING_RATE) # Change this line to use ResNetUNet instead
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    dataset = PanNukeDataset(root_dir='data/raw/folds', fold=TRAIN_FOLD, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = PanNukeDataset(root_dir='data/raw/folds', fold=VAL_FOLD)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    loss_fn = CombinedLoss(alpha=0.5)
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):

        # if unfreeze_encoder(model, epoch, unfreeze_epoch=3): # Change this line to use ResNetUNet instead
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1

        train_fn(dataloader, model, optimizer, loss_fn, scaler)
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint, filename=f'checkpoints/ResNetUNet/checkpoint_{epoch}.pth')
        
        # check_accuracy(val_dataloader, model, device=DEVICE)
        
        save_predictions_as_imgs(val_dataloader, model, epoch=epoch, folder='results/ResNetUNet/', device=DEVICE)

if __name__ == '__main__':
    main()