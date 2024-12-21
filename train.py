import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.models.UNet import UNet
from src.datasets.dataset import PanNukeDataset
from src.utils.utils import save_checkpoint, load_checkpoint, check_accuracy, save_predictions_as_imgs
from torch.utils.data import DataLoader

class DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, predictions, targets):
        # Input shapes: B x C x H x W
        batch_size = predictions.size(0)
        
        # Apply log_softmax for numerical stability
        pred_probs = F.softmax(predictions, dim=1)
        
        # Calculate Dice score for each sample and class
        intersection = (pred_probs * targets).sum(dim=(0, 2, 3))
        union = pred_probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        
        # Calculate Dice score
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Apply class weights if provided
        if self.weight is not None:
            dice_score = dice_score * self.weight.to(dice_score.device)
        
        # Average over classes
        dice_loss = 1 - dice_score.mean()
        
        return dice_loss

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.BCE = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(weight=weight)

    def forward(self, predictions, targets):
        BCE_loss = self.BCE(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        
        return self.alpha * BCE_loss + (1 - self.alpha) * dice_loss

LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 10
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
LOAD_MODEL = False
ROOT_DIR = 'data/raw/folds'
TRAIN_FOLD = 1
VAL_FOLD = 2

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, mask) in enumerate(loop):
        data = data.to(DEVICE)
        mask = mask.float().to(DEVICE)
        
        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = loss_fn(prediction, mask)
            
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
        model = UNet(in_channels=3, out_channels=1).to(DEVICE)
        load_checkpoint('checkpoints/checkpoint.pth', model)
    else:
        model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    
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
        train_fn(dataloader, model, optimizer, loss_fn, scaler)
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint, filename=f'checkpoints/checkpoint_{epoch}.pth')
        
        # check_accuracy(val_dataloader, model, device=DEVICE)
        
        save_predictions_as_imgs(val_dataloader, model, epoch=epoch, folder='results/', device=DEVICE)

if __name__ == '__main__':
    main()