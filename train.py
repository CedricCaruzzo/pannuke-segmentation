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

def calculate_class_weights(dataset):
    """
    Calculate class weights for a dataset where masks have shape (B, C, H, W)
    Returns tensor of shape (C,) with weights for each class
    """
    all_masks = []
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Accumulate all masks
    for _, mask in tqdm(dataloader, desc="Calculating class weights"):
        all_masks.append(mask)
    
    # Concatenate all masks along batch dimension
    all_masks = torch.cat(all_masks, dim=0)  # Shape: (N, C, H, W)
    
    # Calculate number of pixels per class
    class_samples = all_masks.sum((0, 2, 3))  # Sum over batch, height, width
    
    # Calculate weights inversely proportional to class frequency
    total_samples = class_samples.sum()
    weights = total_samples / (len(dataset) * class_samples)
    
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    
    return weights

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

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, predictions, targets):
        # Ensure all tensors are on the same device as predictions
        device = predictions.device
        
        # Apply log_softmax for numerical stability
        log_prob = F.log_softmax(predictions, dim=1)
        prob = torch.exp(log_prob)
        
        # Calculate focal loss
        focal_weight = (1 - prob) ** self.gamma
        
        # Convert targets to class indices
        target_indices = targets.argmax(dim=1)
        
        # Ensure target indices and weight are on the same device as predictions
        target_indices = target_indices.to(device)
        if self.weight is not None:
            self.weight = self.weight.to(device)
        
        # Compute the loss
        loss = F.nll_loss(
            (focal_weight * log_prob),
            target_indices,
            weight=self.weight,
            reduction='mean'
        )
        
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.5, gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.focal = FocalLoss(weight=weight, gamma=gamma)
        self.dice = DiceLoss(weight=weight)

    def forward(self, predictions, targets):
        focal_loss = self.focal(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        
        return self.alpha * focal_loss + (1 - self.alpha) * dice_loss

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
    
    valid_transform = A.Compose(
        [
            ToTensorV2()
        ]
    )
    
    if LOAD_MODEL:
        model = UNet(in_channels=3, out_channels=6).to(DEVICE)
        load_checkpoint('checkpoints/checkpoint.pth', model)
    else:
        model = UNet(in_channels=3, out_channels=6).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    dataset = PanNukeDataset(root_dir='data/raw/folds', fold=TRAIN_FOLD, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = PanNukeDataset(root_dir='data/raw/folds', fold=VAL_FOLD)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    weights = calculate_class_weights(dataset)    
    loss_fn = CombinedLoss(
        weight=weights,
        alpha=0.5,
        gamma=2.0
    )
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(dataloader, model, optimizer, loss_fn, scaler)
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint, filename=f'checkpoints/checkpoint_{epoch}.pth')
        
        check_accuracy(val_dataloader, model, device=DEVICE)
        
        save_predictions_as_imgs(val_dataloader, model, epoch=epoch, folder='results/', device=DEVICE)

if __name__ == '__main__':
    main()