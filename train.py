import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from src.models.UNet import UNet
from src.models.UNetpp import UNetPlusPlus
from src.models.ResNetUNet_pt import get_model_and_optimizer, unfreeze_encoder
from src.models.ResNetUNet import ResNetUNet
from src.models.DeepLabV3p import DeepLabV3Plus

from src.datasets.dataset import PanNukeDataset
from src.utils.utils import save_checkpoint, load_checkpoint, check_accuracy, save_predictions_as_imgs
from torch.utils.data import DataLoader
from src.utils.losses import BinaryDiceLoss, BinaryCombinedLoss, FocalDiceLoss

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
MODEL = 'DeepLabV3+'
LOSS = 'FocalDiceLoss'

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

    if MODEL == 'UNet':
        model = UNet(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            features=FEATURES,
        ).to(DEVICE)
        load_checkpoint('checkpoints/UNet/checkpoint.pth', model)
    elif MODEL == 'UNet++':
        model = UNetPlusPlus(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            features=FEATURES,
            deep_supervision=DEEP_SUPERVISION
        ).to(DEVICE)
    elif MODEL == 'ResNetUNet_pt':
        model, optimizer = get_model_and_optimizer(device=DEVICE, out_channels=OUT_CHANNELS, learning_rate=LEARNING_RATE)
    elif MODEL == 'ResNetUNet':
        model = ResNetUNet(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            features=FEATURES,
        ).to(DEVICE)
    elif MODEL == 'DeepLabV3+':
        model = DeepLabV3Plus(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
        ).to(DEVICE)
    else:
        raise ValueError('Invalid model')
    
    if LOAD_MODEL:
        load_checkpoint(f'checkpoints/{MODEL}/checkpoint.pth', model)
    
    if MODEL != 'ResNetUNet_pt':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.01
        )
    
    dataset = PanNukeDataset(root_dir='data/raw/folds', fold=TRAIN_FOLD, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = PanNukeDataset(root_dir='data/raw/folds', fold=VAL_FOLD)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    if LOSS == 'FocalDiceLoss':
        loss_fn = FocalDiceLoss()
    elif LOSS == 'BinaryCombinedLoss':
        loss_fn = BinaryCombinedLoss()
    elif LOSS == 'BinaryDiceLoss':
        loss_fn = BinaryDiceLoss()
    else:
        raise ValueError('Invalid loss function')
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):

        if MODEL == 'ResNetUNet_pt':
            if unfreeze_encoder(model, epoch, unfreeze_epoch=3):
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

        train_fn(dataloader, model, optimizer, loss_fn, scaler)
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint, filename=f'checkpoints/{MODEL}/checkpoint_{epoch}.pth')
        
        # check_accuracy(val_dataloader, model, device=DEVICE)
        
        save_predictions_as_imgs(val_dataloader, model, epoch=epoch, folder=f'results/{MODEL}/', device=DEVICE)

if __name__ == '__main__':
    main()