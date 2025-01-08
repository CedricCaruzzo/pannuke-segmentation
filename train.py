import argparse
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


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, mask) in enumerate(loop):
        data = data.to(args.device)
        mask = mask.float().to(args.device)

        with torch.cuda.amp.autocast():
            predictions = model(data)  # May return list of predictions with deep supervision
            loss = loss_fn(predictions, mask)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main(args):
    DEVICE = args.device
    train_transform = A.Compose(
        [
            A.Rotate(limit=360, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
        ]
    )

    if args.model == 'UNet':
        model = UNet(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            features=args.features,
        ).to(DEVICE)
        if args.load_model:
            load_checkpoint(f'checkpoints/UNet/checkpoint.pth', model)
    elif args.model == 'UNet++':
        model = UNetPlusPlus(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            features=args.features,
            deep_supervision=args.deep_supervision
        ).to(DEVICE)
    elif args.model == 'ResNetUNet_pt':
        model, optimizer = get_model_and_optimizer(device=DEVICE, out_channels=args.out_channels, learning_rate=args.learning_rate)
    elif args.model == 'ResNetUNet':
        model = ResNetUNet(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            features=args.features,
        ).to(DEVICE)
    elif args.model == 'DeepLabV3+':
        model = DeepLabV3Plus(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
        ).to(DEVICE)
    else:
        raise ValueError('Invalid model')

    if args.load_model and args.model != 'UNet':
        load_checkpoint(f'checkpoints/{args.model}/checkpoint.pth', model)

    if args.model != 'ResNetUNet_pt':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01
        )

    dataset = PanNukeDataset(root_dir=args.root_dir, fold=args.train_fold, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = PanNukeDataset(root_dir=args.root_dir, fold=args.val_fold)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    if args.loss == 'FocalDiceLoss':
        loss_fn = FocalDiceLoss()
    elif args.loss == 'BinaryCombinedLoss':
        loss_fn = BinaryCombinedLoss()
    elif args.loss == 'BinaryDiceLoss':
        loss_fn = BinaryDiceLoss()
    else:
        raise ValueError('Invalid loss function')

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.num_epochs):

        if args.model == 'ResNetUNet_pt':
            if unfreeze_encoder(model, epoch, unfreeze_epoch=3):
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

        train_fn(dataloader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint, filename=f'checkpoints/{args.model}/checkpoint_{epoch}.pth')

        # Uncomment the next line if you want to check accuracy
        # check_accuracy(val_dataloader, model, device=DEVICE)

        save_predictions_as_imgs(val_dataloader, model, epoch=epoch, folder=f'results/{args.model}/', device=DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--model', type=str, default='DeepLabV3+', help='Model to use (e.g., UNet, UNet++, ResNetUNet, DeepLabV3+)')
    parser.add_argument('--loss', type=str, default='FocalDiceLoss', help='Loss function to use')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='Number of output channels')
    parser.add_argument('--features', type=list, default=[64, 128, 256, 512], help='Features for the model')
    parser.add_argument('--load_model', type=bool, default=False, help='Load a pre-trained model')
    parser.add_argument('--root_dir', type=str, default='data/raw/folds', help='Root directory of the dataset')
    parser.add_argument('--train_fold', type=int, default=1, help='Train fold index')
    parser.add_argument('--val_fold', type=int, default=2, help='Validation fold index')
    parser.add_argument('--deep_supervision', type=bool, default=True, help='Enable deep supervision for UNet++')

    args = parser.parse_args()
    main(args)
