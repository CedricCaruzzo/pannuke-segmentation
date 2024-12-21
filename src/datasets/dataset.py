import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PanNukeDataset(Dataset):
    def __init__(self, root_dir, fold=1, transform=None):
        """
        Args:
            root_dir (str): Path to dataset directory.
            fold (int): Fold number (1, 2, or 3).
            transform (callable, optional): Optional transform to apply.
        """
        self.images_path = os.path.join(root_dir, f"Fold {fold}/images/fold{fold}/images.npy")
        self.masks_path = os.path.join(root_dir, f"Fold {fold}/masks/fold{fold}/masks.npy")
        self.types_path = os.path.join(root_dir, f"Fold {fold}/images/fold{fold}/types.npy")

        self.images = np.load(self.images_path, mmap_mode='r')  # N x W x H x C
        self.masks = np.load(self.masks_path, mmap_mode='r')    # N x W x H x 6
        self.types = np.load(self.types_path, mmap_mode='r')    # N tissue types

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()
        
        # Binary mask for segmentation
        collapsed_array = np.max(mask, axis=2)
        binary_array = np.where(collapsed_array > 1, 1, 0)
        mask = binary_array
        
        # Normalize image to [0, 1] range and ensure float32
        image = (image / 255).astype(np.float32)
        mask = mask.astype(np.float32)
        
        if self.transform:
            # Apply transforms with named arguments
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].unsqueeze(0)
            
        else:
            # Convert to torch tensors if no transform
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).unsqueeze(0)
        # Ensure float32 type for tensors
        image = image.float()
        mask = mask.float()
            
        return image, mask
    
def test():
    from torch.utils.data import DataLoader
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    import matplotlib.pyplot as plt
    
    train_transform = A.Compose(
        [
            A.Rotate(limit=360, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
        ]
    )
    
    dataset = PanNukeDataset(root_dir='data/raw/folds', fold=1, transform=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, (image, mask) in enumerate(dataloader):
        print(f'Image input shape: {image.shape}')
        print(f'Image input type: {image.dtype}')
        print(f'Image input min: {image.min()}\tImage input max: {image.max()}')
        
        print(f'Mask input shape: {mask.shape}')
        print(f'Mask input type: {mask.dtype}')
        print(f'Mask input min: {mask.min()}\tMask input max: {mask.max()}')

        plt.imshow(image.squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()

        plt.imshow(mask.squeeze(0).permute(1, 2, 0), cmap='gray')
        plt.show()
        
        break
    
    from src.models.UNet import UNet
    
    model = UNet(in_channels=3, out_channels=1)
    model = model.float()  # Ensure model is in float32
    preds = model(image)
    print(f"Input shape: {image.shape}")
    print(f"Output shape: {preds.shape}")
    assert preds.shape[2:] == image.shape[2:]
    plt.imshow(mask, cmap='gray')
    plt.show()
    
if __name__ == '__main__':
    test()