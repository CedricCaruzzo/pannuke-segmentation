import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from dataset import PanNukeDataset
from model import UNet
import os

def visualize_prediction(image, true_mask, pred_mask, save_path=None):
    # Convert tensors to numpy arrays and move channel dimension to the end
    image = image.cpu().numpy().transpose(1, 2, 0)
    true_mask = true_mask.cpu().numpy().transpose(1, 2, 0)
    pred_mask = pred_mask.cpu().numpy().transpose(1, 2, 0)
    
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot true mask (combine all channels using different colors)
    true_mask_colored = np.zeros((*true_mask.shape[:2], 3))
    colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]  # Different colors for each class
    for i in range(true_mask.shape[-1]):
        for j in range(3):
            true_mask_colored[:,:,j] += true_mask[:,:,i] * colors[i][j]
    true_mask_colored = np.clip(true_mask_colored, 0, 1)
    
    axes[1].imshow(true_mask_colored)
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Plot predicted mask
    pred_mask_colored = np.zeros((*pred_mask.shape[:2], 3))
    for i in range(pred_mask.shape[-1]):
        for j in range(3):
            pred_mask_colored[:,:,j] += pred_mask[:,:,i] * colors[i][j]
    pred_mask_colored = np.clip(pred_mask_colored, 0, 1)
    
    axes[2].imshow(pred_mask_colored)
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def inference(model_path, dataset, device, num_samples=5):
    # Load model
    model = UNet(in_channels=3, out_channels=6)
    
    # Load checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create output directory
    os.makedirs('predictions', exist_ok=True)
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            image = image.to(device)
            mask = mask.to(device)
            
            # Get prediction
            output = model(image)
            #pred_mask = torch.sigmoid(output)  # Apply sigmoid for binary mask
            
            # Visualize and save results
            visualize_prediction(
                image[0],
                mask[0],
                output[0],
                save_path=f'predictions/prediction_{i+1}.png'
            )

def main():
    # Settings
    model_path = 'checkpoints/best_model_epoch_1.pth'
    root_dir = "data/raw/folds/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load validation dataset
    val_dataset = PanNukeDataset(root_dir=root_dir, fold=2)
    
    # Run inference
    inference(model_path, val_dataset, device)
    
    print("Inference completed. Check the 'predictions' folder for results.")

if __name__ == "__main__":
    main()