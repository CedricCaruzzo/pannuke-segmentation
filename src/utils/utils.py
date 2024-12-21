import torch
import torchvision
import os

def save_checkpoint(state, filename='checkpoints/checkpoint.pth'):
    print("Saving checkpoint ...")
    torch.save(state, filename)
    
def load_checkpoint(filepath, model):
    print("Loading checkpoint ...")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    
def check_accuracy(loader, model, device="cuda"):
    """
    Calculate accuracy metrics for multi-class segmentation:
    - Per-class Dice score
    - Per-class IoU (Intersection over Union)
    - Per-class Precision
    - Per-class Recall
    """
    model.eval()
    
    # Initialize metrics for each class
    dice_scores = torch.zeros(6).to(device)  # 6 classes
    ious = torch.zeros(6).to(device)
    precisions = torch.zeros(6).to(device)
    recalls = torch.zeros(6).to(device)
    
    n_samples = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # Get predictions
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            # Calculate metrics for each class
            for class_idx in range(6):
                pred_class = preds[:, class_idx]
                target_class = y[:, class_idx]
                
                # Intersection and Union
                intersection = (pred_class * target_class).sum()
                union = pred_class.sum() + target_class.sum() - intersection
                
                # True Positives, False Positives, False Negatives
                tp = intersection
                fp = pred_class.sum() - tp
                fn = target_class.sum() - tp
                
                # Dice Score: 2*TP/(2*TP + FP + FN)
                dice = (2. * tp) / (2. * tp + fp + fn + 1e-8)
                dice_scores[class_idx] += dice
                
                # IoU: TP/(TP + FP + FN)
                iou = tp / (tp + fp + fn + 1e-8)
                ious[class_idx] += iou
                
                # Precision: TP/(TP + FP)
                precision = tp / (tp + fp + 1e-8)
                precisions[class_idx] += precision
                
                # Recall: TP/(TP + FN)
                recall = tp / (tp + fn + 1e-8)
                recalls[class_idx] += recall
            
            n_samples += 1
    
    # Average metrics over all samples
    dice_scores /= n_samples
    ious /= n_samples
    precisions /= n_samples
    recalls /= n_samples
    
    # Print results
    print("\nMetrics per class:")
    print("Class\t\tDice\t\tIoU\t\tPrecision\tRecall")
    print("-" * 65)
    
    class_names = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6']  # Replace with your actual class names
    
    for i in range(6):
        print(f"{class_names[i]:<12} {dice_scores[i]:.4f}\t{ious[i]:.4f}\t{precisions[i]:.4f}\t{recalls[i]:.4f}")
    
    # Print mean metrics
    print("-" * 65)
    print(f"Mean\t\t{dice_scores.mean():.4f}\t{ious.mean():.4f}\t{precisions.mean():.4f}\t{recalls.mean():.4f}")
    
    model.train()
    
    return {
        'dice_scores': dice_scores.cpu().numpy(),
        'ious': ious.cpu().numpy(),
        'precisions': precisions.cpu().numpy(),
        'recalls': recalls.cpu().numpy(),
        'mean_dice': dice_scores.mean().item(),
        'mean_iou': ious.mean().item(),
        'mean_precision': precisions.mean().item(),
        'mean_recall': recalls.mean().item()
    }
        
def save_predictions_as_imgs(loader, model, epoch, mod=50, folder='results/', device='cuda'):
    print('Saving predictions as images ...')
    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx%mod == 0:
            x = x.to(device=device)
            y = y.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            
            def make_folder(path):
                # Check if the folder exists, and if not, create it
                if not os.path.exists(path):
                    os.makedirs(path)

            
            # Save images
            make_folder(f"{folder}pred/epoch_{epoch}/")
            torchvision.utils.save_image(preds, f"{folder}pred/epoch_{epoch}/pred_{idx}.jpg")
            make_folder(f"{folder}true/epoch_{epoch}/")
            torchvision.utils.save_image(y, f"{folder}true/epoch_{epoch}/true_{idx}.jpg")
            make_folder(f"{folder}image/epoch_{epoch}/")
            torchvision.utils.save_image(x, f"{folder}image/epoch_{epoch}/image{idx}.jpg")
            
def test():
    from src.datasets.dataset import PanNukeDataset
    from torch.utils.data import DataLoader
    
    from src.models.UNet import UNet
    
    DEVICE = 'cuda'
    
    dataset = PanNukeDataset(root_dir='data/raw/folds', fold=2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = UNet(in_channels=3, out_channels=6).to(DEVICE)
    for epoch in range(1):
        for i, (image, mask) in enumerate(dataloader):
            #check_accuracy(dataloader, model, device=DEVICE)
            
            save_predictions_as_imgs(dataloader, model, epoch=epoch, folder='results/', device=DEVICE)
            
            break
    
if __name__ == '__main__':
    test()