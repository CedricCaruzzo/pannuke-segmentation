import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        Binary Dice Loss for semantic segmentation.
        Args:
            smooth (float): Small value to avoid division by zero.
        """
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Forward pass for Binary Dice Loss.
        Args:
            predictions (Tensor or list of Tensors): Model predictions (logits).
            targets (Tensor): Ground truth labels (binary).
        Returns:
            Loss (Tensor): Dice loss value.
        """
        if isinstance(predictions, list):
            # Handle deep supervision
            total_loss = 0
            for pred in predictions:
                total_loss += self._compute_dice_loss(pred, targets)
            return total_loss / len(predictions)
        else:
            # Single prediction
            return self._compute_dice_loss(predictions, targets)

    def _compute_dice_loss(self, predictions, targets):
        predictions = torch.sigmoid(predictions)  # Apply sigmoid for binary logits
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        """
        Binary Focal Loss for addressing class imbalance.
        Args:
            alpha (float): Weighting factor for positive class.
            gamma (float): Focusing parameter to reduce loss for well-classified examples.
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        """
        Forward pass for Binary Focal Loss.
        Args:
            predictions (Tensor): Model predictions (logits).
            targets (Tensor): Ground truth labels (binary).
        Returns:
            Loss (Tensor): Focal loss value.
        """
        predictions = torch.sigmoid(predictions)  # Apply sigmoid for binary logits
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Compute the focal loss components
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


class BinaryCombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-5):
        """
        Combined BCE + Dice Loss for semantic segmentation.
        Args:
            alpha (float): Weight for BCE loss. (1-alpha) is the weight for Dice loss.
            smooth (float): Small value to avoid division by zero in Dice loss.
        """
        super(BinaryCombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss(smooth=smooth)

    def forward(self, predictions, targets):
        """
        Forward pass for Combined Loss.
        Args:
            predictions (Tensor or list of Tensors): Model predictions (logits).
            targets (Tensor): Ground truth labels (binary).
        Returns:
            Loss (Tensor): Combined loss value.
        """
        if isinstance(predictions, list):
            # Handle deep supervision
            bce_loss = sum(self.bce(pred, targets) for pred in predictions) / len(predictions)
            dice_loss = self.dice(predictions, targets)
        else:
            # Single prediction
            bce_loss = self.bce(predictions, targets)
            dice_loss = self.dice(predictions, targets)

        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, focal_alpha=0.25, smooth=1e-5):
        """
        Combined Focal + Dice Loss for semantic segmentation.
        Args:
            alpha (float): Weight for Focal loss. (1-alpha) is the weight for Dice loss.
            gamma (float): Focusing parameter for Focal loss.
            focal_alpha (float): Weighting factor for positive class in Focal loss.
            smooth (float): Small value to avoid division by zero in Dice loss.
        """
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.focal = BinaryFocalLoss(gamma=gamma, alpha=focal_alpha)
        self.dice = BinaryDiceLoss(smooth=smooth)

    def forward(self, predictions, targets):
        """
        Forward pass for Focal + Dice Loss.
        Args:
            predictions (Tensor or list of Tensors): Model predictions (logits).
            targets (Tensor): Ground truth labels (binary).
        Returns:
            Loss (Tensor): Combined loss value.
        """
        if isinstance(predictions, list):
            # Handle deep supervision
            focal_loss = sum(self.focal(pred, targets) for pred in predictions) / len(predictions)
            dice_loss = self.dice(predictions, targets)
        else:
            # Single prediction
            focal_loss = self.focal(predictions, targets)
            dice_loss = self.dice(predictions, targets)

        return self.alpha * focal_loss + (1 - self.alpha) * dice_loss
