import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BCEDiceLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, smooth):

        
        # Compute Binary Cross-Entropy Loss
        bce = F.binary_cross_entropy_with_logits(input, target)
        
        # Apply sigmoid to get probabilities
        input = torch.sigmoid(input)
        
        # Reshape tensors to (batch_size, -1)
        num = target.size(0)
        input = input.reshape(num, -1)
        target = target.reshape(num, -1)
        
        # Compute Dice Loss
        intersection = (input * target).sum(1)
        dice = (2. * intersection + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        
        # Combine BCE and Dice Loss
        return 0.5 * bce + dice