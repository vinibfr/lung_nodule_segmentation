import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)
def dice_coef(output, target):
    smooth = 1e-5

    # Extract the first channel and keep the channel dimension
    output = output[:, 0:1, :, :]  # Shape: [batch_size, 1, height, width]
    target = target[:, 0:1, :, :]  # Shape: [batch_size, 1, height, width]

    # Apply sigmoid and flatten the tensors
    output = torch.sigmoid(output).reshape(-1).data.cpu().numpy()
    
    # Flatten the target tensor
    target = target.reshape(-1).data.cpu().numpy()
    
    # Compute the intersection
    intersection = (output * target).sum()
    
    # Calculate the Dice coefficient
    dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)
    
    return dice

def dice_coef_old(output, target):
    smooth = 1e-5

    # we need to use sigmoid because the output of Unet is logit.
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    print('output shape:',output.shape)
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def precision(output, target):
    smooth = 1e-5

    # we need to use sigmoid because the output of Unet is logit.
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    
    true_positive = (output * target).sum()
    
    return (true_positive + smooth) / (output.sum() + smooth)

def recall(output, target):
    smooth = 1e-5

    # we need to use sigmoid because the output of Unet is logit.
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    
    true_positive = (output * target).sum()
    
    return (true_positive + smooth) / (target.sum() + smooth)

def dice_coef2(output, target):
    "This metric is for validation purpose"
    smooth = 1e-5

    output = output.view(-1)
    output = (output>0.5).float().cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def dice_coef3(output, target):
    smooth = 1e-5
    
    # we need to use sigmoid because the output of Unet is logit.
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)