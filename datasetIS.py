import os
import numpy as np
import glob

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import torchvision
from torchvision import transforms

import albumentations as albu
from albumentations.pytorch import ToTensorV2

class MyLidcDataset(Dataset):
    def __init__(self, IMAGES_PATHS, MASK_PATHS,CROP_SIZE=512,Albumentation=False,sep=False):
        """
        IMAGES_PATHS: list of images paths ['./Images/0001_01_images.npy','./Images/0001_02_images.npy']
        MASKS_PATHS: list of masks paths ['./Masks/0001_01_masks.npy','./Masks/0001_02_masks.npy']
        """
        self.image_paths = IMAGES_PATHS
        self.mask_paths= MASK_PATHS
        self.albumentation = Albumentation
        self.crop_size = CROP_SIZE
        self.mask_shape = True
        self.sep = sep
        self.albu_transformations = albu.Compose([
            albu.Affine(shear=[-15,15], p=1,translate_px=(-15, 15)),
            ToTensorV2()
        ])
        self.transformations = transforms.Compose([transforms.ToTensor()])
    
    def transform(self, image, mask,index):
        #Transform to tensor
        if self.albumentation:
            #It is always best to convert the make input to 3 dimensional for albumentation

            if image.shape[0] == 2:
                image = image.transpose(1,2,0)
            elif image.shape[0] == 20:
                image = image.transpose(0,2,3,1)

            if mask.shape[0] == 2:
                mask = mask.transpose(1,2,0)
            elif mask.shape[0] == 20:
                mask = mask.transpose(0,2,3,1)

            # Without this conversion of datatype, it results in cv2 error. Seems like a bug
            mask = mask.astype('uint8')
            augmented=  self.albu_transformations(image=image,mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask= mask.permute(2,1,0)
            if self.sep:
                mask = mask[0]
                mask = mask.unsqueeze(0)                    

        else:
            image = self.transformations(image)
            mask = self.transformations(mask)
            image = image.permute(1,2,0)
            mask = mask.permute(1,2,0)     
            if self.sep:
                mask = mask[0]
                mask = mask.unsqueeze(0)             
        image,mask = image.type(torch.FloatTensor), mask.type(torch.FloatTensor)
        return image,mask

    def __getitem__(self, index):
        image = np.load(self.image_paths[index])
        mask = np.load(self.mask_paths[index])        
        image,mask = self.transform(image,mask,index)
        return image,mask

    def __len__(self):
        return len(self.image_paths)
