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
    def __init__(self, IMAGES_PATHS, MASK_PATHS,compose,CROP_SIZE=512,WIDTH_SIZE=804,Albumentation=False):
        """
        IMAGES_PATHS: list of images paths ['./Images/0001_01_images.npy','./Images/0001_02_images.npy']
        MASKS_PATHS: list of masks paths ['./Masks/0001_01_masks.npy','./Masks/0001_02_masks.npy']
        """
        self.image_paths = IMAGES_PATHS
        self.mask_paths= MASK_PATHS
        self.albumentation = Albumentation
        self.crop_size = CROP_SIZE
        self.albu_transformations = compose
        self.WIDTH_SIZE = WIDTH_SIZE
        self.transformations = transforms.Compose([transforms.ToTensor()])
    
    def transform(self, image, mask,index):
        #Transform to tensor
        if self.albumentation:
            #It is always best to convert the make input to 3 dimensional for albumentation
            try:
                image = image.reshape(self.crop_size,self.WIDTH_SIZE,1)
            except:
                print(image.shape)
                image = image.reshape(self.crop_size,self.WIDTH_SIZE,1)
            mask = mask.reshape(self.crop_size,self.WIDTH_SIZE,1)
            # Without this conversion of datatype, it results in cv2 error. Seems like a bug
            mask = mask.astype('uint8')
            augmented=  self.albu_transformations(image=image,mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask= mask.reshape([1,self.crop_size,self.WIDTH_SIZE])
        else:
            image = self.transformations(image)
            mask = self.transformations(mask)

        image,mask = image.type(torch.FloatTensor), mask.type(torch.FloatTensor)

        return image,mask

    def __getitem__(self, index):
        image = np.load(self.image_paths[index])
        mask = np.load(self.mask_paths[index])        
        image,mask = self.transform(image,mask,index)
        return image,mask

    def __len__(self):
        return len(self.image_paths)
