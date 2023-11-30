import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from testDataset import TestCarDataset
from settings import createCarPartsDictionary, DATA_PATH
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import ToPILImage

class CarDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # getting the masks
        self.mask_5door_paths = [os.path.join(data_dir, 'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "black" in mask]
        self.mask_photo_paths = [os.path.join(data_dir, 'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "photo" in mask]
        self.mask_test_photo_paths = self.mask_photo_paths[:30]

        # combine mask paths into 1 object, but not test masks
        self.all_mask_paths = self.mask_5door_paths + self.mask_photo_paths[30:]

        self.augmentation = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            ToTensorV2(),  # Converts the image to a PyTorch tensor
        ], additional_targets={ 'image' : 'mask'})

    def __len__(self):
        return len(self.all_mask_paths)

    def __getitem__(self, idx):
        mask_path = self.all_mask_paths[idx]

        mask = np.load(mask_path)
        img = mask[:, :, :3].astype('uint8')
        mask_split = mask[:, :, 3]
        mask_split = mask_split // 10
        mask_split = mask_split.astype(int)

        # Visualize the original and augmented images along with their masks
        to_pil = ToPILImage()

        imggg = to_pil(img)
        imggg.show()
        
        horizontal = A.Compose([
            A.HorizontalFlip(),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast((0,0.5),(0,0.5)),
            ToTensorV2(),  # Converts the image to a PyTorch tensor
        ], additional_targets={ 'image' : 'mask'})
        
        vertical = A.Compose([
            A.VerticalFlip(),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast((0,0.5),(0,0.5)),
            ToTensorV2(),  # Converts the image to a PyTorch tensor
        ], additional_targets={ 'image' : 'mask'})
        
        transformed_horiz = horizontal(image=img, mask=mask_split)
        transformed_vertical = vertical(image=img, mask=mask_split)

        imh = transformed_horiz['image']
        imv = transformed_vertical['image']
        immh = to_pil(imh.permute(2,0,1))
        immh.show()
        immv = to_pil(imv.permute(2,0,1))
        immv.show()

        # return transformed['image'], transformed['mask']

    def create_test_dataset(self):
        test_mask_paths = self.mask_test_photo_paths
        test_dataset = TestCarDataset(test_mask_paths)
        return test_dataset


if __name__ == '__main__':
    dataset = CarDataset(DATA_PATH)
    img, mask = dataset[30]  # Get transformed image and mask
