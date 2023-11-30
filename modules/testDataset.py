from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TestCarDataset(Dataset):
    def __init__(self, mask_paths, transform=None):
        self.all_mask_paths = mask_paths
        self.transform = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast((0,0.5),(0,0.5)),
            A.GaussNoise(),
            ToTensorV2()  # Converts the image to a PyTorch tensor
        ])

    def __len__(self):
        return len(self.all_mask_paths)

    def __getitem__(self, idx):
        
        mask_path = self.all_mask_paths[idx]
        
        mask = np.load(mask_path)
        img = mask[:, :, :3]
        mask_split = mask[:, :, 3]
        mask_split = mask_split//10
        mask_split = mask_split.astype(int)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
        
        one_hot_encoded = np.eye(10, dtype=int)[mask_split.squeeze()]
        ms = torch.from_numpy(one_hot_encoded)
        
        img = img.to(torch.float32)
        reshaped_mask_tensor = ms.permute(2, 0, 1)
        reshaped_mask_tensor = reshaped_mask_tensor.to(torch.float32)

        return img, reshaped_mask_tensor 