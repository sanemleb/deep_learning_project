from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T

class TestCarDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.all_img_paths = img_paths
        self.all_mask_paths = mask_paths
        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.all_img_paths)

    def __getitem__(self, idx):
        
        mask_path = self.all_mask_paths[idx]
        
        mask = np.load(mask_path).astype(np.double)
        img = mask[:, :, :3]
        mask_split = mask[:, :, 3]
        mask_split = mask_split//10
        mask_split = mask_split.astype(int)
        
        if self.transform is not None:
            img = self.transform(img)

        one_hot_encoded = np.eye(10)[mask.squeeze()]
        
        reshaped_image_tensor = img.permute(2, 0, 1)
        reshaped_image_tensor = reshaped_image_tensor.to(torch.float32)
        reshaped_mask_tensor = one_hot_encoded.permute(2, 0, 1)
        reshaped_mask_tensor = reshaped_mask_tensor.to(torch.float32)

        return reshaped_image_tensor, reshaped_mask_tensor 