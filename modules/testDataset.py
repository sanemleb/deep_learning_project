from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T

class TestCarDataset(Dataset):
    def __init__(self, mask_paths, transform=None):
        self.all_mask_paths = mask_paths
        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.all_mask_paths)

    def __getitem__(self, idx):
        
        mask_path = self.all_mask_paths[idx]
        
        mask = np.load(mask_path)
        img = mask[:, :, :3]
        mask_split = mask[:, :, 3]
        mask_split = mask_split//10
        mask_split = mask_split.astype(int)
        
        if self.transform is not None:
            img = self.transform(img)
        
        one_hot_encoded = np.eye(10, dtype=int)[mask_split.squeeze()]
        ms = torch.from_numpy(one_hot_encoded)
        
        img = img.to(torch.float32)
        reshaped_mask_tensor = ms.permute(2, 0, 1)
        reshaped_mask_tensor = reshaped_mask_tensor.to(torch.float32)

        return img, reshaped_mask_tensor 
    
    def calculate_mean_std(self):
        # Calculate mean and std
        num_channels = 3  # Assuming RGB images
        mean_accumulator = np.zeros(num_channels)
        std_accumulator = np.zeros(num_channels)

        for idx in range(len(self)):
            img, _ = self[idx]
            img_np = img.numpy()

            mean_accumulator += np.mean(img_np, axis=(1, 2))
            std_accumulator += np.std(img_np, axis=(1, 2))

        mean = mean_accumulator / len(self)
        std = std_accumulator / len(self)

        return mean, std