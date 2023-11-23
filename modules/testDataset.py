from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image

class TestCarDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.all_img_paths = img_paths
        self.all_mask_paths = mask_paths
        # self.transform = transform

    def __len__(self):
        return len(self.all_img_paths)

    def __getitem__(self, idx):
        img_path = self.all_img_paths[idx]
        mask_path = self.all_mask_paths[idx]

        img = np.array(Image.open(img_path))
        mask = np.load(mask_path).astype(np.double)

        # Convert to PyTorch tensors
        mask_split = mask[:, :, 3]
        mask_split = mask_split//10
        mask_split = mask_split.astype(int)

        # Convert to PyTorch tensors
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask_split)

        # one_hot_encoding = torch.nn.functional.one_hot(mask.view(-1), num_classes=10)
        # one_hot_encoding = one_hot_encoding.view(mask.shape[0], mask.shape[1], 10)
        
        reshaped_image_tensor = img.permute(2, 0, 1)
        reshaped_image_tensor = reshaped_image_tensor.to(torch.float32)
        reshaped_mask_tensor = mask.to(torch.long)

        return reshaped_image_tensor, reshaped_mask_tensor # here we will need to change to return a tensor(?), this formrat need to be accepted by datalaoder 
                         # possibly from_numpy() on its own is not enough
