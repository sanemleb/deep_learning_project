from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image

class TestCarDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        # self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = np.array(Image.open(img_path))
        mask = np.load(mask_path).astype(np.float32)

        # if self.transform:
        #     img = self.transform(img)
        #     mask = self.transform(mask)

        mask_split = mask[:, :, 3]
        mask_split = mask_split//10
        
        masks = mask_split.int()

        masks = masks.to(torch.int64)

        one_hot_encoding = torch.nn.functional.one_hot(masks.view(-1), num_classes=10)
        one_hot_encoding = one_hot_encoding.view(masks.shape[0], masks.shape[1], 10)
        
        reshaped_image_tensor = img.permute(2, 0, 1)
        reshaped_image_tensor = reshaped_image_tensor.to(torch.float32)
        reshaped_mask_tensor = one_hot_encoding.permute(2, 0, 1)
        reshaped_mask_tensor = reshaped_mask_tensor.to(torch.float32)
                
        return reshaped_image_tensor, reshaped_mask_tensor # here we will need to change to return a tensor(?), this formrat need to be accepted by datalaoder 
                         # possibly from_numpy() on its own is not enough
