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

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return img, mask 