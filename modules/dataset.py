import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import transforms as T
from testDataset import TestCarDataset
from settings import createCarPartsDictionary, DATA_PATH

class CarDataset(Dataset):
    def __init__(self, data_dir, mean, std,  transform=None):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.transform = transform

        # getting the masks
        self.mask_5door_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "black" in mask]
        self.mask_3door_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "orange" in mask]
        self.mask_photo_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "photo" in mask]
        self.mask_test_photo_paths = self.mask_photo_paths[:30]

        # combine mask paths into 1 object, but not test masks
        # self.all_mask_paths = self.mask_5door_paths + self.mask_3door_paths + self.mask_photo_paths[30:]
        self.all_mask_paths = self.mask_5door_paths + self.mask_photo_paths[30:]

 
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
            # print(img[0][:100])

        t = T.Compose([T.ToTensor()])
        t1 = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])

        img1 = t(img)
        img2 = t1(img)
        print(torch.unique(img1))        
        print(torch.unique(img2))

        one_hot_encoded = np.eye(10, dtype=int)[mask_split.squeeze()]
        ms = torch.from_numpy(one_hot_encoded)
        
        img = img.to(torch.float32)
        print(img[0][:5])
        reshaped_mask_tensor = ms.permute(2, 0, 1)
        reshaped_mask_tensor = reshaped_mask_tensor.to(torch.float32)

        return img, reshaped_mask_tensor 

    def create_test_dataset(self):
        test_mask_paths = self.mask_test_photo_paths

        test_dataset = TestCarDataset(test_mask_paths)
        return test_dataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10
    MEAN=[0.485, 0.456, 0.406]
    STD=[0.229, 0.224, 0.225]

    carParts = createCarPartsDictionary()
    dataset = CarDataset(DATA_PATH, MEAN, STD)

    # Test the __getitem__ function
    index = 0#random.randint(0, len(dataset))  # Change this to the index of the item you want to retrieve
    item = dataset.__getitem__(index)
    # Display the item (for testing purposes)
    img, mask = item  # Assuming the __getitem__ function returns an image and a mask

    print(img.shape)
    print(mask.shape)

    # # Plot the image and mask side by side
    # plt.figure(figsize=(10, 5))
    # # Original Image
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.title('Original Image')
    # plt.axis('off')
    # # Segmentation Mask
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask, cmap='viridis')  # Adjust the cmap as needed
    # plt.title('Segmentation Mask')
    # plt.axis('off')
    # plt.show()
    