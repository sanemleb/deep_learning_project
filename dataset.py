import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from settings import createCarPartsDictionary, DATA_PATH
from torchvision import transforms
from utlis import get_data_loaders

from testDataset import TestCarDataset

class CarDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        
        # getting the photos
        self.img_5door_paths_ns = [os.path.join(data_dir, 'images//black_5_doors//no_segmentation', img) for img in os.listdir(os.path.join(data_dir, 'images//black_5_doors//no_segmentation'))]
        self.img_5door_paths_s = [os.path.join(data_dir, 'images//black_5_doors//with_segmentation', img) for img in os.listdir(os.path.join(data_dir, 'images//black_5_doors//with_segmentation'))]
        self.img_3door_paths_ns = [os.path.join(data_dir, 'images//orange_3_doors//no_segmentation', img) for img in os.listdir(os.path.join(data_dir, 'images//orange_3_doors//no_segmentation'))]
        self.img_3door_paths_s = [os.path.join(data_dir, 'images//orange_3_doors//with_segmentation', img) for img in os.listdir(os.path.join(data_dir, 'images//orange_3_doors//with_segmentation'))]
        self.img_photo_paths_ns = [os.path.join(data_dir, 'images//photo//no_segmentation', img) for img in os.listdir(os.path.join(data_dir, 'images//photo//no_segmentation'))]
        self.img_photo_paths_s = [os.path.join(data_dir, 'images//photo//with_segmentation', img) for img in os.listdir(os.path.join(data_dir, 'images//photo//with_segmentation'))]
        self.img_test_photo_paths = self.img_photo_paths_ns[:30]

        # combine photo paths into 1 object, but not test photo paths
        self.all_img_paths = self.img_5door_paths_ns + self.img_3door_paths_ns + self.img_photo_paths_ns[30:]

        # getting the masks
        self.mask_5door_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "black" in mask]
        self.mask_3door_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "orange" in mask]
        self.mask_photo_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "photo" in mask]
        self.mask_test_photo_paths = self.mask_photo_paths[:30]

        # combine mask paths into 1 object, but not test masks
        self.all_mask_paths = self.mask_5door_paths + self.mask_3door_paths + self.mask_photo_paths[30:]
        
        self.transform = transform
 
    def __len__(self):
        return len(self.all_img_paths)

    def __getitem__(self, idx):

        img_path = self.all_img_paths[idx]
        mask_path = self.all_mask_paths[idx]

        img = np.array(Image.open(img_path))
        mask = np.load(mask_path).astype(np.double)
        print(mask.shape[2])

        # if self.transform:
            # img, mask = self.transform(img, mask)

        # Convert to PyTorch tensors
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return img, mask # here we will need to change to return a tensor(?), this formrat need to be accepted by datalaoder 
                         # possibly from_numpy() on its own is not enough

    def create_test_dataset(self):
        test_img_paths = self.img_test_photo_paths
        test_mask_paths = self.mask_test_photo_paths

        test_dataset = TestCarDataset(test_img_paths, test_mask_paths)
        return test_dataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 9
    carParts = createCarPartsDictionary()
    dataset = CarDataset(DATA_PATH)

    # Test the __getitem__ function
    index = random.randint(0, len(dataset))  # Change this to the index of the item you want to retrieve
    item = dataset.__getitem__(index)
    # Display the item (for testing purposes)
    img, mask = item  # Assuming the __getitem__ function returns an image and a mask

    # print(img.shape)
    # print(mask.shape)

    # Normalize the mask values to be in the range [0, 1]
    # normalized_mask = (mask - mask.min()) / (mask.max() - mask.min())

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
