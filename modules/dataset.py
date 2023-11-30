import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import transforms as T
from modules.testDataset import TestCarDataset
from modules.settings import createCarPartsDictionary, DATA_PATH

class CarDataset(Dataset):
    def __init__(self, data_dir,transform=None):
        self.data_dir = data_dir
        # self.mean = mean
        # self.std = std
        self.transform = transform

        # getting the masks
        self.new_data = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays'))]
        self.mask_5door_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "black" in mask]
        self.mask_3door_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "orange" in mask]
        self.mask_photo_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "photo" in mask]
        self.mask_test_photo_paths = self.mask_photo_paths[:30]

        # combine mask paths into 1 object, but not test masks
        # self.all_mask_paths = self.mask_5door_paths + self.mask_3door_paths + self.mask_photo_paths[30:]
        # self.all_mask_paths = self.mask_5door_paths + self.mask_photo_paths[30:]
        self.all_mask_paths = self.new_data

 
    def __len__(self):
        return len(self.all_mask_paths)

    def __getitem__(self, idx):
        
        mask_path = self.all_mask_paths[idx]
        
        mask = np.load(mask_path)
        img = mask[:, :, :3]
        mask_split = mask[:, :, 3]
        # mask_split = mask_split//10
        mask_split = mask_split.astype(int)
        
        if self.transform is not None:
            img = self.transform(img)
            # print(img[0][:100])

        t = T.Compose([T.ToTensor()])
        img = t(img)
        img = img.to(torch.float32)

        ms = torch.from_numpy(mask_split).long()


        return img, ms 

    def create_test_dataset(self):
        test_mask_paths = self.mask_test_photo_paths

        test_dataset = TestCarDataset(test_mask_paths)
        return test_dataset
    
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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # import torch
    # # Create a tensor of shape (3,)
    # tensor = torch.tensor([1, 2, 3])
    # # Add a singleton dimension at index 0
    # tensor_unsqueezed = torch.unsqueeze(tensor, 0)

    # print("Original Tensor:", tensor)
    # print("Unsqueezed Tensor:", tensor_unsqueezed)
    # print(tensor.shape)
    # print(tensor_unsqueezed.shape)

    num_classes = 10
    MEAN=[0.485, 0.456, 0.406] # values from ImageNet
    STD=[0.229, 0.224, 0.225] # values from ImageNet

    carParts = createCarPartsDictionary()
    dataset = CarDataset(DATA_PATH, MEAN, STD)

    # mean, std = dataset.calculate_mean_std()
    # test = dataset.create_test_dataset()
    # mean, std= test.calculate_mean_std()

    # print("mean: ", mean)
    # print("std: ", std)

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
    