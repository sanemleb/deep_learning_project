import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
#from setting import data_path, car_parts, num_classes, device

def createCarPartsDictionary():
    carParts = {
        "hood": {
            "color": "orange",
            "class_value": 10
        },
        "front door": {
            "color": "dark green",
            "class_value": 20
        },
        "rear door": {
            "color": "yellow",
            "class_value": 30
        },
        "frame": {
            "color": "cyan",
            "class_value": 40
        },
        "rear quarter panel": {
            "color": "purple",
            "class_value": 50
        },
        "trunk lid": {
            "color": "light green",
            "class_value": 60
        },
        "fender": {
            "color": "blue",
            "class_value": 70
        },
        "bumper": {
            "color": "pink",
            "class_value": 80
        },
        "rest of car": {
            "color": "no color",
            "class_value": 90
        }
    }
    return carParts

class CarDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        
        # getting the photos
        self.img_5door_paths_ns = [os.path.join(data_dir, 'images//black_5_doors//no_segmentation', img) for img in os.listdir(os.path.join(data_dir, 'images//black_5_doors//no_segmentation'))]
        self.img_5door_paths_s = [os.path.join(data_dir, 'images//black_5_doors//with_segmentation', img) for img in os.listdir(os.path.join(data_dir, 'images//black_5_doors//with_segmentation'))]
        self.img_3door_paths_ns = [os.path.join(data_dir, 'images//orange_3_doors//no_segmentation', img) for img in os.listdir(os.path.join(data_dir, 'images//black_5_doors//no_segmentation'))]
        self.img_3door_paths_s = [os.path.join(data_dir, 'images//orange_3_doors//with_segmentation', img) for img in os.listdir(os.path.join(data_dir, 'images//orange_3_doors//with_segmentation'))]
        self.img_photo_paths_ns = [os.path.join(data_dir, 'images//photo//no_segmentation', img) for img in os.listdir(os.path.join(data_dir, 'images//photo//no_segmentation'))]
        self.img_photo_paths_s = [os.path.join(data_dir, 'images//photo//with_segmentation', img) for img in os.listdir(os.path.join(data_dir, 'images//photo//with_segmentation'))]
        # print((self.img_photo_paths_ns[29]))
        self.all_img_paths = self.img_5door_paths_ns + self.img_3door_paths_ns + self.img_photo_paths_ns[29:]

        # getting the masks
        self.mask_5door_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "black" in mask]
        self.mask_3door_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "orange" in mask]
        self.mask_photo_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "photo" in mask]

        self.all_mask_paths = self.mask_5door_paths + self.mask_3door_paths + self.mask_photo_paths[29:]

    def __len__(self):
        return len(self.all_img_paths)

    def __getitem__(self, idx):
        #print(self.all_img_paths)
        img_path = self.all_img_paths[idx]
        mask_path = self.all_mask_paths[idx]

        img = np.array(Image.open(img_path))
        mask = np.load(mask_path) # here possible conversion to float needed?

        # mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # if self.transform:
        #     img = self.transform(img)
        #     mask = self.transform(mask)

        return img, mask

if __name__ == '__main__':
    dir_path = r"C://Users//micha//Downloads//carseg_data//carseg_data"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 9
    carParts = createCarPartsDictionary()
    dataset = CarDataset(dir_path)

    # Test the __getitem__ function
    # index = -1  # Change this to the index of the item you want to retrieve
    # item = dataset.__getitem__(index)
    # # Display the item (for testing purposes)
    # img, mask = item  # Assuming the __getitem__ function returns an image and a mask
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.title('Image')
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask, cmap='viridis')
    # plt.title('Mask')
    # plt.show()
