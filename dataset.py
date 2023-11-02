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
        
        # getting the masks
        self.mask_5door_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "black" in mask]
        self.mask_3door_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "orange" in mask]
        self.mask_photo_paths = [os.path.join(data_dir,'arrays', mask) for mask in os.listdir(os.path.join(data_dir, 'arrays')) if "photo" in mask]
        print(self.mask_3door_paths[-1])
        print(len(self.mask_3door_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.path, name)
        # mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # if self.transform:
        #     img = self.transform(img)
        #     mask = self.transform(mask)

        return path#img, mask

if __name__ == '__main__':

    dir_path = r"C://Users//micha//Downloads//carseg_data//carseg_data"

    # Load the .npy file
    # file_path = dir_path+'//arrays//black_5_doors_0001.npy'
    # data = np.load(file_path)

    # # Display the content (assuming it's an image or a 2D array)
    # plt.imshow(data, cmap='viridis')  # Change the colormap to your preference
    # plt.title('Numpy .npy File Content')
    # plt.colorbar()  # Add a colorbar if needed
    # plt.show()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 9
    #car_parts = ['hood','front_door','rear door','frame','rear quater panel','trunk lid','fender','bumper','rest of car']
    carParts = createCarPartsDictionary()
    dataset = CarDataset(dir_path)
    #print(carParts['hood']['class_value'])
    #print(os.listdir(data_path))