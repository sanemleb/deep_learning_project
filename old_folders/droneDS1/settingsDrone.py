import os
import torch


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

# data settings
car_parts = ['hood', 'front door', 'rear door', 'frame', 'rear quarter panel', 'trunk lid', 'fender', 'bumper', 'rest of car']
num_classes = 23
# total images for train+val = 2975 
# SPLIT_RATIO = 0.85
# train_size = 2530 # 2975 - 445(val) = 2530(train)
# val_size = 445 # 15% of data for validation
# COLAB_PATH = '/content/drive/MyDrive/data/carseg_data'
# DATA_PATH = r'./data/carseg_data'
IMAGE_PATH = r'./data/drone_dataset/semantic_drone_dataset/original_images/'
MASK_PATH = r'./data/drone_dataset/semantic_drone_dataset/label_images_semantic/'
# image settings
# IMAGE_WIDTH = 256
# IMAGE_HEIGHT = 256

# train hyperparameters settings
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 5
NUM_WORKERS = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
