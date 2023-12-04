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
num_classes = 10
# total images for train+val = 2975 
SPLIT_RATIO = 0.85
# train_size = 2530 # 2975 - 445(val) = 2530(train)
# val_size = 445 # 15% of data for validation
COLAB_PATH = '/content/drive/MyDrive/data/carseg_data'
DATA_PATH = r'./data/carseg_data'

# image settings
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
MEAN=[0.485, 0.456, 0.406] # values from ImageNet
STD=[0.229, 0.224, 0.225] # values from ImageNet

# train hyperparameters settings
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DROPOUT = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 70
NUM_WORKERS = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
