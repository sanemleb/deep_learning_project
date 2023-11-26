import os
import torch

# data settings
num_classes = 9
IMAGE_PATH = r'./data/car_dataset/images/'
MASK_PATH = r'./data/car_dataset/masks/'

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 15
NUM_WORKERS = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
