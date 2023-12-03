from torch.utils.data import random_split, DataLoader
from modules.dataset import CarDataset
import numpy as np

def get_data_loaders(data_path, batch_size, split_ratio=0.8):
    dataset = CarDataset(data_path)

    # Calculate the number of samples for training and validation based on the split ratio
    num_samples = len(dataset)
    num_train = int(split_ratio * num_samples)
    num_val = num_samples - num_train

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader