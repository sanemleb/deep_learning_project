from torch.utils.data import random_split, DataLoader, Subset
import modules.settings
from modules.dataset import CarDataset
import numpy as np
from modules.settings import MEAN, STD

# def get_data_loaders(data_path, batch_size, split_ratio, random_seed=42):
#     car_dataset = CarDataset(data_path)

#     # Create test dataset
#     test_dataset = car_dataset.create_test_dataset()

#     total_data = len(car_dataset)
#     index_list = list(range(total_data))

#     # Shuffle the indices
#     np.random.seed(random_seed)
#     np.random.shuffle(index_list)

#     # Define the ratio for train/validation split
#     split_index = int(total_data * split_ratio)

#     # Split the index lists
#     train_index_list = index_list[:split_index]
#     val_index_list = index_list[split_index:]

#     # Create data loaders for training, validation, and test
#     train_dataset = Subset(car_dataset, train_index_list)
#     val_dataset = Subset(car_dataset, val_index_list)

#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_dataloader, val_dataloader, test_dataloader


def get_data_loaders(data_path, batch_size, split_ratio=0.8):
    dataset = CarDataset(data_path, MEAN, STD)

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