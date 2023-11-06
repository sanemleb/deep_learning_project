from torch.utils.data import random_split, DataLoader, Subset
# from settings import createCarPartsDictionary, data_path, batch_size
import settings
from dataset import CarDataset
import numpy as np


if __name__ == '__main__':

    # data_loader = DataLoader(CarDataset(settings.data_path, batch_size=settings.batch_size, shuffle=False))

    # dataset
    car_dataset = CarDataset(settings.data_path)

    # create test dataset
    # access test image/mask paths
    test_img_paths = CarDataset(settings.data_path).img_test_photo_paths
    test_mask_paths = CarDataset(settings.data_path).mask_test_photo_paths
    test_dataset = car_dataset.create_test_dataset() # call to function to create a test subset from whole dataset

    # split data into train, val
    total_data = len(car_dataset)

    index_list = list(range(total_data))

    # Shuffle the indices
    random_seed = 42  # You can use any seed for reproducibility
    np.random.seed(random_seed)
    np.random.shuffle(index_list)

    # Define the ratio for train/validation split
    split_ratio = settings.split_ratio  # 85% for training, 15% for validation
    split_index = int(total_data * split_ratio)

    # Split the index lists
    train_index_list = index_list[:split_index]
    val_index_list = index_list[split_index:]

   # Create data loaders for training and validation
    train_dataset = Subset(car_dataset, train_index_list)
    val_dataset = Subset(car_dataset, val_index_list)
    
    train_dataloader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=settings.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=False)

    # skeleton for train/val loops when laoding data from dataLoaders
    # for epoch in range(num_epochs):
    # # Training loop
    #     for batch in train_dataloader:
    #         inputs, masks = batch
    #         # Perform training steps

    #     # Validation loop
    #     for batch in val_dataloader:
    #         inputs, masks = batch
    #         # Perform validation steps
