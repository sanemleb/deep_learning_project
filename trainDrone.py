from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
# from modules.model import UNET

import albumentations as A
from torch.utils.data import Dataset, DataLoader
import cv2
from droneDS1.model_droneDS import UNET_drone as UNET_d
from droneDS1.droneDataset import DroneDataset
from droneDS1.settingsDrone import IMAGE_PATH, NUM_EPOCHS, MASK_PATH, LEARNING_RATE, device
# from modules.utils import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def train():
    model = UNET_d(in_channels=3, out_channels=23)
    model.to(device)
    print(summary(model, (3, 256, 256)) )

    for param in model.parameters():
        param.to(device)


    # train_dl, val_dl, test_dl = get_data_loaders(DATA_PATH, BATCH_SIZE, SPLIT_RATIO)

    n_classes = 23 

    def create_df():
        name = []
        for dirname, _, filenames in os.walk(IMAGE_PATH):
            for filename in filenames:
                name.append(filename.split('.')[0])
        return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

    df = create_df()

    #split data
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    t_train = A.Compose([A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(), 
                     A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                     A.GaussNoise()])

    t_val = A.Compose([A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                   A.GridDistortion(p=0.2)])

    #datasets
    train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train, patch=False)
    val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val, patch=False)

    #dataloader
    batch_size= 3

    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=True)         

    # train_dl, val_dl, test_dl = get_data_loaders(DATA_PATH, BATCH_SIZE, SPLIT_RATIO)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # class_weights = torch.tensor([1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 2.0]).to(device)
    # normalized_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    train_epoch_losses = []
    val_epoch_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        for images, masks in tqdm(train_dl, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_dl)
        train_epoch_losses.append(average_loss)
        writer.add_scalar("Loss/train", average_loss, epoch)

        # Validation loop
        model.eval()

        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_dl, desc=f"Validation {epoch + 1}/{NUM_EPOCHS}"):
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        average_val_loss = val_loss / len(val_dl)
        val_epoch_losses.append(average_val_loss)
        writer.add_scalar("Loss/validation", average_val_loss, epoch)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "unet_model_drone_0.pth")
    
    # Save the loss data 
    file = "loss_data_drone_0.txt"
    data = np.column_stack((np.arange(len(train_epoch_losses)), train_epoch_losses, val_epoch_losses))
    np.savetxt(file, data, header="Index Train_Loss Val_Loss", comments="", fmt="%d %.4f %.4f")
    return file
    
if __name__ == "__main__":
    train()
