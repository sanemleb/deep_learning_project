from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from model import UNET
from settings import DATA_PATH, NUM_EPOCHS,BATCH_SIZE,SPLIT_RATIO, LEARNING_RATE, device
from utils import get_data_loaders
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

def train(): 
    model = UNET(in_channels=3, out_channels=9)
    model.to(device)
    train, val, test = get_data_loaders(DATA_PATH, BATCH_SIZE, SPLIT_RATIO)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    # train_epoch_losses = []
    # val_epoch_losses = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        
        running_loss = 0.0
        for images, masks in tqdm(train, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            print(images.shape)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train)
        writer.add_scalar("Loss/train", average_loss, epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val, desc=f"Validation {epoch + 1}/{NUM_EPOCHS}"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        average_val_loss = val_loss / len(val)
        writer.add_scalar("Loss/validation", average_val_loss, epoch)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "unet_model.pth")

if __name__ == "__main__":
    train()
    
    

