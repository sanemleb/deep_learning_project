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
    # TODO:  Do we need to feed the image size here ??
    model = UNET(in_channels=3, out_channels=9)
    model.to(device)
    
    train_dl, val_dl, test_dl = get_data_loaders(DATA_PATH, BATCH_SIZE, SPLIT_RATIO)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    criterion = nn.functional.nll_loss
    writer = SummaryWriter()

    train_epoch_losses = []
    val_epoch_losses = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        
        running_loss = 0.0
        for images, masks in tqdm(train_dl, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            reshaped_image_tensor = images.permute(0, 3, 1, 2)
            reshaped_image_tensor = reshaped_image_tensor.to(torch.float32)
            reshaped_mask_tensor = masks.permute(0, 3, 1, 2)
            reshaped_mask_tensor = reshaped_mask_tensor.to(torch.float32)

            outputs = model(reshaped_image_tensor)

            # Extract predicted class indices
            predicted_classes = torch.argmax(outputs[:, :, :, :], dim=1)
            softmax_output = torch.nn.functional.softmax(predicted_classes.float(), dim=1)
            # Extract ground truth class indices from the 4th channel of the mask
            ground_truth_classes = reshaped_mask_tensor[:, 3, :, :]

            loss = nn.functional.cross_entropy(softmax_output, ground_truth_classes)
            # loss.backward()
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
                
                
                reshaped_image_tensor = images.permute(0, 3, 1, 2)
                reshaped_image_tensor = reshaped_image_tensor.to(torch.float32)
                reshaped_mask_tensor = masks.permute(0, 3, 1, 2)
                reshaped_mask_tensor = reshaped_mask_tensor.to(torch.float32)

                outputs = model(reshaped_image_tensor)

                # Extract predicted class indices
                predicted_classes = torch.argmax(outputs[:, :, :, :], dim=1)
                softmax_output = torch.nn.functional.softmax(predicted_classes.float(), dim=1)
                # Extract ground truth class indices from the 4th channel of the mask
                ground_truth_classes = reshaped_mask_tensor[:, 3, :, :]

                loss = nn.functional.cross_entropy(softmax_output, ground_truth_classes)
                
                
                val_loss += loss.item()

        average_val_loss = val_loss / len(val_dl)
        val_epoch_losses.append(average_val_loss)
        writer.add_scalar("Loss/validation", average_val_loss, epoch)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "unet_model.pth")

if __name__ == "__main__":
    train()
    
    

