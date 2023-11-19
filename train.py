from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from modules.model import UNET
from modules.settings import DATA_PATH, NUM_EPOCHS,BATCH_SIZE,SPLIT_RATIO, LEARNING_RATE, device
from modules.utils import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

def train():
    # TODO:  Do we need to feed the image size here ??
    model = UNET(in_channels=3, out_channels=10)
    model.to(device)
    print(summary(model, (3, 256, 256)) )

    for param in model.parameters():
        param.to(device)

    train_dl, val_dl, test_dl = get_data_loaders(DATA_PATH, BATCH_SIZE, SPLIT_RATIO)
    #optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
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

            outputs = model(images)

            # # Extract predicted class indices
            # predicted_classes = torch.argmax(outputs[:, :, :, :], dim=1)
            # softmax_output = torch.nn.functional.softmax(predicted_classes.float(), dim=1)
            # # Extract ground truth class indices from the 4th channel of the mask
            # ground_truth_classes = masks[:, 3, :, :]

            loss = nn.functional.cross_entropy(outputs, masks)
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

                loss = nn.functional.cross_entropy(outputs, masks)

                val_loss += loss.item()

        average_val_loss = val_loss / len(val_dl)
        val_epoch_losses.append(average_val_loss)
        writer.add_scalar("Loss/validation", average_val_loss, epoch)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "unet_model.pth")
    
    # Save the loss data 
    file = "loss_data.txt"
    data = np.column_stack((np.arange(len(train_epoch_losses)), train_epoch_losses, val_epoch_losses))
    np.savetxt(file, data, header="Index Train_Loss Val_Loss", comments="", fmt="%d %.4f %.4f")
    return file
    
if __name__ == "__main__":
    train()
    
    

