from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from modules.model import UNET, UNET_RESNET
from modules.unetPP import UNetPP
from modules.settings import DATA_PATH, COLAB_PATH, NUM_EPOCHS, BATCH_SIZE, SPLIT_RATIO, LEARNING_RATE, device, num_classes
from modules.utils import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from modules.dice import dice_loss
import os

def train(true_if_unet_resnet = False, data_dir = "arrays", saveIntermediateModels = False):
    print(device)
    if true_if_unet_resnet:
        model = UNET_RESNET(in_channels=3, out_channels=10)
    else:
        model = UNET(in_channels=3, out_channels=10)

    model.to(device)
    print(summary(model, (3, 256, 256)) )

    for param in model.parameters():
        param.to(device)

    train_dl, val_dl = get_data_loaders(data_dir, DATA_PATH, BATCH_SIZE, SPLIT_RATIO)
    optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    train_epoch_losses = []
    val_epoch_losses = []

    if true_if_unet_resnet:
        output_models_dir = "output_models_" + data_dir + "_unet_resnet"
    else:
        output_models_dir = "output_models_" + data_dir + "_unet"
    os.makedirs(output_models_dir, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        for images, masks in tqdm(train_dl, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss += dice_loss(torch.nn.functional.softmax(outputs, dim=1).float(),
                              torch.nn.functional.one_hot(masks, num_classes).permute(0, 3, 1, 2).float(),
                              multiclass=True)
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
                loss += dice_loss(torch.nn.functional.softmax(outputs, dim=1).float(),
                              torch.nn.functional.one_hot(masks, num_classes).permute(0, 3, 1, 2).float(),
                              multiclass=True)
                val_loss += loss.item()

        average_val_loss = val_loss / len(val_dl)
        val_epoch_losses.append(average_val_loss)
        writer.add_scalar("Loss/validation", average_val_loss, epoch)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}")

        if saveIntermediateModels:
            # Save the trained model every 10 epochs after the 50th epoch
            if (epoch + 1) % 10 == 0:
                model_save_path = os.path.join("experiment_outputs", output_models_dir, f"unet_model_epoch_{epoch + 1}.pth")
                os.makedirs(os.path.join("experiment_outputs", output_models_dir), exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at epoch {epoch + 1} to {model_save_path}")
        
    if not saveIntermediateModels:
        # Save the trained model at the end
        os.makedirs(os.path.join("experiment_outputs", output_models_dir), exist_ok=True)
        torch.save(model.state_dict(), os.path.join("experiment_outputs", output_models_dir, f"unet_model_epoch_{epoch + 1}.pth"))

    # Save the loss data
    data = np.column_stack((np.arange(len(train_epoch_losses)), train_epoch_losses, val_epoch_losses))
    path = os.path.join("experiment_outputs", output_models_dir, "loss_data.txt")
    os.makedirs(os.path.join("experiment_outputs", output_models_dir), exist_ok=True)
    np.savetxt(path, data, header="Index Train_Loss Val_Loss", comments="", fmt="%d %.4f %.4f")
    return path

if __name__ == "__main__":
    train(true_if_unet_resnet = False, data_dir = "arrays_0", saveIntermediateModels = True)
    train(true_if_unet_resnet = False, data_dir = "arrays_1", saveIntermediateModels = True)
    train(true_if_unet_resnet = False, data_dir = "arrays_2", saveIntermediateModels = True)
    train(true_if_unet_resnet = False, data_dir = "arrays_3", saveIntermediateModels = True)
    train(true_if_unet_resnet = False, data_dir = "arrays_4", saveIntermediateModels = True)
    train(true_if_unet_resnet = True, data_dir = "arrays_0", saveIntermediateModels = True)
    train(true_if_unet_resnet = True, data_dir = "arrays_1", saveIntermediateModels = True)
    train(true_if_unet_resnet = True, data_dir = "arrays_2", saveIntermediateModels = True)
    train(true_if_unet_resnet = True, data_dir = "arrays_3", saveIntermediateModels = True)
    train(true_if_unet_resnet = True, data_dir = "arrays_4", saveIntermediateModels = True)
