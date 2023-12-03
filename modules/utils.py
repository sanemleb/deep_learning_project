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



# Evaluation functions

def pixel_accuracy(predicted, target):
    """
    Calculate pixel accuracy.

    Args:
        predicted (torch.Tensor): Predicted segmentation mask.
        target (torch.Tensor): Ground truth segmentation mask.

    Returns:
        float: Pixel accuracy.
    """
    correct_pixels = (predicted == target).sum().item()
    total_pixels = target.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy

def mean_pixel_accuracy(predicted_list, target_list):
    """
    Calculate the mean pixel accuracy across multiple images.

    Args:
        predicted_list (list of torch.Tensor): List of predicted segmentation masks.
        target_list (list of torch.Tensor): List of ground truth segmentation masks.

    Returns:
        float: Mean pixel accuracy.
    """
    total_accuracy = 0.0
    num_images = len(predicted_list)

    for predicted, target in zip(predicted_list, target_list):
        accuracy = pixel_accuracy(predicted, target)
        total_accuracy += accuracy

    mean_accuracy = total_accuracy / num_images
    return mean_accuracy

def dice_coefficient(predicted, target, smooth=1e-5):
    """
    Calculate the Dice Coefficient.

    Args:
        predicted (torch.Tensor): Predicted binary mask.
        target (torch.Tensor): Ground truth binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice Coefficient.
    """
    intersection = (predicted * target).sum()
    total_area_pred = predicted.sum()
    total_area_gt = target.sum()

    dice = (2.0 * intersection + smooth) / (total_area_pred + total_area_gt + smooth)
    return dice.item()



def save_metric_to_file(file_name, model_type, model_name, mean_accuracy):
    """
    Save model type, name, and mean accuracy to a text file.

    Args:
        model_type (str): Type of the model.
        model_name (str): Name of the model.
        mean_accuracy (float): Mean accuracy value to be saved.
        file_name (str): Name of the text file.

    Returns:
        None
    """
    try:
        # Open the file in append mode
        with open(file_name, 'a') as file:
            # Write the model type, name, mean accuracy, and a newline
            file.write(f"{model_type} - {model_name} Mean Pixel Accuracy: {mean_accuracy:.5f}\n")
    except FileNotFoundError:
        # If the file doesn't exist, create a new file
        with open(file_name, 'w') as file:
            file.write(f"{model_type} - {model_name} Mean Pixel Accuracy: {mean_accuracy:.5f}\n")

