from torch.utils.data import random_split, DataLoader
from modules.dataset import CarDataset
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

def get_data_loaders(data_dir, data_path, batch_size, split_ratio=0.8):
    dataset = CarDataset(data_path, data_dir)

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

class AddGaussianNoise(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, pil_image):
        # Convert PIL image to a PyTorch tensor
        tensor_image = F.to_tensor(pil_image)

        # Adding Gaussian noise to the tensor
        noise = torch.randn_like(tensor_image) * self.std + self.mean
        noisy_tensor = tensor_image + noise

        # Convert the noisy tensor back to a PIL image
        noisy_pil_image = F.to_pil_image(noisy_tensor)

        return noisy_pil_image

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.weight is not None:
            pt = pt * self.weight.gather(0, target.view(-1))

        loss = -((1 - pt) ** self.gamma) * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

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

def dice_coefficient(y_true, y_pred, axis=(2, 3), smooth=1e-5):
    """
    Calculate the Dice Coefficient.

    Args:
        predicted (torch.Tensor): Predicted binary mask.
        target (torch.Tensor): Ground truth binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice Coefficient.
    """

    intersection = torch.sum(y_true * y_pred, dim=axis)
    union = torch.sum(y_true, dim=axis) + torch.sum(y_pred, dim=axis)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    mean = weighted_average_dice(y_true, dice)
    return mean, dice


def weighted_average_dice(mask, dice_scores):
    """
    Calculate the weighted average Dice score based on the pixel count of each class.

    Parameters:
    - mask: A 3D array representing the segmentation mask with shape (num_classes, height, width).
    - dice_scores: A 1D array containing Dice scores for each class.

    Returns:
    - Weighted average Dice score.
    """
    mask = mask[0]
    # Calculate pixel count for each class
    class_sizes = torch.sum(mask, axis=(1, 2))  # Sum along height and width dimensions

    # Calculate the weighted mean Dice score
    weighted_mean_dice = torch.sum(dice_scores * class_sizes / torch.sum(class_sizes))

    return weighted_mean_dice


def save_metric_to_file(file_name, model_type, model_name, mean_accuracy, mean_dice):
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
            file.write(f"{model_type} - {model_name} Mean Dice Score: {mean_dice:.5f}\n")
    except FileNotFoundError:
        # If the file doesn't exist, create a new file
        with open(file_name, 'w') as file:
            file.write(f"{model_type} - {model_name} Mean Pixel Accuracy: {mean_accuracy:.5f}\n")
            file.write(f"{model_type} - {model_name} Mean Dice Score: {mean_dice:.5f}\n")


def save_dice_loss_to_file(file_name, outputs, masks, filenames_in_order):

    model_output_mean_dice = []
    model_output_dice_per_class = []
    class_average_dice = []
    
    for i in range(0,len(outputs)):
        
        one_hot_encoded = np.eye(10, dtype=int)[masks[i].squeeze()]
        dice_mask = np.expand_dims(one_hot_encoded.transpose(2,0,1), axis=0)
        mean_dice, dice_per_class = dice_coefficient(torch.from_numpy(dice_mask), torch.nn.functional.one_hot(outputs[i], num_classes=10).permute(2,0,1).unsqueeze(0))
        model_output_mean_dice.append(mean_dice)
        model_output_dice_per_class.append(dice_per_class)
        
    for i in range(0,10):
        dice_sum = 0
        for j in range(0, len(model_output_dice_per_class)):
            dice_sum += model_output_dice_per_class[j][0][i].item()
        class_average_dice.append(dice_sum/len(model_output_dice_per_class))
        
    
    try:
        # Open the file in append mode
        with open(file_name, 'w+') as file:
            pass
        with open(file_name, 'a') as file:
            # Write the model type, name, mean accuracy, and a newline
            file.write(f"DICE SCORE RESULTS FOR {len(outputs)} TEST IMAGES\n")
            file.write(f"\n \n")
            file.write(f"\n \n")

            for i in range(0, len(filenames_in_order)):
                file.write(f"Test Image {filenames_in_order[i]} Dice Score Results\n")
                file.write(f"\n")
                file.write(f"Mean Dice Score: {model_output_mean_dice[i]}\n")
                file.write(f"\n")
                for j in range(0, 10):
                    file.write(f"Dice Score for class {j}: {model_output_dice_per_class[i][0][j]}\n")
                file.write(f"\n \n")
                    
            file.write(f"AVERAGE DICE SCORE PER CLASS: \n")
            file.write(f"\n \n")
            for m in range(0, 10):
                file.write(f"Dice Score for class {m}: {class_average_dice[m]}\n")

    except FileNotFoundError:
        # If the file doesn't exist, create a new file
        with open(file_name, 'w') as file:
            # Write the model type, name, mean accuracy, and a newline
            file.write(f"DICE SCORE RESULTS FOR {len(outputs)} TEST IMAGES\n")
            file.write(f"\n \n")
            file.write(f"\n \n")

            for i in range(0, len(filenames_in_order)):
                file.write(f"Test Image {filenames_in_order[i]} Dice Score Results\n")
                file.write(f"\n")
                file.write(f"Mean Dice Score: {model_output_mean_dice[i]}\n")
                file.write(f"\n")
                for j in range(0, 10):
                    file.write(f"Dice Score for class {j}: {model_output_dice_per_class[i][0][j]}\n")
                file.write(f"\n \n")
                    
            file.write(f"AVERAGE DICE SCORE PER CLASS: \n")
            file.write(f"\n \n")
            for m in range(0, 10):
                file.write(f"Dice Score for class {m}: {class_average_dice[m]}\n")
            
    length = len(model_output_mean_dice)
    return sum(model_output_mean_dice).item()/length
