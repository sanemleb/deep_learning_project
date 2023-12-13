import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch import Tensor
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F



################################################################################################
################################################################################################
################################################################################################
# Settings


def createCarPartsDictionary():
    carParts = {
        "hood": {
            "color": "orange",
            "class_value": 10
        },
        "front door": {
            "color": "dark green",
            "class_value": 20
        },
        "rear door": {
            "color": "yellow",
            "class_value": 30
        },
        "frame": {
            "color": "cyan",
            "class_value": 40
        },
        "rear quarter panel": {
            "color": "purple",
            "class_value": 50
        },
        "trunk lid": {
            "color": "light green",
            "class_value": 60
        },
        "fender": {
            "color": "blue",
            "class_value": 70
        },
        "bumper": {
            "color": "pink",
            "class_value": 80
        },
        "rest of car": {
            "color": "no color",
            "class_value": 90
        }
    }
    return carParts

# data settings
car_parts = ['hood', 'front door', 'rear door', 'frame', 'rear quarter panel', 'trunk lid', 'fender', 'bumper', 'rest of car']
num_classes = 10
# total images for train+val = 2975 
SPLIT_RATIO = 0.85
# train_size = 2530 # 2975 - 445(val) = 2530(train)
# val_size = 445 # 15% of data for validation
DATA_PATH = r'./data/carseg_data'

# image settings
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
MEAN=[0.485, 0.456, 0.406] # values from ImageNet
STD=[0.229, 0.224, 0.225] # values from ImageNet

# train hyperparameters settings
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DROPOUT = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 40
NUM_WORKERS = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################################################################################
################################################################################################
################################################################################################
# Dice Loss


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)




################################################################################################
################################################################################################
################################################################################################
# Models


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=num_classes, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Convert the final output to the same data type as the input
        x = x.to(dtype=x.dtype)

        x = self.final_conv(x)

        return x
    
    
    
    
######################################## RESNET UNET ###########################################
class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU -> Dropout
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True, dropout_prob=0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
            x = self.dropout(x)
        return x

class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose", dropout_prob=0.5):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels, dropout_prob=dropout_prob)
        self.conv_block_2 = ConvBlock(out_channels, out_channels, dropout_prob=dropout_prob)

    def forward(self, up_x, down_x):
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class UNET_RESNET(nn.Module):
    DEPTH = 6

    def __init__(self, in_channels=3, out_channels=10, dropout_prob=0.5):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024, dropout_prob=dropout_prob))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512, dropout_prob=dropout_prob))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256, dropout_prob=dropout_prob))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128,
                                                    dropout_prob=dropout_prob))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64,
                                                    dropout_prob=dropout_prob))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNET_RESNET.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNET_RESNET.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

################################################################################################
################################################################################################
################################################################################################
# Dataset and Data Loaders


class CarDataset(Dataset):
    def __init__(self, data_dir , data_dir_path):
        self.data_dir = data_dir
        self.mask_5door_paths = [os.path.join(data_dir,data_dir_path, mask) for mask in os.listdir(os.path.join(data_dir, data_dir_path)) if "black" in mask]
        self.mask_3door_paths = [os.path.join(data_dir,data_dir_path, mask) for mask in os.listdir(os.path.join(data_dir, data_dir_path)) if "orange" in mask]
        self.mask_photo_paths = [os.path.join(data_dir,data_dir_path, mask) for mask in os.listdir(os.path.join(data_dir, data_dir_path)) if "photo" in mask]

        self.all_mask_paths = self.mask_5door_paths + self.mask_3door_paths + self.mask_photo_paths

        self.transform_vanilla = transforms.Compose([transforms.ToTensor()]) 

 
    def __len__(self):
        return len(self.all_mask_paths)

    def __getitem__(self, idx):
        
        mask_path = self.all_mask_paths[idx]
        
        mask = np.load(mask_path, allow_pickle=True)
        img = mask[:, :, :3]
        mask_split = mask[:, :, 3]
        mask_split = mask_split.astype(int)
        
        img_transformed = self.transform_vanilla(img)

        img = img_transformed.to(torch.float32)
        ms = torch.from_numpy(mask_split).long()

        return img, ms 
    
    def calculate_mean_std(self):
        # Calculate mean and std
        num_channels = 3  # Assuming RGB images
        mean_accumulator = np.zeros(num_channels)
        std_accumulator = np.zeros(num_channels)

        for idx in range(len(self)):
            img, _ = self[idx]
            img_np = img.numpy()

            mean_accumulator += np.mean(img_np, axis=(1, 2))
            std_accumulator += np.std(img_np, axis=(1, 2))

        mean = mean_accumulator / len(self)
        std = std_accumulator / len(self)

        return mean, std


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


################################################################################################
################################################################################################
################################################################################################
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
