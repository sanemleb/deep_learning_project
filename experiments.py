from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from modules.model import UNET, UNET_RESNET
from modules.unetPP import UNetPP 
from modules.settings import DATA_PATH, NUM_EPOCHS,BATCH_SIZE,SPLIT_RATIO, LEARNING_RATE, device
from modules.utils import get_data_loaders, pixel_accuracy, save_metric_to_file, mean_pixel_accuracy, save_dice_loss_to_file
from modules.resizeImages import resize_with_pad
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import os
import cv2
from typing import Tuple
from torchsummary import summary
from torchvision import transforms as T
from sklearn.metrics import confusion_matrix


def load(model_path, is_resnet = False):
    if not is_resnet:
        mdl = UNET()
    else:
        mdl = UNET_RESNET(3)
    # './models/experiment_outputs/output_models_arrays_0_unet/' + model_name
    mdl.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return mdl
    
    
def test(mdl, model_type, model_name):
    masks_path = './data/carseg_data/test_arrays_mask_corrected'
    torch.set_printoptions(sci_mode=False)

    # Create a list to store the model outputs
    model_outputs = []
    model_outputs_images = []
    model_outputs_gt = []
    model_output_pixelAcc = [] 
    filenames_in_order = []

    # Set the model to evaluation mode
    mdl.eval()

    # Define the transformation to be applied to the input images
    transform = transforms.Compose([transforms.ToTensor()])
    reverse_transform = transforms.Compose([transforms.ToPILImage()])
    how_many_images_to_test = 30
    index = 0

    for filename in os.listdir(masks_path):
        mask_path = os.path.join(masks_path, filename)
        mask = np.load(mask_path, allow_pickle=True)
        
                
        img = mask[:, :, :3]
        mask_split = mask[:, :, 3]
        mask_split = mask_split.astype(int)
        one_hot_encoded = np.eye(10, dtype=int)[mask_split.squeeze()]
        dice_mask = np.expand_dims(one_hot_encoded.transpose(2,0,1), axis=0)
                
        img = transform(img)
        img_rev = reverse_transform(img)
                    
        mdl.eval()
        # Make the prediction
        with torch.no_grad():
            img = img.unsqueeze(0)
            filenames_in_order.append(filename)
            output = mdl(img)
            # sigmoid_output = torch.sigmoid(output)
            argmax_output = torch.argmax(output, dim=1)

            # Calculate pixel accuracy
            accuracy = pixel_accuracy(argmax_output[0], torch.tensor(mask_split))
            model_output_pixelAcc.append(accuracy)
            
        # Store the output in the list
        model_outputs.append(argmax_output[0])
        model_outputs_images.append(img_rev)
        model_outputs_gt.append(torch.tensor(mask_split))

        #detect for first 10 images
        if index == how_many_images_to_test:
            break
        index = index + 1

    # The `model_outputs` list now contains the model's output for each test image
    # Calculate and print mean pixel accuracy
    mean_accuracy = mean_pixel_accuracy(model_outputs, model_outputs_gt)


    # Save scores to a file
    mean_dice = save_dice_loss_to_file("./models/experiment_results/"+ model_type +"_" +model_name + "_dice_scores.txt", model_outputs, model_outputs_gt, filenames_in_order)
    save_metric_to_file("./models/experiment_results/scores.txt", model_type, model_name, mean_accuracy, mean_dice)
    
    
def save_plot_loss_from_txt(file_path, name):
    # Read data from the text file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract data from each line
    indices, train_losses, val_losses = [], [], []
    for line in lines[1:]:  # Assuming the first line contains column headers
        index, train_loss, val_loss = map(float, line.strip().split())
        indices.append(index)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Plot the training and validation losses in overlapping subplots
    plt.figure(figsize=(10, 5))

    # Overlapping subplot for both losses
    plt.plot(indices, train_losses, label='Train Loss', marker='o', linestyle='-', color='blue')
    plt.plot(indices, val_losses, label='Validation Loss', marker='o', linestyle='-', color='orange')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot as PNG
    plt.savefig(name + '.png', bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    
    os.makedirs("./models/experiment_results", exist_ok=True)
    
    ############# ARRAYS 0 -- UNET #####################
    ep = "40"
    path_model = "./models/experiment_outputs/output_models_arrays_0_unet/unet_model_epoch_40.pth"
    path_txt = "./models/experiment_outputs/output_models_arrays_0_unet/loss_data.txt"
    mdl = load(model_path=path_model, is_resnet=False)
    test(mdl, "arrays_0_unet_", ep + "_epochs")
    save_plot_loss_from_txt(path_txt, "./models/experiment_results/" + "arrays_0_unet_" + ep + "_epoch_loss")
    
    
    ############# ARRAYS 1 -- UNET #####################
    ep = "40"
    path_model = "./models/experiment_outputs/output_models_arrays_1_unet/unet_model_epoch_40.pth"
    path_txt = "./models/experiment_outputs/output_models_arrays_1_unet/loss_data.txt"
    mdl = load(model_path=path_model, is_resnet=False)
    test(mdl, "arrays_1_unet_", ep + "_epochs")
    save_plot_loss_from_txt(path_txt, "./models/experiment_results/" + "arrays_1_unet_" + ep + "_epoch_loss")
    
    
    ############# ARRAYS 2 -- UNET #####################
    ep = "40"
    path_model = "./models/experiment_outputs/output_models_arrays_2_unet/unet_model_epoch_40.pth"
    path_txt = "./models/experiment_outputs/output_models_arrays_2_unet/loss_data.txt"
    mdl = load(model_path=path_model, is_resnet=False)
    test(mdl, "arrays_2_unet_", ep + "_epochs")
    save_plot_loss_from_txt(path_txt, "./models/experiment_results/" + "arrays_2_unet_" + ep + "_epoch_loss")
    
    
    ############# ARRAYS 3 -- UNET #####################
    ep = "40"
    path_model = "./models/experiment_outputs/output_models_arrays_3_unet/unet_model_epoch_40.pth"
    path_txt = "./models/experiment_outputs/output_models_arrays_3_unet/loss_data.txt"
    mdl = load(model_path=path_model, is_resnet=False)
    test(mdl, "arrays_3_unet_", ep + "_epochs")
    save_plot_loss_from_txt(path_txt, "./models/experiment_results/" + "arrays_3_unet_" + ep + "_epoch_loss")
    
    
    ############# ARRAYS 4 -- UNET #####################
    ep = "40"
    path_model = "./models/experiment_outputs/output_models_arrays_4_unet/unet_model_epoch_40.pth"
    path_txt = "./models/experiment_outputs/output_models_arrays_4_unet/loss_data.txt"
    mdl = load(model_path=path_model, is_resnet=False)
    test(mdl, "arrays_4_unet_", ep + "_epochs")
    save_plot_loss_from_txt(path_txt, "./models/experiment_results/" + "arrays_4_unet_" + ep + "_epoch_loss")
    
    
    
    ####################################################
    ####################################################
    
    
    ############# ARRAYS 0 -- UNET_RESNET #####################
    ep = "40"
    path_model = "./models/experiment_outputs/output_models_arrays_0_unet_resnet/unet_model_epoch_40.pth"
    path_txt = "./models/experiment_outputs/output_models_arrays_0_unet_resnet/loss_data.txt"
    mdl = load(model_path=path_model, is_resnet=True)
    test(mdl, "arrays_0_unet_resnet_", ep + "_epochs")
    save_plot_loss_from_txt(path_txt, "./models/experiment_results/" + "arrays_0_unet_resnet_" + ep + "_epoch_loss")
    
    
    ############# ARRAYS 1 -- UNET_RESNET #####################
    ep = "40"
    path_model = "./models/experiment_outputs/output_models_arrays_1_unet_resnet/unet_model_epoch_40.pth"
    path_txt = "./models/experiment_outputs/output_models_arrays_1_unet_resnet/loss_data.txt"
    mdl = load(model_path=path_model, is_resnet=True)
    test(mdl, "arrays_1_unet_resnet_", ep + "_epochs")
    save_plot_loss_from_txt(path_txt, "./models/experiment_results/" + "arrays_1_unet_resnet_" + ep + "_epoch_loss")
    
    
    ############# ARRAYS 2 -- UNET_RESNET #####################
    ep = "40"
    path_model = "./models/experiment_outputs/output_models_arrays_2_unet_resnet/unet_model_epoch_40.pth"
    path_txt = "./models/experiment_outputs/output_models_arrays_2_unet_resnet/loss_data.txt"
    mdl = load(model_path=path_model, is_resnet=True)
    test(mdl, "arrays_2_unet_resnet_", ep + "_epochs")
    save_plot_loss_from_txt(path_txt, "./models/experiment_results/" + "arrays_2_unet_resnet_" + ep + "_epoch_loss")
    
    
    ############# ARRAYS 3 -- UNET_RESNET #####################
    ep = "40"
    path_model = "./models/experiment_outputs/output_models_arrays_3_unet_resnet/unet_model_epoch_40.pth"
    path_txt = "./models/experiment_outputs/output_models_arrays_3_unet_resnet/loss_data.txt"
    mdl = load(model_path=path_model, is_resnet=True)
    test(mdl, "arrays_3_unet_resnet_", ep + "_epochs")
    save_plot_loss_from_txt(path_txt, "./models/experiment_results/" + "arrays_3_unet_resnet_" + ep + "_epoch_loss")
    
    
    ############# ARRAYS 4 -- UNET_RESNET #####################
    ep = "40"
    path_model = "./models/experiment_outputs/output_models_arrays_4_unet_resnet/unet_model_epoch_40.pth"
    path_txt = "./models/experiment_outputs/output_models_arrays_4_unet_resnet/loss_data.txt"
    mdl = load(model_path=path_model, is_resnet=True)
    test(mdl, "arrays_4_unet_resnet_", ep + "_epochs")
    save_plot_loss_from_txt(path_txt, "./models/experiment_results/" + "arrays_4_unet_resnet_" + ep + "_epoch_loss")
    