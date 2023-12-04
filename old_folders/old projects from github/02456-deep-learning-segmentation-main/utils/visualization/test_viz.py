from utils.dataloader.car_dataset import *
import matplotlib.pyplot as plt
from torchvision.transforms import *
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

test_path = "E:/dtu/2sem/deeplearn/project/data_folder_2/test/"


def visualize(car_img=None, mask=None, predicted=None):
    n = 3
    plt.figure(figsize=(16, 5))
    plt.subplot(1, n, 1)
    plt.imshow(np.dstack(car_img))
    plt.title("Actual image")
    plt.subplot(1, n, 2)
    plt.imshow(mask)
    plt.title("Mask")
    plt.subplot(1, n, 3)
    plt.imshow(predicted)
    plt.title("Prediction")
    plt.show()

if __name__ == '__main__':
    ENCODER = 'timm-resnest200e'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d'
    DEVICE = 'cuda'

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=8+1,
        activation=ACTIVATION,
        # in_channels=3
    )

    path = 'E:/GitHub/02456-deep-learning-segmentation/models/unetpp_200resnest_8prim_300dooropel.pth'

    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    test_dataset = CarDataset(test_path, test=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    model.eval()
    for i in range(30):
        n = i
        print("image index: ", i)

        image, gt_mask = test_dataset[n]
        gt_mask = gt_mask.permute(1, 2, 0)
        gt_mask = torch.argmax(gt_mask, dim=2)
        print(gt_mask.shape)

        x_tensor = image.unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        print(pr_mask.shape)
        pr_mask = pr_mask.squeeze().permute(1, 2, 0)
        pr_mask = torch.argmax(pr_mask, dim=2)
        print(pr_mask.shape)

        visualize(
            car_img=image,
            mask=gt_mask.numpy(),
            predicted=pr_mask.numpy()
        )
