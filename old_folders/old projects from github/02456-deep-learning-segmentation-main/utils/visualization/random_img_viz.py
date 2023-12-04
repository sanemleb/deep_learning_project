import matplotlib.pyplot as plt
from torchvision.transforms import *
import segmentation_models_pytorch as smp
import torch
import albumentations as albu
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
from PIL import Image, ImageChops
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def visualize(car_img=None, mask=None):
    n = 2
    plt.figure(figsize=(11, 5))
    plt.subplot(1, n, 1)
    plt.imshow(car_img)
    plt.title("Image")
    plt.subplot(1, n, 2)
    plt.imshow(mask)
    plt.title("Predicted mask")
    plt.show()

pil_img = Image.open("E:/carcrash.png")
pil_img = pil_img.resize((256,256))
np_img = np.array(pil_img)[...,:3]

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

img = transforms(np_img)

ENCODER = 'timm-resnest200e'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d'
DEVICE = 'cuda'

model = smp.UnetPlusPlus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=8+1,
    activation=ACTIVATION,
    # in_channels=1
)

model_path = 'E:/GitHub/02456-deep-learning-segmentation/models/unetpp_200resnest_8prim_300dooropel.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

pred = model.predict(img[None,:])
pred = pred[0]
pred_int = torch.argmax(pred,dim=0)

visualize(car_img=np_img, mask=pred_int)