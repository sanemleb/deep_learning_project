import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import torch

#from IPython.display import clear_output
from skimage.io import imread
from skimage import transform as tf
from skimage.transform import resize
import cv2
from PIL import Image
import random
import scipy
#from sklearn.preprocessing import normalize
from os.path import basename
#import regex as re
import os
import torchvision
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, GRU, Conv2d, Dropout, MaxPool2d, BatchNorm1d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.utils.data import DataLoader, random_split

from augmentation.image_processing import*







model = torch.load("/zhome/ed/a/183709/saved_model_dice_deepCE_smol_W_col.pth")


# imports the image processing packets


#from import_data import*
#image_paths = image_path() 
from unet.Network import *
# imports the test images
#from batch_loader import test_images
from sklearn import preprocessing

#test_one_hot_, test_image_, test_image_standard_, test_image_standard_hot_ = test_images()

drive_path = r"/zhome/ed/a/183709/data/data/test/*.*"
image_paths = glob.glob(drive_path)

test_image_ = [np_transform_bgr(np.load(i)[0:3, :, 🙂) for i in image_paths]
test_one_hot_ = [one_hot_image(np.load(i)) for i in image_paths]

test_image_standard_hot_show = [np.expand_dims(np.load(i)[3,:,:], axis = 0) for i in image_paths]
test_image_standard_hot_ = [one_trans(x) for x in test_image_standard_hot_show]

test_image_standard_ = [np.load(i)[0:3,:,:] for i in image_paths] #for non normal
#test_image_standard_ = [np.expand_dims(preprocessing.normalize(rgb_grey(np.load(i))), axis = 0) for i in image_paths]
#test_image_standard_ = [np.expand_dims(rgb_grey(np.load(i)), axis = 0) for i in image_paths] #for non normal



test_image_standard = test_image_standard_
test_image_standard_hot = test_image_standard_hot_

#test_image_standard = [np.load(i)[0:3,:,:] for i in test_image_standard_ ]
#test_image_standard_hot = [(np.load(i)[3,:,:]) for i in test_image_standard_hot_ ]
#test_image_standard_hot = [one_trans(x) for x in test_image_standard_hot]



X_t = torch.from_numpy(np.array(test_image_standard, dtype = 'float32'))
X_t= X_t.to(device='cuda')
Y_t = torch.from_numpy(np.array(test_image_standard_hot, dtype ='float32'))
Y_t = Y_t.to(device = 'cuda')


# imports the network



def accuracy(ys, ts):
    predictions = torch.max(ys, 1)[1]
    correct_prediction = torch.eq(predictions, ts)
    return torch.mean(correct_prediction.float())



print(X_t.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"
Net = U_Net_Model()
Net = torchvision.models.segmentation.deeplabv3_resnet101(num_classes = 9)
Net.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

if torch.cuda.is_available():
    Net.cuda()

Net.load_state_dict(model)
Net.eval()

def nine_to_one(x):
    img = np.zeros((256,256))
    for i in range(len(x)):
        img[:,:] += x[i,:,:]*i
    return img




select_test = np.arange(30)
from pytorch_toolbelt.losses import DiceLoss
criterion1 = DiceLoss( mode = 'multilabel')
count = 0



avg_dice = 0
for i,x in enumerate(select_test):
    test = Net(X_t[x:x+1])['out']
    acc = criterion1(test, Y_t[x:x+1])      
    acc = acc.item()
    #_, test = torch.max(test.data, 1)
    test = torch.max(test, 1)[1]
    test= test.data.cpu().detach().numpy()
    test = np.array(test)[0]
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(test)
    axarr[0].title.set_text("Prediction")
    axarr[0].axis('off')
    axarr[1].imshow(test_image_standard_hot_show[x][0])
    axarr[1].title.set_text("Target")
    axarr[1].axis('off')
    axarr[2].imshow(test_image_[x])
    axarr[2].title.set_text("Original image")
    axarr[2].axis('off')
    f.suptitle("Dice Score: " + str(1 - acc)[:4])
    f.tight_layout()
    f.subplots_adjust(top=1.35)
    plt.savefig('/zhome/ed/a/183709/lol/comparison'+str(x) + '.png')
    plt.show()
    avg_dice += acc/30

print(avg_dice)
