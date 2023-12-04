import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
from torchvision.transforms import *
from utils.dataloader.car_dataset import *
from unet import ResNetUNet

validation_path = "E:/dtu/2sem/deeplearn/project/data_folder_2/validation/"
train_path = "E:/dtu/2sem/deeplearn/project/data_folder_2/train/"
test_path = "E:/dtu/2sem/deeplearn/project/data_folder_2/test/"

transform = transforms.Compose([
            RandomHorizontalFlip(p=0.5),
            RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomApply(transforms=[
                RandomResizedCrop(size=(256, 256)),
            ], p=0.5),
            transforms.RandomApply(transforms=[
                GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            ], p=0.5),
        ])

train_dataset = CarDataset(train_path, num_gan=0, num_deloitte_aug=0, num_opel=300, num_door=300, num_primary_multiple=8, augmentation=transform)
validation_dataset = CarDataset(validation_path, num_gan=0, num_deloitte_aug=0, num_opel=0, num_door=0, num_primary_multiple=1)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
valid_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)

model = ResNetUNet(n_class=9).cuda()
criterion = smp.utils.losses.DiceLoss()
optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, nesterov=True)
min_score = 1.0

DEVICE = 'cuda'

train_logs = []
valid_logs = []

EPOCHS = 5
for i in range(0, EPOCHS):
    print('\nEpoch: {}'.format(i))
    train_log = []
    model.train()
    for image, mask in train_loader:
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)
        
        optimizer.zero_grad()

        pred = model(image)

        loss = criterion(pred, mask)
        loss.backward()
        
        optimizer.step()

        train_log.append(loss.item())

    train_mean = np.mean(train_log)
    print("Mean Training loss: ",train_mean)
    train_logs.append(train_mean)

    valid_log = []
    model.eval()
    for image, mask in valid_loader:
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)   

        pred = model(image)

        loss = criterion(pred,mask)

        valid_log.append(loss.item())

    valid_mean = np.mean(valid_log)
    print("Mean Validation loss: ",valid_mean)
    valid_logs.append(valid_mean)

    if (min_score > valid_mean):
        min_score = valid_mean
        torch.save(model.state_dict(), 'best_model_dict.pth')
        print("Model saved!")
    if i == EPOCHS/2:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('---- Decrease Learning Rate to 1e-5! ----')
