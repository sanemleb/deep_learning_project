import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
from torch.utils.data import DataLoader
import albumentations as albu
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision.transforms import *
from utils.dataloader.car_dataset import *
from utils.background_rem.bg_remover import init_bg_remover
from utils.background_rem.bg_manager import BgManager
from torchmetrics.functional import dice_score, accuracy

def save_logs(train_log, valid_log):
    np.save("./train_log_bo_test.npy", train_log)
    np.save("./valid_log_bo_test.npy", valid_log)

if __name__ == '__main__':    
    print("TORCH V -----------------------------")
    print(torch.__version__, torch.cuda.is_available())
    print("-------------------------------------")
    
    validation_path = "E:/dtu/2sem/deeplearn/project/data_folder_2/validation/"
    train_path = "E:/dtu/2sem/deeplearn/project/data_folder_2/train/"
    test_path = "E:/dtu/2sem/deeplearn/project/data_folder_2/test/"

    transform = transforms.Compose([
        RandomHorizontalFlip(p=0.5),
        RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomApply(transforms=[
            RandomResizedCrop(size=(256, 256), scale=(0.40, 1.0)),
        ], p=0.4),
        transforms.RandomApply(transforms=[
            GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        ], p=0.2),
        transforms.RandomErasing(p=0.2),  # new
        transforms.RandomRotation(degrees=(-10, 10)),  # 
    ])

    predictor = init_bg_remover()

    # train_dataset = CarDataset(train_path, num_gan=0, num_deloitte_aug=0, num_opel=300, num_door=300, num_primary_multiple=8, 
    #     bg_manager = BgManager(), predictor=predictor, augmentation=transform)
    # validation_dataset = CarDataset(validation_path, num_gan=0, num_deloitte_aug=0, num_opel=0, num_door=0, num_primary_multiple=1)

    train_dataset = CarDataset(train_path, num_gan=0, num_deloitte_aug=0, num_opel=300, num_door=300, num_primary_multiple=8, augmentation=transform)
    validation_dataset = CarDataset(validation_path, num_gan=0, num_deloitte_aug=0, num_opel=0, num_door=0, num_primary_multiple=1)

    # train_dataset = CarDataset(train_path, num_gan=0, num_deloitte_aug=0, num_opel=300, num_door=300, num_primary_multiple=8, augmentation=transform, 
    #     grayscale=True)
    # validation_dataset = CarDataset(validation_path, num_gan=0, num_deloitte_aug=0, num_opel=0, num_door=0, num_primary_multiple=1, grayscale=True)

    test_dataset = CarDataset(test_path, test=True)

    print(train_dataset[0][0].shape)
    print(train_dataset[0][1].shape)
    print("train size: ",len(train_dataset))
    print("vali size: ",len(validation_dataset))

    ENCODER = 'timm-resnest200e'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d' 
    DEVICE = 'cuda'
    channels_nr = 3 # 1 if grayscale

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=8+1, 
        activation=ACTIVATION,
        in_channels=channels_nr,
    )
    
    model.to(DEVICE)
    
    def freeze_encoder(model):
        for child in model.encoder.children():
            for param in child.parameters():
                param.requires_grad = False
        return

    # freeze_encoder(model) # lock the backbone

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    criterion = smp.utils.losses.DiceLoss()

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    min_score = 1 

    train_logs = []
    valid_logs = []

    EPOCHS = 80
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
        print("Training loss: ",train_mean)
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
        print("Validation loss: ",valid_mean)
        valid_logs.append(valid_mean)

        if (min_score > valid_mean):
            min_score = valid_mean
            torch.save(model.state_dict(), 'best_model_dict.pth')
            print("Model saved!")
        if i == EPOCHS/2:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('---- Decrease Learning Rate to 1e-5! ----')

    save_logs(train_logs, valid_logs)

    ############### NOW WE TEST ##################

    test_model = smp.UnetPlusPlus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=8+1, 
        activation=ACTIVATION,
        in_channels=channels_nr
    )
    test_model.to(DEVICE)

    test_model.load_state_dict(torch.load('./best_model_dict.pth'))
    
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    dice_scores = []
    accs = []

    test_model.eval()
    for i in test_dataloader: 
        img, mask = i
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        pr_mask = test_model.predict(img)
        pred = pr_mask[0]
        truth = mask[0]

        pred_label = torch.argmax(pred, dim=0)
        truth_label = torch.argmax(truth, dim=0)
        truth_flat = truth_label.view(-1) # go from [256,256] -> [256*256]
        pred_flat = torch.flatten(pred, start_dim=1) # go from [9,256,256] -> [9,256*256]
        pred_flat = pred_flat.permute(1,0) # go from [9,256*256] -> [256*256,9]

        data_dicescore = dice_score(pred_flat, truth_flat, reduction='none', no_fg_score=-1)
        masked_dices = torch.masked_select(data_dicescore,data_dicescore.not_equal(-1))
        dice_scores.append(masked_dices.mean())

        acc = accuracy(pred_label, truth_label,average='macro',num_classes=9)
        accs.append(acc)
    print("dice_scores: ", np.mean(dice_scores))
    print("Accuracy: ", np.mean(accs))