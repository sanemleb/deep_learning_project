import numpy as np
from numpy import load
import random
from glob import glob
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

txt_paths = "E:/GitHub/02456-deep-learning-segmentation/utils/data_prep/"

def get_data_paths(files, data_target):
    data_paths = []
    for f in files:
        for t in data_target:
            if (f[:-4] == t[:-4]):
                data_paths.append(f)
    return data_paths


def get_data_without_folder(files, folder):
    data = []
    for f in files:
        if f not in folder:
            data.append(f)
    return data


def get_augmented(data_no_test, test_data):
    test_augmented = []
    for f in data_no_test:
        for t in test_data:
            if t[:-4]+"-aug" == f[:-6]:
                test_augmented.append(f)
    return test_augmented


def get_specific_data(dataset, data_type):
    dataset_na = []
    for f in dataset:
        if data_type in f:
            dataset_na.append(f)
    return dataset_na


def get_gan_images(dataset):
    gan_file = open(txt_paths+"cycle_gan_files.txt", "r")
    gan_data_jpg = gan_file.read().split("\n")
    gan = get_data_paths(dataset, gan_data_jpg)
    return gan


class CarDataset(Dataset):
    def __init__(self, imgs_dir, seed=42, num_opel=-1, num_door=-1,
                 num_deloitte_aug=-1, num_gan=-1, num_primary_multiple=1, augmentation=None, 
                 test=False, predictor=None, bg_manager=None, grayscale=False):
        self.imgs_dir = imgs_dir
        self.augmentation = augmentation
        self.predictor = predictor
        self.bg_manager = bg_manager
        self.grayscale = grayscale

        raw_ids = os.listdir(imgs_dir)

        random.seed(seed)
        random.shuffle(raw_ids)

        self.ids = []

        if (test == False):
            opel = get_specific_data(raw_ids, 'OPEL')
            if (num_opel == -1):
                self.ids = self.ids + opel
            else:
                assert(len(opel) >= num_opel)
                random.seed(seed)
                sample_opel = random.sample(opel, num_opel)
                self.ids = self.ids + sample_opel

            door = get_specific_data(raw_ids, 'DOOR')
            if (num_door == -1):
                self.ids = self.ids + door
            else:
                assert(len(door) >= num_door)
                random.seed(seed)
                sample_door = random.sample(door, num_door)
                self.ids = self.ids + sample_door

            aug = get_specific_data(raw_ids, '-aug')
            if (num_deloitte_aug == -1):
                self.ids = self.ids + aug
            else:
                assert(len(aug) >= num_deloitte_aug)
                random.seed(seed)
                sample_aug = random.sample(aug, num_deloitte_aug)
                self.ids = self.ids + sample_aug

            gan = get_gan_images(raw_ids)
            if (num_gan == -1):
                self.ids = self.ids + gan
            else:
                assert(len(gan) >= num_gan)
                random.seed(seed)
                sample_gan = random.sample(gan, num_gan)
                self.ids = self.ids + sample_gan

            primary_images = []
            for f in raw_ids:
                if ((f not in aug) and (f not in gan) and (f not in door) and (f not in opel)):
                    primary_images.append(f)
            for i in range(num_primary_multiple):
                self.ids = self.ids + primary_images
        else:
            self.ids = raw_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]

        np_obj = glob(self.imgs_dir + idx)

        data = load(np_obj[0])

        img = data[0:3]
        # We try to remove the background
        if (self.predictor != None):
            image_bg = self.bg_manager.get_image(np_obj[0])
            if (image_bg == 'empty'):
                img_no_bg = self.bg_manager.get_img_no_bg(self.predictor, np.dstack(img))
                if (img_no_bg != 'empty'):
                    self.bg_manager.add_image(np_obj[0],img_no_bg)
                else: 
                    self.bg_manager.add_image(np_obj[0],'empty')
            else: 
                img_no_bg = image_bg
            if (img_no_bg != 'empty'):
                img = torch.from_numpy(img_no_bg).type(
                    torch.FloatTensor)  # numpy -> torch
                # The predictor takes H,W,C so we make it C,H,W again
                img = img.permute(2, 0, 1)
            else:
                # print("------- COULD NOT REMOVE BACKGROUND OF IMAGE: ---------")
                # print(np_obj)
                # print("-------------------------------------------------------")
                img = torch.from_numpy(img).type(
                    torch.FloatTensor)  # numpy -> torch
        else:
            img = torch.from_numpy(img).type(
                torch.FloatTensor)  # numpy -> torch

        mask = data[-1]
        mask = torch.from_numpy(mask).type(torch.FloatTensor)  # numpy -> torch
        mask = torch.nn.functional.one_hot(
            mask.to(torch.int64), 9)  # We one-hot-encode the mask
        mask = mask.permute(2, 0, 1)  # (256,256,9) -> (9,256,256)

        if (self.augmentation != None):
            # Choose a numpy seed and ensure that transforms use it
            seed = np.random.randint(2147483647)
            np.random.seed(seed)
            torch.manual_seed(seed)
            img = self.augmentation(img)
            np.random.seed(seed)
            torch.manual_seed(seed)
            mask = self.augmentation(mask)
        
        if (self.grayscale == True):
            img = T.Grayscale()(img)

        return img, mask