import numpy as np
import glob
import torch
from torch.utils.data import DataLoader, random_split
import cv2
from os.path import basename
from keras.preprocessing.image import ImageDataGenerator
import random

def clahe_normalization(img, clahe_num = 1.5):
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clahe_num,tileGridSize=(8,8))
    lab[:, :, 0] = clahe.apply(lab[:,:, 0])
    lab[:, :, 1] = clahe.apply(lab[:,:, 1])
    lab[:, :, 2] = clahe.apply(lab[:,:, 2])
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return clahe_img



def image_resize(image, size, wrong_channel):
    if wrong_channel == True:
        image = np.rollaxis(image, 0, 3)
        image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        else:
            image = np.rollaxis(image, 2, 0)

    else:
        image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    return image



def create_augmented_images(image_paths, size, percentage_of_augmented_img = 0.25, num_variation_per_image = 4):
    total_augmented_images = 24 # 25 % of total images
    aug = ImageDataGenerator(
  #   rotation_range=45,
     zoom_range=0.25,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
     #shear_range=0.15,
    horizontal_flip=True,
     vertical_flip=True,
     fill_mode="nearest",
    )
    total_augmented_images = int(percentage_of_augmented_img * len(image_paths))
    #Wnum_images = int(total_augmented_images * num_variation_per_image)
    images_to_augment = random.sample(image_paths, total_augmented_images)
    train_image_standard = [np.load(i)[0:3, :, :] for i in images_to_augment]
    train_image_standard = [image_resize(i, size, wrong_channel=True) for i in train_image_standard]
    train_image_standard_hot = [np.expand_dims(np.load(i)[3,:,:], axis = 0) for i in images_to_augment]
    train_image_standard_hot = [image_resize(i, size, wrong_channel = True) for i in train_image_standard_hot]


    to_augment_color = []
    to_augment_class = []
    for j in range(total_augmented_images):
        norm_img = train_image_standard[j]
        norm_img = np.rollaxis(norm_img, 0, 3)
        norm_img = np.expand_dims(norm_img, axis = 0)
        bin_img = train_image_standard_hot[j]
        bin_img = np.rollaxis(bin_img, 0, 3)
        bin_img = np.expand_dims(bin_img, axis = 0)
        to_augment_color.append(norm_img)
        to_augment_class.append(bin_img)
    augmented_color = []
    augmented_class = []
    for i in range(len(to_augment_color)-1):
        n = 0
        for x in aug.flow(to_augment_color[i], to_augment_class[i]):
            augmented_color.append(x[0][0, :, :, :])
            augmented_class.append(x[1][0, :, :, :])
            if n == num_variation_per_image:
                break
            n = n+1
    augmented_class = [np.rollaxis(i, 2, 0) for i in augmented_class]
    augmented_color = [np.rollaxis(i, 2, 0) for i in augmented_color]
    return augmented_color, augmented_class


def get_non_augmented_images(image_paths):
    new_paths = []
    for path in image_paths:
        if "_a" in basename(path):
            new_paths.append(path)
        else:
            pass
    return new_paths

def get_share_of_cp_img(image_paths, share_of_real_img):
    new_paths = []
    for path in image_paths:
        if "_a" in basename(path):
            new_paths.append(path)
        else:
            pass
    sample_number = int(share_of_real_img*len(new_paths))
    paths_cp_generated = []
    for path in image_paths:
        if ("_a" not in basename(path) or not "Opel" in basename(path) or not "DOOR" in basename(path)):
            paths_cp_generated.append(path)
    paths_cp_generated_2 = random.sample(paths_cp_generated, sample_number)
    new_paths.extend(paths_cp_generated_2)
    return new_paths


def one_trans(img, size):
    w = size[0]
    h = size[1]
    new_img = np.zeros((9, w,h))
    for i in range(9):
        new_img[i, :, :] += (np.ones(img[0, :, :].shape)*i == img[0, :, :])
    return new_img

def get_image_path(pathname, share_of_augmented):
    images = glob.glob(pathname + '*.*')
    if share_of_augmented == True:
        image_paths = get_share_of_cp_img(images, share_of_real_img=0.5)
    else:
        image_paths = get_non_augmented_images(images)
    return image_paths



def augment_images(pathname, size, add_augmented = True, percentage_of_augmented_img = 0.25, num_variation_per_image = 4, all_images = False, gray = False, only_augment = False, add_noise = True, share_of_augmented = True, clahe = True):
    if share_of_augmented == True:
        image_paths = get_image_path(pathname, share_of_augmented = True)
    else:
        if all_images == False:
            image_paths = get_image_path(pathname, share_of_augmented=False)
        else:
            image_paths = glob.glob(pathname + '*.*')
    train_image_standard = [np.load(i)[0:3,:,:] for i in image_paths] # is for network: the problem is that the channels they gave us are in the wrong order for plotting
    train_image_standard = [image_resize(i, size, wrong_channel=True) for i in train_image_standard]
    train_image_standard_hot = [np.expand_dims(np.load(i)[3,:,:], axis = 0) for i in image_paths]
    train_image_standard_hot = [image_resize(i, size, wrong_channel = True) for i in train_image_standard_hot]
    train_image_standard_hot = [i.astype(np.float16) for i in train_image_standard_hot]
   # train_image_standard_hot = [one_trans(x, size) for x in train_image_standard_hot]
    if clahe == True:
        train_image_standard = [np.rollaxis(i, 0, 3) * 255 for i in train_image_standard]
        train_image_standard = [clahe_normalization(i) for i in train_image_standard]
        train_image_standard = [np.rollaxis(i, 2, 0) / 255 for i in train_image_standard]
    train_standard = []
    train_hot = []
    if add_noise == True:
        for i in range(len(train_image_standard)):
            noise = np.random.normal(0, 0.1, size)
            train_standard.append(train_image_standard[i] + noise)
            train_hot.append(train_image_standard_hot[i] + noise)

        train_image_standard = train_standard
        train_image_standard_hot = train_hot

    if gray == True:
        train_image_standard = [np.rollaxis(i, 0, 3) * 255 for i in train_image_standard]
        train_image_standard = [i.astype(np.uint8) for i in train_image_standard]
        train_image_standard = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in train_image_standard]
        train_image_standard = [np.expand_dims(i, axis = 0) for i in train_image_standard]
        train_image_standard = [i.astype(np.uint8) for i in train_image_standard]
        #train_image_stanadrd = [i/255 for i in train_image_standard]
    if add_augmented == True:
        augmented_color, augmented_class = create_augmented_images(image_paths,
                                                                   size,
                                                                   percentage_of_augmented_img=percentage_of_augmented_img,
                                                                   num_variation_per_image=num_variation_per_image)

        if gray == True:
            augmented_color = [np.rollaxis(i, 0, 3) for i in augmented_color]
            augmented_color = [i * 255 for i in augmented_color]
            augmented_color = [i.astype(np.uint8) for i in augmented_color]
            augmented_color = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in augmented_color]
          #  train_image_standard = [i / 255 for i in train_image_standard]
            augmented_color = [np.expand_dims(i, axis = 0) for i in augmented_color]

            #augmented_color = [255 * i for i in augmented_color]

        if only_augment == True:
            train_image_standard = augmented_color
            train_image_standard_hot = augmented_class

        else:
            train_image_standard.extend(augmented_color)
            train_image_standard_hot.extend(augmented_class)

    else:
        pass

    return train_image_standard, train_image_standard_hot, image_paths


def prepare_data(train_image_standard, train_image_standard_hot, valid_train_share, batch_size, image_paths):

    train_nr = int(len(image_paths) * (1- valid_train_share))
    valid_nr = len(image_paths) - train_nr


    a = np.expand_dims(train_image_standard_hot[0], axis = 0)

    X_t = torch.from_numpy(np.array(train_image_standard[:train_nr + valid_nr], dtype = 'uint8'))

    Y_t = torch.from_numpy(np.array(train_image_standard_hot[:train_nr + valid_nr], dtype ='uint8')).type(torch.LongTensor)


    # Just checking you have as many labels as inputs
    assert X_t.shape[0] == Y_t.shape[0]


    dset = torch.utils.data.TensorDataset(X_t, Y_t) # merge both together in a dataaset
    train, valid = random_split(dset,[train_nr, valid_nr]) # split them for training and validation

    trainloader = torch.utils.data.DataLoader(train,
                                    batch_size=batch_size, # choose your batch size
                                    shuffle=True) # generally a good idea


    validloader = torch.utils.data.DataLoader(valid,
                                    batch_size=batch_size, # choose your batch size
                                    shuffle=True) # generally a good idea

    return trainloader, validloader, train, valid


#train_image_standard, train_image_standard_hot, image_paths = augment_images(path,
 #                                                                                        size = SIZE,
  #                                                                                       add_augmented=True,
   #                                                                                      percentage_of_augmented_img=0.2,
    #                                                                                     num_variation_per_image=1,
     #                                                                                    all_images = True,
      #                                                                                   gray = True,
       #                                                                                  only_augment=False,
        #                                                                                    add_noise=False)
