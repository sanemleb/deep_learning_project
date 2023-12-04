import random
import tensorflow as tf
import cv2
import numpy as np
from augmentation.image_processing import image_resize

def shear_crop_resize(image, size):  # size has to be tuple
    shear = random.uniform(-0.2, 0.2)  # introduce random effect to introduce flexibility to the input
    afine_tf = tf.AffineTransform(shear=shear)
    # Apply transform to image data
    modified = tf.warp(image, inverse_map=afine_tf)  # perform affine transformation
    col = image.shape[0]
    factor = int(shear * col)
    if shear >= 0:
        cropped_img = modified[int(factor / 2):col - int(factor / 2),
                      int(shear * col):col]  # crop image to remove part where it is black due to affine transformation
    else:
        row = col
        cropped_img = modified[0 - int(factor / 2):row + int(factor / 2), 0:col + (factor)]
    resized_img = image_resize(cropped_img, size)  # resize image to final size
    return resized_img


def clahe_4_rgb(
        input_img):  # clahe normalization for rgb images wwhere value of hsv colorroom was used to mimic gra channel value
    hsvImg = cv2.cvtColor(np.float32(input_img), cv2.COLOR_BGR2HSV)
    value_channel = hsvImg[:, :, 2]
    value_channel = np.uint16(value_channel * 255)
    clahe = cv2.createCLAHE(clipLimit=2)
    value_channel = np.uint16(value_channel * 255) + 30
    final_img = clahe.apply(value_channel)
    final_img = final_img / 255
    hsvImg[:, :, 2] = final_img
    rgb = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    rgb = np.int16(rgb)
    return rgb


def clahe_4_gray(input_img,
                 clipLimit):  # clahe normalization for gray images clip Limit could be hyperparameter if it shows significant changes to performance
    gray = cv2.cvtColor(np.float32(input_image), cv2.COLOR_BGR2GRAY)
    gray = np.uint16(gray * 255)
    clahe = cv2.createCLAHE(clipLimit=clipLimit)
    final_img = clahe.apply(gray)
    final_img = np.int16(final_img)
    del clahe
    return final_img


def random_crop(img, crop_size=(150, 150), seed=1):
    np.random.seed(seed)
    img_shape = img.shape
    pad_size = (img_shape[0] - crop_size[0]) // 2
    assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1], "Crop size should be less than image size"
    img = img.copy()
    w, h = img.shape[:2]
    x, y = np.random.randint(h - crop_size[0]), np.random.randint(w - crop_size[1])
    img = img[y:y + crop_size[0], x:x + crop_size[1]]

    return np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')




# rotating the image a random amount of times
def ranodom_rotation(batch):
    # rotating the image randomly (only 0 deg, 90 deg, 180 deg and 270 deg for keeping the same informations)
    rand_rot = (random.randint(0, 3))
    for i in range(0, rand_rot):
        if rand_rot == 0:
            return batch
        else:
            batch = np.rot90(batch, 1)
    return batch


# flips the imae
def random_flipper_horizontal(train_image):
    randomizer = [random.randint(0, 1) for x in
                  range(len(train_image))]  # creates boolean list and assigns true or false for each image
    for index in range(len(train_image)):
        image = train_image[index]
        if randomizer == True:  # if true then the element will be flipped
            flipped_image = np.fliplr(train_image)
            train_image[index] = flipped_image

    return train_image

