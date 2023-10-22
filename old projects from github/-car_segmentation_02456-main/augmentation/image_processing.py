import numpy as np
import cv2
import random
from scipy.ndimage import rotate

size = 100
clahe_parameter = int(255 / 10)  # 10 histograms


def np_transform_bgr(a):
    r = a[0, :, :]
    g = a[1, :, :]
    b = a[2, :, :]
    rgb = np.dstack((r, g, b))
    return rgb


def np_transform_rgb_inv(img):
    shape = np.shape(img)
    new_img = np.zeros(shape[::-1])
    new_img[0, :, :] = img[:, :, 0]
    new_img[1, :, :] = img[:, :, 1]
    new_img[2, :, :] = img[:, :, 2]
    return new_img.astype("uint8")


def rgb_img(a):  # images are in bgr format!!!
    # Transforms the data into standard rgb form (3, x, x)
    return a[0:3, :, :]


def rgb_grey(a):
    r = a[0, :, :]
    g = a[1, :, :]
    b = a[2, :, :]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def one_hot_image(a):
    # Seperates the one-hot-encoded part of the data
    return a[3, :, :]


# resizing the image
def image_resize(image, size):
    return cv2.resize(image,
                      size,
                      interpolation=cv2.INTER_NEAREST)


def one_trans(img):
    new_img = np.zeros((9, np.shape(img)[1], np.shape(img)[2]))
    for i in range(9):
        new_img[i, :, :] = (i == img[:, :]).astype(float)
    return new_img


def one_trans_inv(img):
    new_img = np.zeros(np.shape(img))
    for i in range(9):
        new_img[:, :] = img[0, i, :, :] * i
    return new_img

