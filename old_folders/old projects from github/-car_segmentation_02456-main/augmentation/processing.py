from image_processing import *
import numpy as np
from import_data import *


def train_val_image(image_paths):
    train_one_hot = [one_hot_image(np.load(i)) for i in image_paths]
    train_image = [np_transform_bgr(np.load(i)) for i in image_paths]
    train_image_standard = [np.load(i)[0:3, :, :] for i in image_paths]
    train_image_standard_hot = [np.expand_dims(np.load(i)[3, :, :], axis=0) for i in image_paths]
    train_image_standard_hot = [one_trans(x) for x in train_image_standard_hot]
    return train_one_hot, train_image, train_image_standard, train_image_standard_hot



