import os
import numpy as np
import random
import math

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


def save_files(data_path, save_path, arr):
    for i in arr:
        path = data_path + i
        data = np.load(path)
        new_path = save_path + i
        np.save(new_path, data)


# The folder where you run this script
txt_paths = "E:/dtu/2sem/deeplearn/testing/deloitte_data_filter/"
data_path = "E:/dtu/2sem/deeplearn/project/rgb_mask_data_normalized/"
save_path = "E:/dtu/2sem/deeplearn/project/data_folder_2/"
files = os.listdir(data_path)

# First we get the 30 test images given by martin, and remove the augmented versions of them
test_file = open(txt_paths+"test_data.txt", "r")
test_data_jpg = test_file.read().split("\n")

test = get_data_paths(files, test_data_jpg)
data_no_test = get_data_without_folder(files, test)
test_augmented = get_augmented(data_no_test, test)
training_data = get_data_without_folder(data_no_test, test_augmented)

# We then get all the gan images and save them
gan_file = open(txt_paths+"cycle_gan_files.txt", "r")
gan_data_jpg = gan_file.read().split("\n")
gan = get_data_paths(files, gan_data_jpg)
training_no_gan = get_data_without_folder(training_data, gan)

# We get all the opel images and save them
opel = get_specific_data(training_no_gan, 'OPEL')
training_no_opel = get_data_without_folder(training_no_gan, opel)

# We get all the door images and save them
door = get_specific_data(training_no_opel, 'DOOR')
training_no_door = get_data_without_folder(training_no_opel, door)

# We get all the augmented images and save them
aug = get_specific_data(training_no_door, "-aug")
training_primary = get_data_without_folder(training_no_door, aug)

##################################################################################
# Now we split each of the different data kinds up in 80/20 train and validation #
seed_int = 32  # set seed to get same splits everytime

random.seed(seed_int)
train_primary = random.sample(
    training_primary, math.ceil(len(training_primary)*0.80))
val_primary = [i for i in training_primary if i not in train_primary]
print("train prim len:", len(train_primary),
      "- val prim len:", len(val_primary))

# We get all the augmented of our primary data and add them in train and aug
train_aug = get_augmented(aug, train_primary)
val_aug = get_augmented(aug, val_primary)
print("train aug len:", len(train_aug), "- val aug len:", len(val_aug))

random.seed(seed_int)
train_opel = random.sample(opel, math.ceil(len(opel)*0.80))
val_opel = [i for i in opel if i not in train_opel]

random.seed(seed_int)
train_door = random.sample(door, math.ceil(len(door)*0.80))
val_door = [i for i in door if i not in train_door]

random.seed(seed_int)
train_gan = random.sample(gan, math.ceil(len(gan)*0.80))
val_gan = [i for i in gan if i not in train_gan]

# And finally we now have a completely fair split of 80/20 of each kind
validation = val_gan+val_door+val_opel+val_aug+val_primary
train = train_gan+train_door+train_opel+train_aug+train_primary

# Unit tests to ensure that the 3 are completely split up
for i in train:
    assert(i not in validation)
    assert(i not in test)

for i in validation:
    assert(i not in test)

# Save images
save_files(data_path, save_path+"test/", test)
save_files(data_path, save_path+"validation/", validation)
save_files(data_path, save_path+"train/", train)
