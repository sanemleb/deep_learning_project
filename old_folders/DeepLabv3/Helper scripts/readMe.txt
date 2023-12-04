1)saveNumpyAsArray.py should be ran if the dataset is not already extracted to images. The script extracts the numpy arrays to images and labels (both as images)

2) Run resize_image script for resizing the 256x256 images to another image size (for deep lab 380x380)
 python resize_images.py ./path/to/dataset ./path/to/resized_dataset 380 0

 ./path/to/dataset should consist of images, labels folders

 3) Run split_data.py to split the training data into train and val dataset (Mandatory before training a model)
 python split_data.py ./path/to/trainDataset ./path/to/Split
 ./path/to/trainDataset should have the images, labels folder for training data
 ./path/to/Split will be the destination path for train, val split

 Fiji tool https://imagej.net/software/fiji/downloads
 With this tool you can observe the black masks to see if they are correct:
 1) Load a mask
 2) Press Image -> Adjust -> Brightness/Contrast
 3) Press "Auto" under Contrast.