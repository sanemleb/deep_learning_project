import cv2
import numpy as np
from typing import Tuple
import os
from tqdm import tqdm #bar for loops

def resize_with_pad(image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (114, 114, 114)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def resize_images(input_folder, output_folder, new_size):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each file in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read the image using OpenCV
        img = cv2.imread(input_path)

        resized_img=resize_with_pad(img, (new_size,new_size))

        # Save the resized image to the output folder
        cv2.imwrite(output_path, resized_img)


if __name__ == "__main__":
    # Set your input and output folders
    input_folder = r"C:\Users\ELLI\Desktop\Third semester\Deep Learning Elli Georgiou\carseg_data\carseg_data\images\orange_3_doors\no_segmentation"
    output_folder = r"C:\Users\ELLI\Desktop\Third semester\Deep Learning Elli Georgiou\SegmentationPreProcessing\resized2"

    # Set the desired size for the longer side of the image
    new_size = 256

    # Call the function to resize images
    resize_images(input_folder, output_folder, new_size)

