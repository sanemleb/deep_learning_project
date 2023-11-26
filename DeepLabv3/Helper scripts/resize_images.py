import cv2
import numpy as np
import os
import argparse

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    interp = cv2.INTER_LINEAR

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("image_size", help="The size the output image", type=int)
    parser.add_argument("padding_color", help="The padding color of the image", type=int)

    args = parser.parse_args()

    main_folder= args.input_dir

    # create the destination directory if it doesnt exist
    dst_folder = args.output_dir
    if not os.path.isdir(dst_folder):
        os.mkdir(dst_folder)
    else:
        print("Directory \"" + dst_folder + "\" already exists")

    folder_list = ["images", "labels"]
    for folder in folder_list:
        src_path = os.path.join(main_folder, folder)
        if not os.path.isdir(src_path):
            print("Path: " + src_path + " does not exist!")
            return
        print("Processing folder: " + folder)
        
        # list all the image names
        imageNamesList=os.listdir(src_path)
        
        # create destination folders
        dst_directory = os.path.join(dst_folder, folder)
        if not os.path.isdir(dst_directory):
            os.mkdir(dst_directory)
        else:
            print("Directory \"" + dst_directory + "\" already exists")

        # resize the images
        for imageName in imageNamesList:
            src = os.path.join(src_path, imageName)
            img = cv2.imread(src)
            resizedImg = resizeAndPad(img, (args.image_size,args.image_size), args.padding_color)
            dst = os.path.join(dst_directory, imageName)
            cv2.imwrite(dst, resizedImg)

if __name__ == "__main__":
    main()