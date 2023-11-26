import numpy as np
import cv2
import os
from tqdm import tqdm
import glob

def ListFiles(srcFilePath, fileExtensionList):
    os.chdir(srcFilePath)
    allFileNames = []
    for fileExtension in fileExtensionList:
        fileNames = glob.glob(fileExtension)
        allFileNames = allFileNames + fileNames
    return allFileNames

def CreateDirectory(dst_directory):
    if not os.path.isdir(dst_directory):
        os.mkdir(dst_directory)

nupyArrayPath = r"C:\arrays"
dstImagesPath = r"C:\images"
CreateDirectory(dstImagesPath)
dstLabelsPath = r"C:\labels"
CreateDirectory(dstLabelsPath)

arrayNames = ListFiles(nupyArrayPath, ["*.npy"])

for arrayName in tqdm(arrayNames):
    arrayPath = os.path.join(nupyArrayPath, arrayName)
    array = np.load(arrayPath)
    classes = array[:,:,3]
    mask = np.zeros([256, 256, 1], dtype=int)
    for i in range(0, classes.shape[0]):
        for j in range(0, classes.shape[1]):
            mask[i][j] = int(classes[i][j]/10)
            if classes[i][j] == 90:
                mask[i][j] = 0
    
    im_rgb = cv2.cvtColor(array[:,:,0:3], cv2.COLOR_BGR2RGB)
    
    dstImagePath = os.path.join(dstImagesPath, arrayName.replace(".npy", ".png"))
    dstLabelPath = os.path.join(dstLabelsPath, arrayName.replace(".npy", ".png"))
    cv2.imwrite(dstLabelPath, mask)
    cv2.imwrite(dstImagePath, im_rgb)