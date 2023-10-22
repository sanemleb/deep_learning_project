
import glob
from os.path import basename

def image_path(path):
    #image_paths = glob.glob(r"/work/data/*.npy") #Peter
    image_paths = glob.glob()
    print(image_paths)
    new_paths2 = []
    wanted = [""]
    for path in image_paths:
        for substring in wanted:
            if substring in basename(path):
                new_paths2.append(path)
    print("Total Wanted Observations paths:\t", len(new_paths2))
    print("Total Observations paths:\t", len(image_paths))
    return image_paths
