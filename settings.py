import os

def createCarPartsDictionary():
    carParts = {
        "hood": {
            "color": "orange",
            "class_value": 10
        },
        "front door": {
            "color": "dark green",
            "class_value": 20
        },
        "rear door": {
            "color": "yellow",
            "class_value": 30
        },
        "frame": {
            "color": "cyan",
            "class_value": 40
        },
        "rear quarter panel": {
            "color": "purple",
            "class_value": 50
        },
        "trunk lid": {
            "color": "light green",
            "class_value": 60
        },
        "fender": {
            "color": "blue",
            "class_value": 70
        },
        "bumper": {
            "color": "pink",
            "class_value": 80
        },
        "rest of car": {
            "color": "no color",
            "class_value": 90
        }
    }
    return carParts

# data settings
car_parts = ['hood', 'front door', 'rear door', 'frame', 'rear quarter panel', 'trunk lid', 'fender', 'bumper', 'rest of car']
num_classes = 9
# total images for train+val = 2975 
split_ratio = 0.85
# train_size = 2530 # 2975 - 445(val) = 2530(train)
# val_size = 445 # 15% of data for validation
data_path = r"C://Users//micha//Downloads//carseg_data//carseg_data"

# hyperparameters settings
batch_size = 32
epochs = 1
lr = 1e-3
