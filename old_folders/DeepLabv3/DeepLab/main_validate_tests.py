import torch
import numpy as np
from torchvision import transforms
import cv2
from PIL import Image
import os 
import argparse

import custom_model

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Arguments """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input test data directory")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("model_name", help="Model's name")
    args = parser.parse_args()
    test_path = args.input_dir
    save_dir = args.output_dir
    create_dir(save_dir)
    # Number of classes in the dataset
    num_classes = 9

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, input_size = custom_model.initialize_model(num_classes, keep_feature_extract=True, use_pretrained=False)

    model_path = "results"
    model_name = args.model_name
    model_path = os.path.join(model_path, model_name)
    state_dict = torch.load(model_path, map_location=device)

    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    transforms_image =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    VOC_COLORMAP = [
        [0, 0 ,0],[250, 149, 10], [19, 98, 19], [249, 249, 10], [10, 248, 250], [149, 7, 149], [5, 249, 9], [20, 19, 249], [249, 9, 250]
        ]

    for img in os.listdir(test_path):
        image = Image.open(os.path.join(test_path,img))
        name = img.split('/')[-1]

        image_np = np.asarray(image)

        image = Image.fromarray(image_np)
        image = transforms_image(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        
        outputs = model(image)["out"]
        
        _, preds = torch.max(outputs, 1)
        preds = preds.to("cpu")

        preds_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)

        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        for i in range(0, image_np.shape[0]):
            for j in range(0, image_np.shape[1]):
                image_np[i][j] = [VOC_COLORMAP[preds_np[i][j]][2], VOC_COLORMAP[preds_np[i][j]][1], VOC_COLORMAP[preds_np[i][j]][0]]

        path_name = os.path.join(save_dir, name)
        cv2.imwrite(path_name, image_np)
        print("Image " + name + " is saved..")


