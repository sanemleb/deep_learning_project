import matplotlib.pyplot as plt
import numpy as np
import random
from testDataset import TestCarDataset 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'



def plot_transformations(TestCarDataset, idx):
    # Get the transformed image and mask
    transformed_img, transformed_mask = TestCarDataset[idx]

    # Load the original image and mask for comparison
    original_mask = np.load(TestCarDataset.all_mask_paths[idx])
    original_img = original_mask[:, :, :3]
    original_mask = original_mask[:, :, 3] // 10
    original_mask = original_mask.astype(int)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Original Image
    axs[0, 0].imshow(original_img)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Transformed Image
    transformed_img_np = transformed_img.numpy().transpose(1, 2, 0) 
    axs[0, 1].imshow(transformed_img_np)
    axs[0, 1].set_title('Transformed Image')
    axs[0, 1].axis('off')

    # Original Mask
    axs[1, 0].imshow(original_mask, cmap='gray')  
    axs[1, 0].set_title('Original Mask')
    axs[1, 0].axis('off')

    # Transformed Mask
    transformed_mask_np = transformed_mask.numpy()
    axs[1, 1].imshow(transformed_mask_np, cmap='gray')  
    axs[1, 1].set_title('Transformed Mask')
    axs[1, 1].axis('off')

    plt.show()

random_idx = random.randint(0, len(TestCarDataset) - 1)
plot_transformations(TestCarDataset, random_idx)