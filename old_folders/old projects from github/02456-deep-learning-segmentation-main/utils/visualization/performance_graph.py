import numpy as np 
import matplotlib.pyplot as plt 

train_log = np.load('E:/GitHub/02456-deep-learning-segmentation/models/train_log.npy')
valid_log = np.load('E:/GitHub/02456-deep-learning-segmentation/models/valid_log.npy')

plt.clf()
plt.plot(train_log, label="Training")
plt.plot(valid_log, label="Validation")
plt.xlabel('Epochs')
plt.ylabel('Micro Dice Loss')
plt.legend()
plt.show()