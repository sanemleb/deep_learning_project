
import torch
from torch.utils.data import DataLoader, random_split
# imports the prcessed data
from processing import *



def image_standard_format(path):
    train_one_hot, train_image, train_image_standard, train_image_standard_hot = train_val_image(path)
    return train_one_hot, train_image, train_image_standard, train_image_standard_hot

def test_images(train_one_hot, train_image, train_image_standard, train_image_standard_hot):
    test_one_hot = train_one_hot[-10:]
    test_image = train_image[-10:]
    test_image_standard = train_image_standard[-10:]
    test_image_standard_hot = train_image_standard_hot[-10:]
    return test_one_hot, test_image, test_image_standard, test_image_standard_hot



def batch(train_image_standard, train_image_standard_hot):
    batch_size = 9
    train_nr = len(train_image_standard) - int(0.2* len(train_image_standard))
    valid_nr = len(train_image_standard) - train_nr
    print(" train nr",train_nr)
    print(" Val nr ", valid_nr)
    X_t = torch.from_numpy(np.array(train_image_standard[:train_nr + valid_nr], dtype = 'float32'))
    print(X_t.shape)
    Y_t = torch.from_numpy(np.array(train_image_standard_hot[:train_nr + valid_nr], dtype ='float32'))
    print(Y_t.type())
    # Just checking you have as many labels as inputs
    assert X_t.shape[0] == Y_t.shape[0]
    dset = torch.utils.data.TensorDataset(X_t, Y_t)
    train, valid = random_split(dset,[train_nr, valid_nr])
    trainloader = torch.utils.data.DataLoader(train,
                                    batch_size=batch_size, # choose your batch size
                                    shuffle=True) # generally a good idea
    validloader = torch.utils.data.DataLoader(valid,
                                    batch_size=batch_size, # choose your batch size
                                    shuffle=True) # generally a good idea
    print("trainloader length 1", len(trainloader))
    print("validloader length 1", len(validloader))
    return trainloader, validloader
