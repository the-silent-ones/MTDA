import os.path
from scipy.io import loadmat
import torch.utils.data
import numpy as np
from utils import dense_to_one_hot
class SVHN(torch.utils.data.Dataset):
    def __init__(self,root,train):
        self.train = train
        if(train):
            svhn_data = loadmat(os.path.join(root,"train_32x32.mat"))
        else:
            svhn_data = loadmat(os.path.join(root,"test_32x32.mat"))
        svhn_img = svhn_data['X'].transpose(3, 2, 0, 1).astype(np.float32)
        svhn_label = dense_to_one_hot(svhn_data['y'])
        self.img = svhn_img
        self.label = svhn_label

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]
        return img,label