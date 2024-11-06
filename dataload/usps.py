import gzip
import os.path
from scipy.io import loadmat
import torch.utils.data
import numpy as np
import pickle as cPickle
class USPS(torch.utils.data.Dataset):
    def __init__(self,root,train):
        self.train = train
        f = gzip.open(os.path.join(root,"usps_28x28.pkl"),'rb')
        data = cPickle.load(f,encoding='iso-8859-1')
        f.close()
        if(train):
            img = data[0][0]
            label = data[0][1]
            inds = np.random.permutation(img.shape[0])
            img = img[inds][:]
            label = label[inds][:]
        else:
            img = data[1][0]
            label = data[1][0]
        self.img = img
        self.label = label
    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]
        return img,label