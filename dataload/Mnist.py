import os.path
from scipy.io import loadmat
import torch.utils.data
import numpy as np


class Mnist(torch.utils.data.Dataset):
    def __init__(self, root, train):
        self.train = train
        self.data_dir = os.path.join(root, "mnist_data.mat")
        mnist_data = loadmat(self.data_dir)
        if train:
            data = mnist_data['train_28']
            labels = mnist_data['label_train']
            data = data.astype(np.float32).transpose((0, 3, 1, 2))
            inds = np.random.permutation(data.shape[0])
            labels = np.argmax(labels, axis=1)
            data = data[inds]
            labels = labels[inds]
        else:
            data = mnist_data['test_28']
            labels = mnist_data['label_test']
            data = data.astype(np.float32).transpose((0, 3, 1, 2))
            labels = np.argmax(labels, axis=1)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        label = self.labels[item]
        return data, label
