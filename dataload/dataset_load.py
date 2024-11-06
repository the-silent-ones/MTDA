import os.path

import torch.utils.data

from dataload.auto_augment import AutoAugment
import torch.utils.data as data
from torchvision import datasets
from torchvision import transforms as T
from dataload import Mnist
from dataload import mnist_m
from dataload import svhn
from dataload import usps


def convert_dataset(dataset):
    """
    Converts a dataset which returns (img, label) pairs into one that returns (index, img, label) triplets.
    """

    class DatasetWrapper:

        def __init__(self):
            self.dataset = dataset

        def __getitem__(self, index):
            return index, self.dataset[index]

        def __len__(self):
            return len(self.dataset)

    return DatasetWrapper()


def load_digits_dataset(data_name, train, batch_size=16, num_workers=1, pin_memory=True, drop_last=False,
                        shuffle=True):
    if data_name == 'mnist':
        datas = Mnist("dataset/MNIST", train)

    elif data_name == 'mnist_m':
        datas = mnist_m("dataset/mnist_m", train)

    elif data_name == "svhn":
        datas = svhn("dataset/svhn", train)

    elif data_name == "usps":
        datas = usps("dataset/usps", train)

    dataloader = data.DataLoader(datas,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 drop_last=drop_last)

    return dataloader


def office_test_loader(path, batch_size=16, num_workers=1, pin_memory=True, transform=None, shuffle=True):
    if transform is None:
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ])
    else:
        transforms = transform

    dataset = datasets.ImageFolder(path, transforms)
    return data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)


def office_train_loader(path, batch_size=16, num_workers=1, pin_memory=True, drop_last=False, transform=None,
                        shuffle=True):
    if transform is None:
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms = T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    else:
        transforms = transform
    dataset = datasets.ImageFolder(path, transforms)
    dataset.num_classes = len(dataset.classes)

    return data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last=drop_last)


"""
eg:
path = "dataset/office31/amazon"
...
"""


def load_office_dataset(path, train, batch_size=16, num_workers=1, pin_memory=True, drop_last=False, transform=None,
                        shuffle=True):
    if train:
        dataloader = office_train_loader(path, batch_size, num_workers, pin_memory, drop_last, transform,
                                                  shuffle=shuffle)
    else:
        dataloader = office_test_loader(path, batch_size, num_workers, pin_memory, transform, shuffle=shuffle)
    return dataloader


def load_visda_dataset(path, train, batch_size=16, num_workers=1, pin_memory=True, drop_last=False, transform=None,
                       shuffle=True):
    if train:
        dataloader = visda_train_loader(path, batch_size, num_workers, pin_memory, drop_last, transform,
                                                 shuffle=shuffle)
    else:
        dataloader = visda_test_loader(path, batch_size, num_workers, pin_memory, transform, shuffle=shuffle)
    return dataloader


def visda_train_loader(path, batch_size=16, num_workers=1, pin_memory=True, drop_last=False, transform=None,
                       shuffle=True):
    if transform is None:
        transforms = T.Compose([
            T.Resize((256, 256)),
            #T.RandomCrop((224, 224)),
            T.RandomHorizontalFlip(),
            AutoAugment(),
            T.ToTensor(),
            T.Normalize((0.435, 0.418, 0.396), (0.284, 0.308, 0.335)),  # grayscale mean/std
        ])
    else:
        transforms = transform
    dataset = datasets.ImageFolder(path, transforms)
    dataset.num_classes = len(dataset.classes)

    return data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last=drop_last)


def visda_test_loader(path, batch_size=16, num_workers=1, pin_memory=True, transform=None, shuffle=True):
    if transform is None:
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.435, 0.418, 0.396), (0.284, 0.308, 0.335)),  # grayscale mean/std
        ])
    else:
        transforms = transform

    dataset = datasets.ImageFolder(path, transforms)

    return data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)

def load_all_can_use(path, train, batch_size=16, num_workers=1, pin_memory=True, drop_last=False, transform=None,
                     shuffle=True):
    if transform is None:
        if train:
            transforms = T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop((224, 224)),
                T.RandomHorizontalFlip(),
                AutoAugment(),
                T.ToTensor(),
                T.Normalize((0.435, 0.418, 0.396), (0.284, 0.308, 0.335)),  # grayscale mean/std
            ])
            drop_last = True
        else:
            transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.435, 0.418, 0.396), (0.284, 0.308, 0.335)),  # grayscale mean/std
            ])
            drop_last = False
    else:
        transforms = transform
    dataset = datasets.ImageFolder(path, transforms)
    return data.DataLoader(dataset, batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

def data_loader_split(root, data_name, train_rate, batch_size=16, num_workers=1, pin_memory=True, drop_last=False):
    if data_name == "officehome":
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        dataset = datasets.ImageFolder(root,
                                       T.Compose([
                                           T.Resize(256),
                                           T.RandomResizedCrop(224),
                                           T.RandomHorizontalFlip(),
                                           T.ToTensor(),
                                           normalize,
                                       ]))
        train_num = int(train_rate * len(dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_num, len(dataset) - train_num])
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory,
                                       drop_last=drop_last)
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
        return train_loader, test_loader
