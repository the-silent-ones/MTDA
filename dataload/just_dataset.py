from dataload.auto_augment import AutoAugment
from torchvision import datasets, transforms
from dataload import Mnist
from dataload import mnist_m
from dataload import svhn
from dataload import usps


def load_digit_dataset(data_name, train):
    if data_name == 'mnist':
        datas = Mnist("dataset/MNIST", train)

    elif data_name == 'mnist_m':
        datas = mnist_m("dataset/mnist_m", train)

    elif data_name == "svhn":
        datas = svhn("dataset/svhn", train)

    elif data_name == "usps":
        datas = usps("dataset/usps", train)

    return datas

def office_loader(path,transform=None):
    if transform is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        dataset = datasets.ImageFolder(path,
                                       transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize,
                                       ]))
    else:
        dataset = datasets.ImageFolder(path,transform)
    dataset.num_classes = len(dataset.classes)

    return dataset



def visda_loader(path,transform):
    if transform is None:
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.435, 0.418, 0.396), (0.284, 0.308, 0.335)),  # grayscale mean/std
        ])
        dataset = datasets.ImageFolder(path,transform_train)
    else:
        dataset = datasets.ImageFolder(path,transform)
    dataset.num_classes = len(dataset.classes)

    return dataset

def all_use_dataset_loader(path,transform):
    if transform is None:
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.435, 0.418, 0.396), (0.284, 0.308, 0.335)),  # grayscale mean/std
        ])
        dataset = datasets.ImageFolder(path, transform_train)
    else:
        dataset = datasets.ImageFolder(path, transform)
    dataset.num_classes = len(dataset.classes)

    return dataset