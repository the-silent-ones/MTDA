from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_trainsform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def imageclef_load(path, train):
    if (train):
        dataset = datasets.ImageFolder(path, transform=train_transform)
    else:
        dataset = datasets.ImageFolder(path, transform=test_trainsform)
    dataset.num_classes = len(dataset.classes)
    return dataset
