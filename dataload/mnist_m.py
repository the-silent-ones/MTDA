import torch.utils.data
import os
from PIL import Image
from torchvision import transforms

'''
eg/
train_dataset = MNIST_M(root=root_dir, train=True, transform=composed_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
'''
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
class Mnist_m(torch.utils.data.Dataset):
    def __init__(self, root, train):
        self.train = train
        self.transform = transform
        if train:
            self.image_dir = os.path.join(root, 'mnist_m_train')
            labels_file = os.path.join(root, "mnist_m_train_labels.txt")
        else:
            self.image_dir = os.path.join(root, 'mnist_m_test')
            labels_file = os.path.join(root, "mnist_m_test_labels.txt")

        with open(labels_file, "r") as fp:
            content = fp.readlines()
        self.mapping = list(map(lambda x: (x[0], int(x[1])), [c.strip().split() for c in content]))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, item):
        image, label = self.mapping[item]
        image = os.path.join(self.image_dir, image)
        image = Image.open(image).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            label = int(label)

        return image, label
