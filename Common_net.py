import torch.nn as nn

class Feature(nn.Module):
    def __init__(self, resnet_func, pretrained):
        super(Feature, self).__init__()
        self.sharedNet = resnet_func(pretrained)

    def forward(self, input):
        feature = self.sharedNet(input)
        return feature


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, input):
        d_out = self.D(input)
        return d_out


class Classifier(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Classifier, self).__init__()
        self.C = nn.Linear(input_dim, out_dim)

    def forward(self, input):
        c_out = self.C(input)
        return c_out

