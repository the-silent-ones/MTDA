from torch import nn
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class ResClassifier(nn.Module):
    def __init__(self, class_num, feature_dim, bottleneck_dim=256):
        super(ResClassifier, self).__init__()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.bottleneck(x)
        y = self.fc(x)
        return x,y
