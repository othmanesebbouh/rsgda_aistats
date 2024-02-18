import torch

from torch import nn
from torch.nn import functional as F
import torchvision.models as models


def get_classifier(clf_name):
    if clf_name == "ConvElu":
        model = ConvElu()
    return model


class ConvElu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 8, 1, padding='same')
        self.conv2 = nn.Conv2d(64, 128, 6, 2, padding='valid')
        self.conv3 = nn.Conv2d(128, 128, 5, 1, padding='valid')
        self.fc = nn.Linear(8192, 10)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x




