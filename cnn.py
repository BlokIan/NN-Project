import os
import torch
import torch.optim as optim

from torch import nn
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import LeakyReLU
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.nn import Dropout, Dropout2d, Dropout1d
from torch.nn.utils import parametrize


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Lenet 5
        # 15 * 16 -> 19 * 20 -> 15 * 16 -> 8 * 8 -> 4 * 4 -> 2 * 2
        self.conv_layer = nn.Sequential(
            Conv2d(1, 6, 5, padding=2),
            Dropout(p=0.1),
            ReLU(),
            MaxPool2d(2, 2, ceil_mode=True),
            Conv2d(6, 16, 5),
            Dropout(p=0.1),
            ReLU(),
            MaxPool2d(2, 2)
        )

        self.classifier_layer = nn.Sequential(
            Linear(16*2*2, 120),
            Dropout(p=0.1),
            ReLU(),
            Linear(120, 84),
            Dropout(p=0.1),
            ReLU(),
            Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        logits = self.classifier_layer(x)
        return logits