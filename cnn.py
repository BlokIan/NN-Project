import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden_stack = nn.Sequential(
            # First layer (W*L*D = (15,16,1))
            Conv2d(1, 6, 9, padding=1),
            ReLU(),
            MaxPool2d(2, stride=2, ceil_mode=True),
            # After padding (17, 18, 1), convolution (9, 10, 6), pooling (5, 5, 6))
            # Second layer (W*L*D = (5,5,6))
            Conv2d()
        )

    def forward(self, x):
        x = self.flatten(x)
        result = self.hidden_stack(x)
        return result