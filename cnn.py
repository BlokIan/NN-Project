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

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Lenet 5
        # 15 * 16 -> 19 * 20 -> 15 * 16 -> 8 * 8 -> 4 * 4 -> 2 * 2
        self.hidden_layer = nn.Sequential(
            Conv2d(1, 6, 5, padding=2),
            ReLU(),
            MaxPool2d(2, 2, ceil_mode=True),
            Conv2d(6, 16, 5),
            ReLU(),
            MaxPool2d(2, 2)
        )
        self.classifier_layer = nn.Sequential(
            Linear(16*2*2, 120),
            ReLU(),
            Linear(120, 84),
            ReLU(),
            Linear(84, 10)
        )
        
        # Custom lenet 5
        # self.hidden_layer = nn.Sequential(
        #     # First layer (W*L*D = (15,16,1))
        #     Conv2d(1, 6, 9, padding=1),
        #     ReLU(),
        #     MaxPool2d(2, stride=2, ceil_mode=True),
        #     # After padding (17, 18, 1), convolution (9, 10, 6), pooling (5, 5, 6))

        #     # Second layer (W*L*D = (5,5,6))
        #     Conv2d(6, 16, 2),
        #     ReLU(),
        #     MaxPool2d(2, stride=2)
        #     # After convolution (4,4,16), pooling(2,2,16)
        # )
        # self.classifier_layer = nn.Sequential(
        #     # Input from pooling: 64 values (2*2*16 = 64)
        #     Linear(64, 64),
        #     ReLU(),
        #     Linear(64, 32),
        #     ReLU(),
        #     Linear(32, 10)
        # )

    def forward(self, x):
        x = self.hidden_layer(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        logits = self.classifier_layer(x)
        return logits