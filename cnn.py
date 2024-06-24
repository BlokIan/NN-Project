import os
import torch
from torch import nn
from torch.utils.data import DataLoader

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        