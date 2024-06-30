from dataloader import imageDataset
from torch.utils.data import DataLoader
from main import plot
import torch
import os

dataset = imageDataset("Training_data.txt")
data = DataLoader(
            dataset,
            shuffle=True
        )
store_path = r"C:\Users\ianbl\OneDrive\School root\AI\Year 2\Neural Networks\NN-Project\Training_samples"

for i, datapoint in enumerate(data):
    input, label = datapoint
    plot(input.data.cpu().numpy(), torch.argmax(label).data.cpu().numpy(), os.path.join(store_path, f"{i}"))