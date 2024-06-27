from dataloader import imageDataset
from torch.utils.data import DataLoader
import numpy as np

def prototype_images(loader):
    previous_label = -1
    for inputs, labels in loader:
        current_label = labels
        

if __name__ == "__main__":
    dataset = imageDataset("Training_data.txt")
    data_loader = DataLoader(
        dataset,
        shuffle=False
    )
