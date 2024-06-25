import torch
import numpy as np
import random
#abstract class for a dataset
class imageDataloader(torch.utils.data.Dataset): 
    def __init__(self, path, batch_size=50):
        self.batch_size = batch_size
        self.image_size = 240
        self.index = 0
        with open(path, "r") as f:
            self.data = f.read()
        self.data=self.data.replace(" ", "")
        self.data=self.data.replace("\n", "")
        self.labels = np.zeros((len(self.data)//self.image_size, 10)) #n_images x 10 matrix
        j = -1
        images = np.zeros(len(self.data)//self.image_size)
        idx = 0
        for i, label in enumerate(self.labels):
            if i % 100 == 0:
                j += 1
            label[j] = 1
            images[idx] = self.data[idx*self.image_size:idx*self.image_size+self.image_size]
            idx += 1

    def __getitem__(self, index):
        image, label = self.xy[index]
        return image, label

    def __len__(self):
        pass

# load = imageDataloader("Training_data.txt")
# print(next(load))
# print(next(load))
# print(next(load))
# print(next(load))
