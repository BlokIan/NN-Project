import torch
import numpy as np
#abstract class for a dataset
class imageDataloader(torch.utils.data.Dataset): 
    def __init__(self, path, batch_size):
        self.batch_size = batch_size
        self.index = 0
        image_size = 240
        with open(path, "r") as f:
            self.data = f.read()
        data=data.replace(" ", "")
        self.labels = np.zeros((len(data)/image_size, 10)) #n_images x 10 matrix
        for i in self.labels:
            print(i)

    def __getitem__(self):



    def __next__(self):



    def __len__(self):