import torch
import numpy as np

device = ("cuda")

class imageDataset(torch.utils.data.Dataset): 
    def __init__(self, path):
        image_size = 240
        with open(path, "r") as f:
            data = f.read()
        data=data.replace(" ", "")
        data=data.replace("\n", "")
        self.labels = np.zeros((len(data)//image_size, 10)) #n_images x 10 matrix
        j = -1
        self.images = np.zeros(len(data)//image_size, dtype=object)
        idx = 0
        image = np.empty((16, 15), dtype=np.float32)
        for i, label in enumerate(self.labels):
            if i % 100 == 0:
                j += 1
            label[j] = 1
            m = i * image_size # index for parsing .txt file
            for k in range(16):
                for l in range(15):
                    image[k][l] = data[m]
                    m += 1

            self.images[idx] = image
            idx += 1
            image = np.empty((16, 15), dtype=np.float32)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = image.reshape([16,15,1])
        image = np.moveaxis(image, 2, 0)
        image = torch.from_numpy(image).to(device)
        label = torch.from_numpy(label).to(device)
        return image, label

    def __len__(self):
        return len(self.labels)