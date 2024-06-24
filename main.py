import numpy as np
import sklearn
import torch
import torchvision

def get_data(path="mfeat-pix.txt"): 
    # converts data file to array which contains 10 lists (1 for each number)
    # each list contains 200 images
    # each image consists of 240 pixels
    with open(path, "r") as f:
        data = f.read()
    data=data.replace(" ", "")
    array = []
    number = []
    image = []
    for value in data:
        if value == "\n":
            number.append(image)
            image = []
            if len(number) == 200:
                array.append(number)
                number = []
        else:
            image.append(int(value))
    return array

def plot_image():
    pass
        


x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())

