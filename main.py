import numpy as np
import sklearn
import torch
import torchvision
import matplotlib.pyplot as plt
from cnn import ConvolutionalNeuralNetwork

device = ("cuda")
print(f"Using {device} device")


def get_data(path="Training_data.txt") -> np.array: 
    # converts data file to array which contains 10 arrays (1 for each number)
    # each list contains 200 images
    # each image is a 15x16 array
    with open(path, "r") as f:
        data = f.read()
    data=data.replace(" ", "")
    size_class = 100
    array = np.empty((10, size_class), dtype=object)
    number = np.empty(size_class, dtype=object)
    image = np.empty(15 * 16, dtype=np.float32)
    image_index = 0
    number_index = 0
    array_index = 0
    for value in data:
        if value == "\n":
            image = np.reshape(image, (16, 15))
            if number.size == 0:
                number = image
            else:
                number[number_index] = image
                number_index += 1
            image = np.empty(15 * 16, dtype=np.float32)
            image_index = 0
            if number_index == size_class:
                array[array_index] = number
                array_index += 1
                number = np.empty(size_class, dtype=object)
                number_index = 0
        else:
            image[image_index] = float(value) / 6
            image_index += 1
    return array


def plot(image) -> None:
    plt.imshow(image, cmap='gray', aspect='auto')
    plt.title('Grayscale Image')
    plt.show()


def convert_data_to_input(data):
    tensor = 0
    n = 0
    for number in data:
        for image in number:
            image = image.reshape([16,15,-1])[:,:,None,:]
            image = np.moveaxis(image, (2,3), (1,0))
            if type(tensor) == int:
                tensor = image
            else:
                tensor = np.concatenate((tensor, image), axis = 0)
    xdata = torch.from_numpy(tensor).to(device)

    ldata = torch.zeros((len(data), 10))
    print()
    for i in range(10):
        ldata[i*(len(data)//10):i*(len(data)//10)+(len(data)//10)][i] = 1
    print(ldata)
    return xdata #implement labels


def backprop_alg(train_data, test_data, net, epochs, mini_batches=100, gamma=.001, rho=.9):
    '''
    Args:
    xtrain: training samples
    ltrain: training labels
    net: neural network
    epochs: number of epochs
    mini_batches: minibatch size
    gamma: step size (learning rate)
    rho: momentum
    '''

    N = xtrain.size()[0]

    pass


if __name__ == "__main__":
    data = get_data()
    xdata = convert_data_to_input(data)

    # TRY MODEL
    model = ConvolutionalNeuralNetwork().to(device)
    result = model(xdata)
    print(result)
    print(type(result))