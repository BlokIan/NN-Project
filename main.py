import numpy as np
import sklearn
import torch
import torchvision
import matplotlib.pyplot as plt
from cnn import ConvolutionalNeuralNetwork

device = ("cuda")
print(f"Using {device} device")


def get_data(path="mfeat-pix.txt") -> np.array: 
    # converts data file to array which contains 10 arrays (1 for each number)
    # each list contains 200 images
    # each image is a 15x16 array
    with open(path, "r") as f:
        data = f.read()
    data=data.replace(" ", "")
    array = np.empty((10, 200), dtype=object)
    number = np.empty(200, dtype=object)
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
            if number_index == 200:
                array[array_index] = number
                array_index += 1
                number = np.empty(200, dtype=object)
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
    training_data = data[:, :100]
    testing_data = data[:, 100:]
    tensor = 0
    for number in training_data:
        for image in number:
            image = image.reshape([16,15,-1])[:,:,None,:]
            image = np.moveaxis(image, (2,3), (1,0))
            if type(tensor) == int:
                tensor = image
            else:
                tensor = np.concatenate((tensor, image), axis = 0)

    xtrain = torch.from_numpy(tensor).to(device)
    return xtrain


def backprop_alg(train_data, test_data, net, epochs, mini_batches=100, gamma=.001, rho=.9):
        
    pass


if __name__ == "__main__":
    data = get_data()
    xtrain = convert_data_to_input(data)
    # TRY MODEL
    model = ConvolutionalNeuralNetwork().to(device)
    result = model(xtrain)
    print(result)
    print(np.shape(result))