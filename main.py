import numpy as np
import sklearn
import torch
import torchvision
import matplotlib.pyplot as plt
# import cnn

device = ("cuda")
print(f"Using {device} device")


def get_data(path="mfeat-pix.txt"): 
    # converts data file to array which contains 10 arrays (1 for each number)
    # each list contains 200 images
    # each image is a 15x16 array
    with open(path, "r") as f:
        data = f.read()
    data=data.replace(" ", "")
    array = np.empty((10, 200), dtype=object)
    number = np.empty(200, dtype=object)
    image = np.empty(15 * 16, dtype=int)
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
            image = np.empty(15 * 16, dtype=int)
            image_index = 0
            if number_index == 200:
                array[array_index] = number
                array_index += 1
                number = np.empty(200, dtype=object)
                number_index = 0
        else:
            image[image_index] = int(value)
            image_index += 1
    return array


def plot(image):
    plt.imshow(image, cmap='gray', aspect='auto')
    plt.title('Grayscale Image')
    plt.show()


if __name__ == "__main__":
    data = get_data()
    training_data = data[:, :100]
    testing_data = data[:, 100:]

    training_data = training_data.reshape([15,16,-1])[:,:,None,:]
    print(f'shape of xtrain after reshape is {training_data.shape}.')

    #data should be a 4-d tensor where tensor_shape[0] = batch_size, tensor_shape[1] = 1 (grayscale channels), tensor_shape[2] = 16 (height), tensor_shape[3] = 15 (width)

    # batch_size = 32
    # trainloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    # testloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    # model = cnn.ConvolutionalNeuralNetwork()


    # plot(data[0][0])
    # plot(data[1][0])
    # plot(data[2][0])
    # plot(data[3][0])
    # plot(data[4][0])
    # plot(data[5][0])
    # plot(data[6][0])
    # plot(data[7][0])
    # plot(data[8][0])
    # plot(data[9][0])
