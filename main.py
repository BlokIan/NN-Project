import numpy as np
import sklearn
import torch
import torchvision
import matplotlib.pyplot as plt
from cnn import ConvolutionalNeuralNetwork
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = ("cpu")
print(f"Using {device} device")


def get_data(path="Training_data.txt") -> np.array: 
    # converts data file to array which contains 10 arrays (1 for each number)
    # each list contains 200 images
    # each image is a 15x16 array

    with open(path, "r") as f:
        data = f.read()

    size_class = 100
    data=data.replace(" ", "")
    images = np.empty((10, size_class), dtype=object)
    labels = np.empty((10, size_class), dtype=int)
    # number = np.empty(200, dtype=object)
    image = np.empty(15 * 16, dtype=np.float32)

    image_index = 0
    number_index = 0
    array_index = 0

    for value in data:
        if value == "\n":
            image = np.reshape(image, (16, 15))
            # if number.size == 0:
            #     number = image
            # else:
            #     number[number_index] = image
            #     number_index += 1
            images[array_index, number_index] = image
            labels[array_index, number_index] = array_index
            number_index += 1
            image = np.empty(15 * 16, dtype=np.float32)
            image_index = 0
            if number_index == size_class:
                # images[array_index] = number
                array_index += 1
                # number = np.empty(200, dtype=object)
                number_index = 0
        else:
            image[image_index] = float(value) / 6
            image_index += 1
    return images, labels


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
    return xdata

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # Setting the model to training mode is important for normalization but unnecessary in this situation
    # model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    data, labels = get_data()
    xtrain, xtest = convert_data_to_input(data)

    # # TRY MODEL
    # result = model(xtrain)

    model = ConvolutionalNeuralNetwork().to(device)
    # result = model(xtrain)

    # print(model)
    # print(backprop_alg(xtrain, xtest, model, epochs=3))
    # print(torch.cuda.is_available())

    learning_rate = 1e-3
    batch_size = 64
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optimSGD(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(xtrain, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(xtest, batch_size=batch_size, shuffle=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
        
    print("Done!")
