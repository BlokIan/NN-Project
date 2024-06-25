import numpy as np
import sklearn
import torch
import torchvision
import matplotlib.pyplot as plt
from cnn import ConvolutionalNeuralNetwork
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = ("cuda")
print(f"Using {device} device")


def get_data(path) -> np.array:
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
    return xdata


def train_one_epoch(epoch_index, tb_writer) -> float:
    """Trains the model on one epoch. Computes average loss per batch, and reports this.

    Args:
        epoch_index (int): To keep track of the amount of data used (?)
        tb_writer (): Used to report results

    Returns:
        float: The last average loss
    """

    running_loss = 0
    last_loss = 0
    report_per_samples = 100 # Specifies per how many samples you want to report

    for i, inputs, labels in enumerate(dataLoader):
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % report_per_samples == report_per_samples - 1:
            last_loss = running_loss / report_per_samples
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataLoader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0

    return last_loss


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
    data, labels = get_data("Training_data.txt")    
    xtrain = convert_data_to_input(data)

    learning_rate = 1e-3
    batch_size = 64
    epochs = 10

    model = ConvolutionalNeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss() # explain in report
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
 
    # CREATE DATALOADERS HERE
    

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0
    best_vloss = 1_000_000

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        average_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        with torch.no_grad():
            for i, vinputs, vlabels in enumerate(validation_loader):
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        average_vloss = running_vloss / (i+1)
        print('LOSS train {} valid {}'.format(average_loss, average_vloss))

        writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : average_loss, 'Validation' : average_vloss },
                    epoch_number + 1)
        writer.flush()

        if average_vloss < best_vloss:
            best_vloss = average_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
