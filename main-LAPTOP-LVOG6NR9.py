import numpy as np
import sklearn
import torch
import torchvision
import matplotlib.pyplot as plt
from cnn import ConvolutionalNeuralNetwork
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.utils.tensorboard as tensorboard
from datetime import datetime
from dataloader import imageDataset
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


device = ("cuda")
print(f"Using {device} device")


def plot(image) -> None:
    # plt.imshow(image, cmap='gray', aspect='auto')
    plt.imshow(image.reshape([16,15]), cmap='gray', aspect='auto')
    plt.title('Grayscale Image')
    plt.show()


def createConfusionMatrix(loader):
    y_pred = [] # save prediction
    y_true = [] # save ground truth

    # iterate over data
    for inputs, labels in loader:
        output = model(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.append(output[0])  # save prediction

        labels = np.argmax(labels.data.cpu().numpy())
        y_true.append(labels)  # save ground truth

    # constant for classes
    classes = ('0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()


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
    correct_predictions = 0
    for i, data in enumerate(train_dataLoader):
        inputs, labels = data
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
            tb_x = epoch_index * len(train_dataLoader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0
        predict = torch.argmax(outputs)
        target = torch.argmax(labels)
        if predict == target:
            correct_predictions += 1

    accuracy = correct_predictions / len(train_dataLoader)
    print(accuracy)
    writer.add_scalar("Accuracy/val", accuracy, tb_x)
    writer.add_figure("Confusion matrix validation set", createConfusionMatrix(validation_dataloader), epoch)
    writer.add_figure("Confusion matrix training set", createConfusionMatrix(train_dataLoader), epoch)
    return last_loss


if __name__ == "__main__":
    # Creating DataLoaders
    dataset = imageDataset("Training_data.txt")
    val_size = 0.2
    val_amount = int(len(dataset) * val_size)
    train_set, val_set = random_split(dataset, [
        (len(dataset) - val_amount),
        val_amount
    ])

    learning_rate = 0.01
    batch_size = 1
    epochs = 20

    train_dataLoader = DataLoader( #explain in report
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    validation_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True
    )

    model = ConvolutionalNeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss() # explain in report
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    timestamp = datetime.now().strftime('%H_%M')
    writer = tensorboard.SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0
    best_vloss = 1_000_000

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        average_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        with torch.no_grad():
            for i, data in enumerate(validation_dataloader):
                vinputs, vlabels = data
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
            model_path = 'runs\\fashion_trainer_{}\\model_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
