import os
import numpy as np
import sklearn
import torch
import torchvision
import matplotlib.pyplot as plt
from cnn import ConvolutionalNeuralNetwork
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset
from torch.nn import Softmax
import torch.utils.tensorboard as tensorboard
from datetime import datetime
from dataloader import imageDataset
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import random

device = ("cuda")
print(f"Using {device} device")


def plot(image, label = None, path = None) -> None:
    # plt.imshow(image, cmap='gray', aspect='auto')
    plt.imshow(image.reshape([16,15]), cmap='gray', aspect='auto')
    if label is not None:
        plt.title(f"Guessed label: {label}")
    else:
        plt.title('Grayscale Image')
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()


def createConfusionMatrix(loader, mode):
    if mode == "eval":
        model.eval()
    elif mode == "training":
        model.train(True)
    else:
        raise ValueError()
    
    y_pred = [] # save prediction
    y_true = [] # save ground truth

    # iterate over data
    for inputs, labels in loader:
        output = model(inputs)  # Feed Network

        output = (torch.argmax(softmax(output))).data.cpu().numpy()
        y_pred.append(output)  # save prediction

        labels = (torch.argmax(labels)).data.cpu().numpy()
        y_true.append(labels)  # save ground truth

    # constant for classes
    classes = ('0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],  #  / np.sum(cf_matrix, axis=1)[:, None]
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
    tb_x = None
    for i, data in enumerate(train_dataLoader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()
        max_norm(model)

        # Gather data and report
        running_loss += loss.item()
        if i % report_per_samples == report_per_samples - 1:
            last_loss = running_loss / report_per_samples
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataLoader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0

        predict = torch.argmax(softmax(outputs))
        target = torch.argmax(labels)
        if predict == target:
            correct_predictions += 1

    accuracy = correct_predictions / len(train_dataLoader)
    writer.add_scalar("Accuracy/val", accuracy, tb_x)
    writer.add_figure("Confusion matrix validation set", createConfusionMatrix(validation_dataloader, "eval"), epoch)
    writer.add_figure("Confusion matrix training set", createConfusionMatrix(train_dataLoader, "training"), epoch)
    return last_loss


def max_norm(model, max_val=10, eps=1e-8):
    # https://github.com/kevinzakka/pytorch-goodies
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))


if __name__ == "__main__":
    # Creating DataLoaders
    softmax = Softmax(dim=1)
    dataset = imageDataset("Training_data.txt")

    test_model = False
    if test_model:
        path = r"C:\Users\ianbl\OneDrive\School root\AI\Year 2\Neural Networks\NN-Project\runs\fashion_trainer_10_40\model_18"
        store_path = r"C:\Users\ianbl\OneDrive\School root\AI\Year 2\Neural Networks\NN-Project\Results\3\classified_images"
        model = ConvolutionalNeuralNetwork()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        correct_predictions = 0
        incorrect_predictions = 0
        wrong_classified = []
        data = DataLoader(
            dataset,
            shuffle=True
        )
        for i, batch in enumerate(data):
            inputs, labels = batch
            outputs = model(inputs)
            predict = torch.argmax(softmax(outputs))
            target = torch.argmax(labels)
            if predict == target:
                correct_predictions += 1
            else:
                incorrect_predictions += 1
                plot(inputs.data.cpu().numpy(), predict.data.cpu().numpy(), os.path.join(store_path, f"{incorrect_predictions}"))
        exit()

    kfolds = 1

    data = []
    for value in dataset:
        data.append(value)
    random.shuffle(data)
    folds = ([data[i::kfolds] for i in range(kfolds)])

    accuracies = []

    learning_rate = 0.005
    batch_size = 1
    epochs = 20

    for i, fold in enumerate(folds):
        if len(folds) > 1:
            train_set = []
            for j, set in enumerate(folds):
                if j == i:
                    val_set = set
                else:
                    train_set += set
        else:
            val_size = 0.2
            val_amount = int(len(dataset) * val_size)
            train_set, val_set = random_split(dataset, [
                (len(dataset) - val_amount),
                val_amount
            ])

        # learning_rate = 0.01
        # batch_size = 1
        # epochs = 1

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
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        timestamp = datetime.now().strftime('%H_%M')
        writer = tensorboard.SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        loss_fn = nn.CrossEntropyLoss() # explain in report
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        epoch_number = 0
        best_vloss = 1_000_000
        wrong_classified = []

        for epoch in range(epochs):
            print('EPOCH {}:'.format(epoch_number + 1))

            model.train(True)
            average_loss = train_one_epoch(epoch_number, writer)

            running_vloss = 0.0
            correct_predictions = 0
            with torch.no_grad():
                model.eval()
                for i, data in enumerate(validation_dataloader):
                    vinputs, vlabels = data
                    voutputs = model(vinputs)
                    predict = torch.argmax(softmax(voutputs))
                    target = torch.argmax(vlabels)
                    if predict == target:
                        correct_predictions += 1
                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            accuracy = float(correct_predictions / len(validation_dataloader))
            writer.add_scalar("Validation accuracy", accuracy, epoch)
            average_vloss = running_vloss / (i+1)
            print('LOSS train {} valid {}'.format(average_loss, average_vloss))

            writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : average_loss, 'Validation' : average_vloss },
                        epoch_number + 1)

            if average_vloss < best_vloss:
                best_vloss = average_vloss
                model_path = 'runs\\fashion_trainer_{}\\model_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)

            scheduler.step()
            epoch_number += 1
        accuracies.append(accuracy)
    print(sum(accuracies)/len(accuracies), accuracies)
    writer.flush()
