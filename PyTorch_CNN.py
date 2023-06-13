import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from NN_Model import Model
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        flows = item['flows']
        label = item['label']
        flows = torch.stack([torch.tensor([flow['id'], flow['sourceId'], flow['destinationId'], flow['name'], flow['measured'], flow['upperBound'], flow['lowerBound'], flow['tolerance'], flow['isMeasured']]) for flow in flows])
        label = torch.tensor([label])  # Reshape label to have a size of [1]
        return flows, label


# Loading and preparing the training data
data_train = []
arr = os.listdir('data')

for cur_File in arr:
    with open(os.path.join('data', cur_File), 'r') as f:
        data_from_file = json.load(f)
        data_train.append(data_from_file)


batch_size = 110
data_set = MyDataset(data_train)
data_size = len(data_train)
validation_split = .2
split = int(np.floor(validation_split * data_size))
indices = list(range(data_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# Creating data loaders for training and validation
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                         sampler=val_sampler)
# Creating the neural network model
nn_model = Model(9, 9, 1)
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(nn_model.parameters(), lr=1e-5, weight_decay=1e-3)


# Training the model
def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs, lr_sched=None):
    loss_history = []
    train_history = []
    val_history = []
    for epoch in range(num_epochs):
        if lr_sched: lr_sched.step()

        model.train()  # Enter train mode
        loss_accum = 0  # Loss accumulator
        correct_samples = 0  # Number of correct samples
        total_samples = 0  # Total number of samples
        for batch_index, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            output = model(inputs)
            loss_value = loss(output, labels)
            loss_value.backward()
            optimizer.step()

            _, indices = torch.max(output, 1)
            correct_samples += torch.sum(indices == labels)
            total_samples += labels.shape[0]

            loss_accum += loss_value

        ave_loss = loss_accum / (batch_index + 1)
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(model, val_loader)

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)

        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

    return loss_history, train_history, val_history


def compute_accuracy(model, loader):
    model.eval()  # Evaluation mode
    n_correct = 0
    n_total = 0

    for data in loader:
        inputs, labels = data
        batch_pred = model(inputs).argmax(1)
        n_correct += (batch_pred == labels).nonzero().size(0)
        n_total += labels.size(0)
    return n_correct / n_total


def get_predictions(model, loader):
    for data in loader:
        inputs, labels = data
        batch_predictions = model(inputs).argmax(1)
    return batch_predictions


# Training the model
loss_history, train_history, val_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, 35)


# Testing the model on a separate test set
def detect_error_flows():
    data_train2 = []
    arr2 = os.listdir('TestDataset')

    for cur_File2 in arr2:
        with open(os.path.join('TestDataset', cur_File2), 'r') as f2:
            data_from_file2 = json.load(f2)
            data_train2.append(data_from_file2)

    data_set2 = MyDataset(data_train2)
    data_size2 = len(data_train2)
    test_loader = torch.utils.data.DataLoader(data_set2, batch_size=data_size2)
    actual_value = get_predictions(nn_model, test_loader)
    for element in actual_value:
        if element[0] == 0:
            element[0] = -1

    dset_data = data_set2.data
    exp_labels = []
    for output in dset_data:
        if output["label"] == 0:
            exp_labels.append(output["label"])
        else:
            exp_labels.append(output["label"]+1)

    return exp_labels, actual_value+1


user_input = 'y'
while isinstance(user_input, str):
    if len(os.listdir('TestDataset')) > 0 and user_input == 'y':
        start = time.time()
        expected, actual = detect_error_flows()
        end = time.time()
        act_arr = torch.reshape(actual, (1, -1))
        print("Time: ", end - start)
        print("Example expected value ", expected)
        print("Example actual value ", act_arr)
        user_input = input("Start detection again? y/n: ")
    elif user_input == 'n':
        print("Shutdown")
        break
    else:
        print("Incorrect input")
        break


