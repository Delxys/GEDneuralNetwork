import numpy as np
import json
import os
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
#import torchvision.datasets as dset
from NN_Model import Model
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms

data_train = []
arr = os.listdir('data')

for cur_File in arr:
    with open(os.path.join('data', cur_File), 'r') as f:
        data_from_file = json.load(f)
        data_train.append(data_from_file)

batch_size = 3

data_size = len(data_train)
validation_split = .3
split = int(np.floor(validation_split * data_size))
indices = list(range(data_size))
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                         sampler=val_sampler)

nn_model = Model(9, 9, 2)
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(nn_model.parameters(), lr=1e-2, weight_decay=1e-1)


def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs, lr_sched=None):
    loss_history = []
    train_history = []
    val_history = []
    for epoch in range(num_epochs):
        if lr_sched: lr_sched.step()
        model.train()  # Enter train mode

        loss_accum = 0 # счетчик потерь
        correct_samples = 0 # счетчик правильных примеров
        total_samples = 0 # счетчик всех примеров
        for i_step, (x,y) in enumerate(train_loader):
            prediction = model(x)
            loss_value = loss(prediction,y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            # _, indices = torch.max(prediction, 1)
            #correct_samples += torch.sum(indices == y)
            #total_samples += y.shape[0]

            loss_accum += loss_value

        ave_loss = loss_accum / (i_step + 1)
        # train_accuracy = float(correct_samples) / total_samples
        # val_accuracy = compute_accuracy(model, val_loader)

        loss_history.append(float(ave_loss))
        # train_history.append(train_accuracy)
        # val_history.append(val_accuracy)
        train_accuracy, val_accuracy=0
        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

    return loss_history, train_history, val_history
def compute_accuracy(model, loader):
    model.eval()  # Evaluation mode

    n_correct = 0
    n_total = 0

    for x, y in loader:
        batch_pred = model(x).argmax(1)
        n_correct += (batch_pred == y).nonzero().size(0)
        n_total += y.size(0)

    return n_correct / n_total


loss_history, train_history, val_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, 3)