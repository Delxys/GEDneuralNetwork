import numpy as np
import json
import os
# import PIL
import torch
import torch.nn as nn
import torch.optim as optim
from NN_Model import Model
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


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
        label = torch.tensor([label])# Reshape label to have a size of [1]
        return flows, label


data_train = []
arr = os.listdir('data')

for cur_File in arr:
    with open(os.path.join('data', cur_File), 'r') as f:
        data_from_file = json.load(f)
        data_train.append(data_from_file)

batch_size = 50
data_set = MyDataset(data_train)

data_size = len(data_train)
validation_split = .3
split = int(np.floor(validation_split * data_size))
indices = list(range(data_size))
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                         sampler=val_sampler)

nn_model = Model(9, 9, 1)
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(nn_model.parameters(), lr=1e-5, weight_decay=1e-2)


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
def compute_accuracy(model, loader, testing = False):
    model.eval()  # Evaluation mode
    n_correct = 0
    n_total = 0

    for data in loader:
        inputs, labels = data
        batch_pred = model(inputs).argmax(1)
        n_correct += (batch_pred == labels).nonzero().size(0)
        n_total += labels.size(0)
    if testing:
        return batch_pred
    return n_correct / n_total


loss_history, train_history, val_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, 12)

# в конце проверяем на test set
data_train2 = []
arr2 = os.listdir('TestDataSet')

for cur_File2 in arr2:
    with open(os.path.join('TestDataSet', cur_File2), 'r') as f2:
        data_from_file = json.load(f2)
        data_train2.append(data_from_file)

data_set2 = MyDataset(data_train2)

data_size2 = len(data_train2)

test_loader = torch.utils.data.DataLoader(data_set2, batch_size=10)
actual_value = compute_accuracy(nn_model, test_loader, True)
tmp_data = data_set2.data
tmp_labels = []
for aa in tmp_data:
    tmp_labels.append(aa["label"]+1)


print("Example expected value", tmp_labels)
print("Example actual value", actual_value+1)