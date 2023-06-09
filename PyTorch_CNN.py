import numpy as np
import json
import os
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler


data_train=[]
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