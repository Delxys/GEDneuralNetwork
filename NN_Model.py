import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_num, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
