import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class RegressNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(RegressNet, self).__init__()
        self.block1 = torch.nn.Linear(n_feature, n_hidden)   # hidden layer1
        self.block2 = torch.nn.Linear(n_hidden, n_hidden)    # hidden Layer 2
        self.block3 = torch.nn.Linear(n_hidden, n_hidden) 
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.block1(x))      # activation function for hidden layer
        x = F.relu(self.block2(x))
        x = F.relu(self.block3(x))
        x = self.predict(x)             # linear output
        return x