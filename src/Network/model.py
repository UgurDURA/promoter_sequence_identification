import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os

# Add the directory containing the module to the sys.path list
module_directory = os.path.abspath('../')
sys.path.insert(0, module_directory)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out