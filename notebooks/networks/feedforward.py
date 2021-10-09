from torch import nn
from torch.nn import functional as F
import torch

class FNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super().__init__()
        self.max = 5
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc_last = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_dim))
        # n, d = q.shape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_last(x)
        return x
