import torch
from torch.nn import Linear

class Router(torch.nn.Module):
    def __init__(self, input_dim=1024, output_dim=3):
        super(Router, self).__init__()
        self.fc1 = Linear(input_dim, 256)
        self.fc2 = Linear(256, 64)
        self.fc3 = Linear(64, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x