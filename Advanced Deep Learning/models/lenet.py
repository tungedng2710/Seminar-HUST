import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # A classic LeNet has 2 conv layers + 2 linear layers, but we’ll keep it simple
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # input shape: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))       # -> (batch, 6, 28, 28)
        x = F.max_pool2d(x, 2)         # -> (batch, 6, 14, 14)
        x = F.relu(self.conv2(x))      # -> (batch, 16, 10, 10)
        x = F.max_pool2d(x, 2)         # -> (batch, 16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))        # -> (batch, 120)
        x = F.relu(self.fc2(x))        # -> (batch, 84)
        x = self.fc3(x)                # -> (batch, 10)
        return x