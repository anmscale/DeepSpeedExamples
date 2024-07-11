
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.module_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(3072, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.module_2 = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(3072, num_classes),
        )

    def forward(self, x):
        x = self.module_1(x)
        x = self.module_2(x)
        return x
