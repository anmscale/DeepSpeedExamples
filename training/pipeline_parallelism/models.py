
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, stages, num_layers=8, num_classes=10):
        super(MLP, self).__init__()
        assert num_layers % stages == 0, "The number of stages must be a divisor of the number of layers."

        self.stages = stages
        self.num_layers = num_layers

        # Create a list to hold all the modules
        self.layers = [nn.Flatten()]
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(3072, 3072))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout())

        # Add the final output layer
        self.layers.append(nn.Linear(3072, num_classes))

        # Crate model and split into stages
        self.model = nn.Sequential(*self.layers)
        self.stage_layers = self._split_into_stages()

    def _split_into_stages(self):
        layers_per_stage = (self.num_layers // self.stages) * 3
        stage_layers = []

        # First stage (Flatten + first layers)
        stage_layers.append(nn.Sequential(*self.layers[:1 + layers_per_stage]))

        # Intermediate stages
        for i in range(1, self.stages - 1):
            start = 1 + (i - 1) * layers_per_stage
            end = 1 + i * layers_per_stage
            stage_layers.append(nn.Sequential(*self.layers[start:end]))

        # Last stage (last layers + final output layer)
        stage_layers.append(nn.Sequential(*self.layers[-layers_per_stage-1:]))

        return stage_layers

    def forward(self, x):
        for stage in self.stage_layers:
            x = stage(x)
        return x
