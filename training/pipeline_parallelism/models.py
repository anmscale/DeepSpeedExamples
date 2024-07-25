
import torch
import torch.nn as nn
import os

class MLP(nn.Module):
    def __init__(self, feature_size, num_blocks=4, num_classes=10):
        super(MLP, self).__init__()

        self.num_blocks = num_blocks
        
        # Create a list to hold all the modules
        self.layers = []
        for block_idx in range(self.num_blocks):
            torch.manual_seed(block_idx)
            self.layers.append(nn.Linear(feature_size, feature_size))
            self.layers.append(nn.ReLU(inplace=True))
            # self.layers.append(nn.Dropout())
            self.layers.append(nn.Linear(feature_size, feature_size))
            self.layers.append(nn.ReLU(inplace=True))
            # self.layers.append(nn.Dropout())

        # Add the final output layer
        self.layers.append(nn.Linear(feature_size, num_classes))

        self.model = nn.Sequential(*self.layers)

    def load_state_dicts(self, checkpoint_dir):
        """
        Load state dicts for every layer from files in the given directory.

        Args:
            checkpoint_dir (str): Directory containing the state_dict files.
        """
        for i in range(self.num_blocks * 4 + 1):
            state_dict_path = os.path.join(checkpoint_dir, f'layer_{i:02d}-model_states.pt')
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path)
                self.model[i].load_state_dict(state_dict)
            else:
                print(f"State dict file {state_dict_path} not found.")
        print("Model state dictionaries loaded successfully!")

    def forward(self, x):
        for stage in self.stage_layers:
            x = stage(x)
        return x
