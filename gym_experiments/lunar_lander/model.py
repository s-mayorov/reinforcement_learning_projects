import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(in_features=state_size, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=action_size)
               

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        actions = self.fc3(state)
        return actions
