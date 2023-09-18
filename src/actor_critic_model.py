import torch
import torch.nn as nn
from typing import Tuple, List, Any


class ContinuousActorCriticNetwork(nn.Module):
    """
    Continuous Actor Critic Network.
    The actor and critic are both feed forward neural networks.
    They share a common set of layers.
    The actor takes the state/observation and outputs the mean and standard deviation of the action distribution.
    The standard deviation is processed to stay in a predefined range and to ensure that it is always positive.
    The critic takes the state/observation and outputs the value estimate.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: List = None,
        action_std_bound: Tuple = (-20, 2),
    ):
        """
        Initialize the network.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_units (List): List of integers representing the number of units in the hidden layers.
            action_std_bound (Tuple): Tuple of two floats for the action standard deviation bounds.
        """
        super(ContinuousActorCriticNetwork, self).__init__()
        if hidden_units is None:
            hidden_units = [64, 64]

        self.action_std_bound = action_std_bound

        # Construct shared layers dynamically
        self.shared_layers = self.build_shared_layers(state_dim, hidden_units)

        # Actor layers
        self.actor_mu = nn.Linear(hidden_units[-1], action_dim)
        self.actor_log_std = nn.Parameter(
            torch.zeros(action_dim)
        )  # Using nn.Parameter to make it learnable

        # Critic layers
        self.critic = nn.Linear(hidden_units[-1], 1)

    def build_shared_layers(self, state_dim: int, hidden_units: List) -> Any:
        """Build shared neural network layers.

        Args:
            state_dim (int): Dimension of the state space.
            hidden_units (List): List of integers representing the number of units in the hidden layers.

        Returns:
            nn.Sequential: The sequential layers.

        """
        layers = []
        input_dim = state_dim

        for units in hidden_units:
            layers.extend([nn.Linear(input_dim, units), nn.ReLU()])
            input_dim = units

        return nn.Sequential(*layers)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            state (torch.Tensor): The observations/states

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): The mean, standard deviation for action distrubution and value estimate from the critic.

        """
        features = self.shared_layers(state)
        mu = self.actor_mu(features)

        log_std = torch.clamp(
            self.actor_log_std, self.action_std_bound[0], self.action_std_bound[1]
        )
        std = torch.nn.functional.softplus(log_std)

        value_estimate = self.critic(features)
        return mu, std, value_estimate
