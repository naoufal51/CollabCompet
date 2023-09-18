import torch
import torch.nn as nn


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
        self, obs_dim, action_dim, hidden_units=[64, 64], action_std_bound=(-20, 2)
    ):
        super(ContinuousActorCriticNetwork, self).__init__()

        self.action_std_bound = action_std_bound

        # Construct shared layers dynamically
        shared_layers = []
        input_dim = obs_dim
        for units in hidden_units:
            shared_layers.append(nn.Linear(input_dim, units))
            shared_layers.append(nn.ReLU())
            input_dim = units
        self.shared_layers = nn.Sequential(*shared_layers)

        # Actor layers
        self.actor_mu = nn.Linear(hidden_units[1], action_dim)
        self.actor_log_std = nn.Parameter(
            torch.zeros(action_dim)
        )  # Using nn.Parameter to make it learnable

        # Critic layers
        self.critic = nn.Linear(hidden_units[1], 1)

    def forward(self, obs: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        features = self.shared_layers(obs)
        mu = self.actor_mu(features)

        # Clamp log_std values and use softplus to ensure the std is positive
        log_std = torch.clamp(
            self.actor_log_std, self.action_std_bound[0], self.action_std_bound[1]
        )
        std = torch.nn.functional.softplus(log_std)

        value_estimate = self.critic(features)
        return mu, std, value_estimate
