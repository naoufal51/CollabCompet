import torch
import numpy as np
from torch import optim
from torch.nn.utils import clip_grad_norm_
from actor_critic_model import ContinuousActorCriticNetwork
from utils import generate_batches
import torch.nn.functional as F
from typing import Tuple, List


class PPOContinuous:
    """
    PPO agent.
    It is trained using the PPO (Proximal Policy Optimization) algorithm.
    PPO is part of the policy gradient family, which means it directly optmizes the policy by estimating the gradient of the reward with respect to the policy parameters.
    It uses clipped surrogate loss to train the policy. Which ensures that new policy is not so far away from the old policy.
    It is known for its efficiency and stability.
    Given our Environment type, we choose to support continuous action space.
    """

    def __init__(
        self,
        state_size,
        action_size,
        n_epochs=100,
        gamma=0.99,
        clip_epsilon=0.2,
        lr=1e-4,
        action_std_bound=[1e-2, 1.0],
        normalize_advantage=True,
        tau=0.95,
        logger=None,
        value_coef=0.5,
        entropy_coef=0.01,
        max_norm=0.5,
        hidden_units=[64, 64],
    ):
        """
        Initialize the PPO agent for continuous actions.

        Args:
            state_size (int):  The state space dimension.
            action_size (int): The action space dimension.
            n_epochs (int): The number of epochs to train the models.
            gamma (float): Discount factor.
            clip_epsilon (float): Clip parameter for the PPO objective.
            lr (float): Learning rate.
            action_std_bound (list): Lower and upper bound for the standard deviation of the action.
            normalize_advantage (bool): Whether to normalize the advantage.
            tau (float): Parameter for soft update of the target network.
            logger (Logger): Logger for logging the training process.
            value_coef (float): Coefficient for the value loss.
            entropy_coef (float): Coefficient for the entropy loss.
            max_norm (float): Maximum gradient norm.
            hidden_units (list): List of hidden units for the actor and critic networks.

        """
        self.actor_critic = ContinuousActorCriticNetwork(
            state_size, action_size, hidden_units=hidden_units
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.tau = tau
        self.logger = logger
        self.clip_epsilon = clip_epsilon
        self.action_std_bound = action_std_bound
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_norm = max_norm

        self.normalize_advantage = normalize_advantage
        (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones,
            self.log_probs,
        ) = ([], [], [], [], [], [])

    def act(self, states: List) -> Tuple[np.ndarray, np.ndarray]:
        """
         Compute the actions and the log probabilities of the given states.

        Args:
            states (List): List of states.

        Returns:
            actions (np.ndarray): numpy array containing the actions.
            log_probs (np.ndarray): numpy array containing the log probabilities.
        """
        actions = []
        log_probs = []
        for state in states:
            state = torch.tensor(state).float()
            with torch.no_grad():
                mu, log_std, _ = self.actor_critic(state)
                action_dist = torch.distributions.Normal(mu, log_std.exp())
                sampled_action = action_dist.sample().detach()
                action_log_prob = action_dist.log_prob(sampled_action).sum(dim=-1)
                actions.append(sampled_action.numpy())
                log_probs.append(action_log_prob.numpy())

        return np.array(actions), np.array(log_probs)

    def memorize(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: List,
        next_states: np.ndarray,
        dones: List,
        log_probs: np.ndarray,
    ) -> None:
        """
        Memorize the given trajectories/experiences.

        Args:
            states (np.ndarray): numpy array containing states.
            actions (np.ndarray): numpy array containing actions.
            rewards (List): List of rewards.
            next_states (np.ndarray): numpy array containing next states.
            dones (List): List of terminal states.
            log_probs (np.ndarray): numpy array containing the log probabilities.
        """
        for i in range(states.shape[0]):
            self.states.append(states[i])
            self.actions.append(actions[i])
            self.rewards.append(rewards[i])
            self.next_states.append(next_states[i])
            self.dones.append(dones[i])
            self.log_probs.append(log_probs[i])

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Generalized Advantage Estimation (GAE) for policy gradient optimization.

        GAE is a technique that stabilizes the advantage function by averaging over
        multiple n-step advantage estimates. The goal is to reduce variance in the
        advantage estimate without introducing much bias.

        The GAE for time step 't' is formulated as:

        A^{GAE(γ, τ)}_t = Σ_{l=0}^{∞} (γτ)^l δ_{t+l}
        with δ_t = r_t + γV(s_{t+1}) - V(s_t)

        where:
        - δ_t represents the Temporal Difference (TD) error.
        - r_t is the reward at time t.
        - V(s) is the value function of state s.
        - γ is the discount factor for future rewards.
        - τ is a hyperparameter for GAE, weighting different n-step estimators.

        Args:
            rewards (torch.Tensor): Tensor of rewards for the trajectories.
            values (torch.Tensor): Tensor of value estimates for each state in trajectories.
            next_values (torch.Tensor): Tensor of value estimates for each next state in trajectories.
            dones (torch.Tensor): Tensor indicating if the state is terminal (1 if terminal, 0 otherwise).

        Returns:
            advantages Tensor: GAE advantages for each state.
        """

        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(deltas)
        advantage = torch.tensor(0.0)

        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + self.gamma * self.tau * advantage
            advantages[t] = advantage

        return advantages

    def update(self, batch_size=64):
        """
        Update the policy and value function using the collected data.

        Args:
            batch_size (int): Number of data points to use per update.

        Returns:
            None
        """
        states = torch.tensor(np.vstack(self.states)).float()
        actions = torch.tensor(np.vstack(self.actions)).float()
        rewards = torch.tensor(np.vstack(self.rewards)).float().squeeze(-1)
        next_states = torch.tensor(np.vstack(self.next_states)).float()
        dones = torch.tensor(np.vstack(self.dones)).float().squeeze(-1)
        old_log_probs = torch.tensor(np.vstack(self.log_probs)).float().squeeze(-1)

        entropy_losses = []
        policy_gradient_losses = []
        value_losses = []
        approx_kls = []
        explained_variances = []
        clip_fractions = []
        stds = []

        with torch.no_grad():
            _, _, values = self.actor_critic(states)
            _, _, next_values = self.actor_critic(next_states)

        advantages = self.compute_gae(
            rewards, values.squeeze(), next_values.squeeze(), dones
        )
        td_targets = advantages + values.squeeze()

        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            for (
                batch_states,
                batch_actions,
                _,
                _,
                _,
                batch_old_log_probs,
                start,
                end,
            ) in generate_batches(
                batch_size, states, actions, rewards, next_states, dones, old_log_probs
            ):
                # compute action log probabilities and entropy loss
                mu, log_std, value_estimate = self.actor_critic(batch_states)
                log_std = torch.clamp(
                    log_std, self.action_std_bound[0], self.action_std_bound[1]
                )
                action_dist = torch.distributions.Normal(mu, log_std.exp())
                action_log_probs = action_dist.log_prob(batch_actions).sum(dim=-1)
                entropy_loss = action_dist.entropy().mean()

                # compute ratio between new and old action log probabilities (new and old policy)
                ratios = (action_log_probs - batch_old_log_probs).exp()

                # compute surrogate loss for PPO
                batch_advantages = advantages[start:end]
                surrogate_loss = -torch.min(
                    ratios * batch_advantages,
                    torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages,
                ).mean()

                batch_td_targets = td_targets[start:end]

                value_loss = F.mse_loss(value_estimate.squeeze(), batch_td_targets)

                # Compute gradient and perform a single step of gradient descent on the actor-critic model
                # We clip the norm of the gradients to prevent gradient explosion
                self.optimizer.zero_grad()
                (
                    surrogate_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                ).backward()
                clip_grad_norm_(self.actor_critic.parameters(), max_norm=self.max_norm)
                self.optimizer.step()

                # Compute approximate KL divergence
                log_ratio = action_log_probs - batch_old_log_probs
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

                # Compute clip fraction
                clip_fraction = (
                    (
                        (ratios < 1 - self.clip_epsilon)
                        | (ratios > 1 + self.clip_epsilon)
                    )
                    .float()
                    .mean()
                    .item()
                )

                # Explained variance
                y_true = batch_td_targets.numpy()
                y_pred = value_estimate.detach().squeeze().numpy()
                mask = ~np.isnan(y_true)
                y_true = y_true[mask]
                y_pred = y_pred[mask]
                explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)

                # Store metrics
                entropy_losses.append(entropy_loss.item())
                policy_gradient_losses.append(surrogate_loss.item())
                value_losses.append(value_loss.item())
                approx_kls.append(approx_kl)
                explained_variances.append(explained_var)
                clip_fractions.append(clip_fraction)
                if hasattr(self.actor_critic, "actor_log_std"):
                    stds.append(torch.exp(log_std).mean().item())

        if self.logger:
            self.logger.log({"mean_entropy_loss": np.mean(entropy_losses)})
            self.logger.log(
                {"mean_policy_gradient_loss": np.mean(policy_gradient_losses)}
            )
            self.logger.log({"mean_value_loss": np.mean(value_losses)})
            self.logger.log({"mean_approx_kl": np.mean(approx_kls)})
            self.logger.log({"mean_explained_variance": np.mean(explained_variances)})
            self.logger.log({"mean_clip_fraction": np.mean(clip_fractions)})
            if stds:
                self.logger.log({"mean_std": np.mean(stds)})

        (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones,
            self.log_probs,
        ) = ([], [], [], [], [], [])
