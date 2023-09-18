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

        self.initialize_parameters(
            n_epochs,
            gamma,
            clip_epsilon,
            action_std_bound,
            normalize_advantage,
            tau,
            logger,
            value_coef,
            entropy_coef,
            max_norm,
        )

        self.clear_memory()

    def initialize_parameters(
        self,
        n_epochs,
        gamma,
        clip_epsilon,
        action_std_bound,
        normalize_advantage,
        tau,
        logger,
        value_coef,
        entropy_coef,
        max_norm,
    ):
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.action_std_bound = action_std_bound
        self.normalize_advantage = normalize_advantage
        self.tau = tau
        self.logger = logger
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_norm = max_norm

    def clear_memory(self):
        """
        Clear the memory after training using the generated data from the currect policy.
        Given that PPO is an on-policy algorithm, we cannot reuse data of older policies.
        Therefore, we clear the memory after each training.
        """
        (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones,
            self.log_probs,
        ) = ([] for _ in range(6))

    def act(self, states: List) -> Tuple[np.ndarray, np.ndarray]:
        """
         Compute the actions and the log probabilities of the given states.

        Args:
            states (List): List of states.

        Returns:
            actions (np.ndarray): numpy array containing the actions.
            log_probs (np.ndarray): numpy array containing the log probabilities.
        """
        actions, log_probs = [], []

        for state in states:
            action, log_prob = self.compute_action_log_prob(torch.tensor(state).float())
            actions.append(action.numpy())
            log_probs.append(log_prob.numpy())

        return np.array(actions), np.array(log_probs)

    def compute_action_log_prob(
        self,
        state: torch.Tensor,
        use_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the actions and the log probabilities of the given states.

        Args:
            state (torch.Tensor): Tensor of states.
            use_grad (bool, optional): Whether to compute with gradients. Defaults to False.

        Returns:
            actions (torch.Tensor): Tensor of actions.
            log_probs (torch.Tensor): Tensor of log probabilities.

        """
        if use_grad:
            mu, log_std, _ = self.actor_critic(state)
        else:
            with torch.no_grad():
                mu, log_std, _ = self.actor_critic(state)

        action_dist = torch.distributions.Normal(mu, log_std.exp())
        sampled_action = action_dist.sample()
        action_log_prob = action_dist.log_prob(sampled_action).sum(dim=-1)

        return sampled_action, action_log_prob

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
        self.states.extend(states)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.next_states.extend(next_states)
        self.dones.extend(dones)
        self.log_probs.extend(log_probs)

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

    def update(self, batch_size: int = 64) -> None:
        """
        Update the policy and value function using the collected data.

        Args:
            batch_size (int): Number of data points to use per update.

        Returns:
            None
        """

        states, actions, rewards, next_states, dones, old_log_probs = map(
            lambda x: torch.tensor(np.vstack(x)).float(),
            [
                self.states,
                self.actions,
                self.rewards,
                self.next_states,
                self.dones,
                self.log_probs,
            ],
        )

        self.train_model(
            states, actions, rewards, next_states, dones, old_log_probs, batch_size
        )
        self.clear_memory()

    def train_model(
        self, states, actions, rewards, next_states, dones, old_log_probs, batch_size
    ):
        """
        Train the policy and value function using the collected data.

        Args:
            states (torch.Tensor): Tensor of the current states.
            actions (torch.Tensor): Tensor of actions taken by the actors.
            rewards (torch.Tensor): Tensor of rewards given by the environment to the actors.
            next_states (torch.Tensor): Tensor of the next states.
            dones (torch.Tensor): Tensor of dones, indicating the end of episodes.
            old_log_probs (torch.Tensor): Tensor of old log probabilities, which are used to compute the ratio for PPO.
            batch_size (int): Number of data points to use per update.

        Returns:
            None
        """
        # Metrics to store
        entropy_losses, policy_gradient_losses, value_losses = [], [], []
        approx_kls, explained_variances, clip_fractions, stds = [], [], [], []

        # Compute Actor-Critic current and next states values using the actor-critic model
        with torch.no_grad():
            _, _, values = self.actor_critic(states)
            _, _, next_values = self.actor_critic(next_states)

        # Compute GAE (Generalized Advantage Estimation)
        advantages = self.compute_gae(
            rewards.squeeze(-1),
            values.squeeze(),
            next_values.squeeze(),
            dones.squeeze(-1),
        )

        # Compute TD targets for Value Estimator
        td_targets = advantages + values.squeeze()

        # Normalize advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Train Actor-Critic model
        for _ in range(self.n_epochs):
            for batch_data in generate_batches(
                batch_size, states, actions, rewards, next_states, dones, old_log_probs
            ):
                self.train_batch(
                    advantages,
                    td_targets,
                    batch_data,
                    entropy_losses,
                    policy_gradient_losses,
                    value_losses,
                    approx_kls,
                    explained_variances,
                    clip_fractions,
                    stds,
                )

        # Log metrics using WandB
        if self.logger:
            self.log_metrics(
                entropy_losses,
                policy_gradient_losses,
                value_losses,
                approx_kls,
                explained_variances,
                clip_fractions,
                stds,
            )

    def train_batch(
        self,
        advantages: torch.Tensor,
        td_targets: torch.Tensor,
        batch_data: tuple,
        entropy_losses: list,
        policy_gradient_losses: list,
        value_losses: list,
        approx_kls: list,
        explained_variances: list,
        clip_fractions: list,
        stds: list,
    ):
        """
         Train the policy and value function using a batch of data.

        Args:
            advantages (torch.Tensor): Tensor of advantages computed using GAE.
            td_targets (torch.Tensor): Tensor of TD targets used for value loss computation.
            batch_data (tuple): Tuple of data to train the model.
            entropy_losses (list): List of entropy losses used for metric tracking.
            policy_gradient_losses (list): List of policy gradient losses used for metric tracking.
            value_losses (list): List of value losses used for metric tracking.
            approx_kls (list): List of approximated KL divergences used for metric tracking.
            explained_variances (list): List of explained variances used for metric tracking.
            clip_fractions (list): List of clip fractions used for metric tracking.
            stds (list): List of standard deviations used for metric tracking.

        Returns:
            None

        """
        (
            batch_states,
            batch_actions,
            _,
            _,
            _,
            batch_old_log_probs,
            start,
            end,
        ) = batch_data

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

        # compute surrogate loss for PPO (Small policy update)
        batch_advantages = advantages[start:end]
        surrogate_loss = -torch.min(
            ratios * batch_advantages,
            torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * batch_advantages,
        ).mean()

        # Compute value loss for Value Estimator
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

        # Compute approximate KL divergence to track the difference between old and new policy
        approx_kl = self.compute_kl_divergence(action_log_probs, batch_old_log_probs)

        # Compute clip fraction
        clip_fraction = self.compute_clip_fraction(ratios)

        # Compute the explained variance  for value estimator
        explained_variance = self.compute_explained_variance(
            batch_td_targets, value_estimate
        )

        # Store metrics
        entropy_losses.append(entropy_loss.item())
        policy_gradient_losses.append(surrogate_loss.item())
        value_losses.append(value_loss.item())
        approx_kls.append(approx_kl)
        explained_variances.append(explained_variance)
        clip_fractions.append(clip_fraction)
        stds.append(log_std.exp().mean().item())

    def compute_kl_divergence(
        self, current_log_probs: torch.Tensor, old_log_probs: torch.Tensor
    ) -> float:
        """
        Compute approximate KL divergence to track the difference between old and new policy.

        Kullback–Leibler divergence is a type of statistical distance that measures how one probability distribution is different from another.
        It is used here to control how much the old policy is distant from the new policy.

        Args:
            current_log_probs (torch.Tensor): Tensor of current policy log probabilities.
            old_log_probs (torch.Tensor): Tensor of old policy log probabilities.

        Returns:
            approx_kl (float): Approximate KL divergence.

        """
        log_ratio = current_log_probs - old_log_probs
        approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
        return approx_kl

    def compute_clip_fraction(self, ratios: torch.Tensor) -> float:
        """
        Compute the fraction of times that the ratio between the new and old policy
        is greater than 1 + epsilon or less than 1 - epsilon.

        Args:
            ratios (torch.Tensor): Tensor of ratios between new and old policy.

        Returns:
            float: Clip fraction.

        """
        return (
            ((ratios < 1 - self.clip_epsilon) | (ratios > 1 + self.clip_epsilon))
            .float()
            .mean()
            .item()
        )

    def compute_explained_variance(
        self, batch_td_targets: torch.Tensor, value_estimate: torch.Tensor
    ) -> float:
        """
        Compute the explained variance between TD targets and value estimates.

        It is a measure of how well the value estimates capture the
        variance of the true TD targets.

        Args:
            batch_td_targets (torch.Tensor): True TD targets.
            value_estimate (torch.Tensor): Estimated values.

        Returns:
            explained_variance (float): Explained variance, ranging from -inf to 1.
        """
        y_true = batch_td_targets.detach().numpy()
        explained_variance = 1 - np.var(
            y_true - value_estimate.detach().squeeze().numpy()
        ) / np.var(y_true)
        return explained_variance

    def log_metrics(
        self,
        entropy_losses: List,
        policy_gradient_losses: List,
        value_losses: List,
        approx_kls: List,
        explained_variances: List,
        clip_fractions: List,
        stds: List,
    ) -> None:
        """
        Log metrics to Wandb.

        Args:
            entropy_losses (list): List of entropy losses.
            policy_gradient_losses (list): List of policy gradient losses.
            value_losses (list): List of value losses.
            approx_kls (list): List of approximate KL divergences.
            explained_variances (list): List of explained variances.
            clip_fractions (list): List of clip fractions.
            stds (list): List of standard deviations.

        Returns:
            None.

        """
        self.logger.log({"mean_entropy_loss": np.mean(entropy_losses)})
        self.logger.log({"mean_policy_gradient_loss": np.mean(policy_gradient_losses)})
        self.logger.log({"mean_value_loss": np.mean(value_losses)})
        self.logger.log({"mean_approx_kl": np.mean(approx_kls)})
        self.logger.log({"mean_explained_variance": np.mean(explained_variances)})
        self.logger.log({"mean_clip_fraction": np.mean(clip_fractions)})
        self.logger.log({"mean_std": np.mean(stds)})
