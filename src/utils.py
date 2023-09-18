import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple
import random


def generate_batches(
    batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    log_probs: torch.Tensor,
) -> Tuple:
    """
    Generate batches of experience to tain our actor critic model.

    Args:
        batch_size (int): The size of the batch
        states (torch.Tensor): The current state of the environment.
        actions (torch.Tensor): The actions takes by the agents.
        rewards (torch.Tensor): The rewards received by the agents after interacting with the environment.
        next_states (torch.Tensor): The next states of the environment after the agents' interaction.
        dones (torch.Tensor): To indicate if the episode is finished.
        log_probs (torch.Tensor): Log probabilities of the actions to compute the ratios for the PPO algorithm.

    Yields:
        (tuple): A tuple that contains the batch of experiences:
            states the states, actions, rewards, next_states, dones, log_probs, start, end.
            start (int): Start index of the batch.
            end (int): End index of the batch.

    """
    num_samples = states.shape[0]
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield (
            states[start:end],
            actions[start:end],
            rewards[start:end],
            next_states[start:end],
            dones[start:end],
            log_probs[start:end],
            start,
            end,
        )


def set_seeds(seed: int = 42) -> None:
    """Helper function to set seeds in relevant sections to ensure reproducibility.
    We set seed for torch, numpy and python's random.

    Args:
        seed (int): The seed to use.

    Returns:
        None.

    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def plot_scores(scores, save_path="results/scores_plot.png"):
    """Plot scores and save the figure.
    Args:
        scores (list): The scores to plot.
        save_path (str): The path to save the figure.

    Returns:
        None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.title("Training progress of PPO on Reacher environment")
    plt.savefig(save_path)
    plt.close()


def save_scores(scores, file_name="results/scores.npy"):
    """Save scores to a npy file for later use.
    Args:
        scores (list): The scores to save.
        file_name (str): The path to save the scores.

    Returns:
        None.
    """
    print(f"Saving scores to {file_name}")
    np.save(file_name, scores)
