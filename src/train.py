import numpy as np
from unityagents import UnityEnvironment
from collections import deque
import wandb
from utils import set_seeds, plot_scores, save_scores
from ppo_agent import PPOContinuous
import torch


def train(
    env: UnityEnvironment,
    brain_name: str,
    ppo: PPOContinuous,
    n_episodes: int = 2000,
    batch_size: int = 128,
):
    """
    Main training loop.
    Args:
        env (UnityEnvironment): the UnityEnvironment instance.
        brain_name (str): the brain name of the environment.
        ppo (PPOContinuous): the PPOContinuous instance.
        n_episodes (int): number of episodes to train the ppo agent for.
        batch_size (int): the batch size to use when updating the ppo agent.

    Returns:
        None
    """
    scores_window = deque(maxlen=100)
    scores = []
    timesteps = 0
    max_score = 0.5
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(states.shape[0])
        while True:
            actions, log_probs = ppo.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score += rewards

            ppo.memorize(states, actions, rewards, next_states, dones, log_probs)
            states = next_states
            timesteps += 1

            if any(dones):
                break
        ppo.update(batch_size=batch_size)
        mean_score = np.mean(score)
        wandb.log({"Episode Mean Score": mean_score})
        scores.append(mean_score)
        print(f"Episode {i_episode} Score: {mean_score}")

        scores_window.append(mean_score)
        mean_score_window = np.mean(scores_window)
        wandb.log({"Episode Mean Score Window": mean_score_window})

        if i_episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_window)
                )
            )

        if np.mean(scores_window) >= max_score:
            print(
                f"The environment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window)}"
            )
            torch.save(
                ppo.actor_critic.state_dict(),
                "results/weights/actor_critic_512_entropy_coef_re.pth",
            )
            max_score = np.mean(scores_window)

    return scores


if __name__ == "__main__":
    wandb.init(project="ppo_competition")
    set_seeds(42)
    env = UnityEnvironment(file_name="./Tennis.app", seed=42, worker_id=0)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    ppo = PPOContinuous(
        state_size=state_size,
        action_size=action_size,
        logger=wandb,
        entropy_coef=0.005,
        hidden_units=[128, 64],
    )

    scores = train(
        env,
        brain_name,
        ppo,
        n_episodes=50000,
        batch_size=512,
    )

    plot_scores(scores, "results/scores_512_entropy_coef_0.005_re.png")
    save_scores(scores, "results/scores_512_entropy_coef_0.005_re.npy")

    env.close()
    wandb.finish()
