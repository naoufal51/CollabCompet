import numpy as np
from unityagents import UnityEnvironment
from utils import set_seeds
from ppo_agent import PPOContinuous
import torch
import matplotlib.pyplot as plt
from utils import save_scores


def evaluate(env, brain_name, ppo, n_episodes=10, train_mode=False):
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(states.shape[0])
        while True:
            actions, _ = ppo.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score += rewards

            states = next_states

            if any(dones):
                break

        mean_score = np.mean(score)
        scores.append(mean_score)
        print(f"Evaluation Episode {i_episode} Score: {mean_score}")

    return scores


if __name__ == "__main__":
    set_seeds(42)
    env = UnityEnvironment(file_name="./Tennis.app", seed=42, worker_id=1)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]

    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    ppo = PPOContinuous(state_size=state_size, action_size=action_size)
    ppo = PPOContinuous(
        state_size=state_size,
        action_size=action_size,
        hidden_units=[128, 64],
    )
    ppo.actor_critic.load_state_dict(
        torch.load("results/weights/actor_critic_512_entropy_coef_re.pth")
    )
    scores = evaluate(env, brain_name, ppo, n_episodes=1000, train_mode=True)

    # save scores
    save_scores(scores, "results/scores_eval_1000epsbis.npy")

    env.close()
    print(f"Mean Evaluation Score over {len(scores)} episodes: {np.mean(scores)}")

    # Plot the histogram of scores
    plt.hist(scores, bins=20, edgecolor="black")
    plt.title("Distribution of Scores over Episodes")
    plt.xlabel("Score")
    plt.ylabel("Number of Episodes")
    # save plot
    plt.savefig("results/dist_scores_eval_1000epsbis.png")
    plt.close()
