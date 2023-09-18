# Playing Tennis using PPO agents

In this project, we try to tackle a multi-agent system by leveraging advanced techniques in deep reinforcement learning. We use an implementation of the Proximal Policy Optimization (PPO) algorithm tailored for Unity's Tennis environment (continuous action space and multi-agent).

**Notes:** This project shares a significant portion of code with my other [Continuous Control](https://github.com/naoufal51/ContinuousControlProject.git) project.

## Environment Details

### Overview

This environment presents two agents who control rackets to volley a tennis ball. A positive reward of +0.1 is received if the ball crosses the net, while a punishment of -0.01 is given for every miss, emphasizing the need for continued volleys.

#### Observation Space
- **Dimension**: 8 
- **Details**: The variables are related to the position coordinates and the velocity of both the ball and racket.

#### Action Space
- **Type**: Continuous
- **Dimension**: 2 
- **Actions**: The agent can move towards and away from the net, and jump.

#### Solving Criteria
The task is episodic. A moving average score of +0.5 over the last 100 episodes (considering the max score of the two agents in each episode) is necessary to consider the environment as solved.

## Setting Up

#### Prerequisites

1. **Python Environment**: Ensure Python (>=3.6) is installed. Due to package dependencies, it's advised to set up the `drlnd` kernel. Comprehensive setup guide available [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). Activation command:
    ```bash
    source activate drlnd
    ```

2. **Weights & Biases Integration**: For real-time performance tracking, analysis, and hyperparameter tuning, Weights & Biases is integrated. Registration might be required for first-time users.

#### Installation
1. Clone the repository to access the latest features and implementations:
    ```bash
    git clone https://github.com/naoufal51/CollabCompet.git
    ```

2. Ensure `Tennis.app` aligns with your OS specifications.
   
3. Transition to the project's root directory and install the pertinent dependencies:
    ```bash
    cd CollabCompet
    pip3 install -r requirements.txt
    ```

4. Synchronize Weights & Biases with your unique API key to facilitate seamless experiment tracking:
    ```bash
    export WANDB_API_KEY=<your_wandb_api_key>
    ```

**Dependencies**:
- `numpy`: For numerical operations and data manipulation.
- `torch`: Deep learning framework used for modeling and optimization.
- `unityagents`: Interface and utilities for Unity-based environments.
- `wandb`: Performance tracking and visualization tool.
- `matplotlib`: For creating static, animated, and interactive visualizations.

## Execution

### Training

Initiate the training sequence with:

```bash
python src/train.py
```

During training, performance metrics are systematically logged via `wandb` and curated under the `results` directory. Post-convergence or post the exhaustion of episodes, the model weights are archived in `results/weights`.

### Evaluation

Post-training, the model's efficacy can be gauged via:

```bash
python src/evaluate.py
```

The evaluation matrices, inclusive of a histogram detailing score distributions, are consolidated under `results`.

### Result Analysis

For an insightful drill-down into the performance metrics:

```bash
jupyter notebook results_visualisation.ipynb
```

## Repository Structure

```
.
├── README.md
├── Tennis.app
├── python
├── results
│   ├── scores*.npy
│   ├── scores*.png
│   └── weights
│       ├── actor_critic*.pth
├── results_visualisation.ipynb
├── src
│   ├── actor_critic_model.py
│   ├── evaluate.py
│   ├── ppo_agent.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── report.md

```


### Credits
This project draws inspiration from :
1. Udacity DRLND.
2. [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/)

### License
You can freely use the code present in the repo.

For modules provided by Udacity DRLND check their repo [DRLND](https://github.com/udacity/deep-reinforcement-learning#dependencies).
