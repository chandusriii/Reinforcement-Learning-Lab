# Reinforcement Learning Lab

A structured, experiment-driven repository for learning and implementing Reinforcement Learning (RL), from tabular methods to Deep Q-Networks (DQN) and Double DQN.

> **Experiment integrity:** Results are recorded only after an experiment has actually run. The current repository documents the learning path and execution setup. Flappy Bird and Mario game-training runs are **pending**; no game-training reward, checkpoint, or learning curve is claimed yet.

## Learning roadmap

| Stage | Topics / implementation | Status |
|---|---|---|
| 01 — Foundations | Agent, environment, state, action, reward, policy, return, discount factor, MDP, Bellman equations, exploration vs. exploitation | Notes to be organized |
| 02 — Tabular RL | Dynamic Programming, Monte Carlo, Temporal-Difference learning, SARSA, Q-Learning | Notebook source reviewed; reproducible reruns pending |
| 03 — Deep RL | Neural Q-functions, DQN, replay buffer, target network, epsilon-greedy exploration | Architecture reviewed; environment runs pending |
| 04 — Projects | Flappy Bird DQN, Mario Double DQN | Not yet trained end-to-end |
| 05 — Experiments | Configurations, seeds, evaluation, reward curves, checkpoints | To be populated with measured runs |

## Repository structure

```text
.
├── README.md
├── docs/
│   ├── EXECUTION_STATUS.md
│   ├── SETUP_GUIDE.md
│   └── RL_FUNDAMENTALS.md
├── notebooks/
│   ├── tabular/
│   └── deep_rl/
├── projects/
│   ├── flappy_bird_dqn/
│   └── mario_double_dqn/
├── experiments/
│   ├── results/
│   └── logs/
└── requirements/
```

Folders for notebooks, projects, and experiments will be populated as implementations are added and validated.

## Projects and references

### Flappy Bird — DQN
The reviewed reference implementation is [shradha-khapra/RL_DQN](https://github.com/shradha-khapra/RL_DQN). Its network consumes a 12-feature state vector, uses a 256-unit hidden layer, and outputs two action values. The reviewed training design includes a replay buffer, a target network, and epsilon-greedy action selection.

**Important algorithm note:** the inspected target calculation uses the maximum action value from the target network. That is standard DQN target selection/evaluation, not the Double DQN action-selection/evaluation split.

The Gymnasium environment package is documented at [flappy-bird-gymnasium on PyPI](https://pypi.org/project/flappy-bird-gymnasium/) and in the [Gymnasium third-party environment catalogue](https://gymnasium.farama.org/v1.2.3/environments/third_party_environments/).

### Super Mario Bros — Double DQN
The learning reference is the [official PyTorch Mario RL tutorial](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html). It demonstrates a Double Deep Q-Network workflow with frame preprocessing, stacked observations, replay memory, and target-network learning.

The tutorial's original dependency stack includes legacy Gym-based packages. Treat its environment setup as a compatibility-sensitive reference, not as a modern, guaranteed plug-and-play installation.

## Reproducibility policy

For every experiment, record:
- Python and package versions
- Operating system and device (CPU/GPU)
- Environment ID and wrappers
- Hyperparameters and random seed
- Number of training episodes/steps
- Evaluation protocol and number of evaluation episodes
- Mean and standard deviation of evaluation return
- Checkpoint path and training-log path
- Known warnings, failures, and deviations from the reference

Do not commit large generated checkpoints or raw game recordings unless needed; use Git LFS or release assets when appropriate. Preserve upstream attribution and license notices when adapting code.

## Current execution summary

See [docs/EXECUTION_STATUS.md](docs/EXECUTION_STATUS.md) for what has and has not been executed.