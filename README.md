# Reinforcement Learning Lab

A hands-on learning and implementation repository covering **Reinforcement Learning from first principles to Deep Reinforcement Learning**. It brings together structured notes, algorithm explanations, implementation work, game-playing projects, and reproducible experiments.

The goal is to make this repository useful as a learning resource: a student or engineer should be able to understand what RL is, how an agent learns from rewards, how the main algorithms differ, and how those ideas are implemented in practical environments.

## Learning roadmap

| Section | Topics | Learning material and practical focus |
|---|---|---|
| **01 — Foundations** | Agent, environment, state, action, reward, return, policy, discount factor, MDP, Markov property, value functions, Bellman equations, exploration vs. exploitation | Core concepts, mathematical intuition, grid-world examples, and the agent–environment interaction loop. |
| **02 — Tabular RL** | Dynamic Programming, Monte Carlo, Temporal-Difference learning, SARSA, Q-Learning | Bellman backups, episode-based returns, TD error, on-policy vs. off-policy learning, Cliff Walking, and Gymnasium interaction. |
| **03 — Deep RL** | Neural Q-functions, DQN, replay buffer, target network, epsilon-greedy exploration, Double DQN | Function approximation, training workflow, target calculations, stability techniques, and evaluation. |
| **04 — Projects** | Flappy Bird DQN, Mario Double DQN | Applying value-based RL to game environments, understanding state/observation design, action selection, rewards, network architectures, and training pipelines. |
| **05 — Experiments** | Configurations, seeds, evaluation, reward curves, checkpoints | Recording implementation choices, comparing agent behaviour, measuring episode returns, evaluating learned policies, and documenting observations. |

## Learning notes

1. [RL Fundamentals](docs/RL_FUNDAMENTALS.md) — the agent–environment loop, reward and return, policy, MDP, Markov property, value functions, Bellman equations, and exploration/exploitation.
2. [Tabular Reinforcement Learning](docs/TABULAR_RL.md) — Dynamic Programming, Monte Carlo, TD learning, SARSA, Q-Learning, Cliff Walking, and Gymnasium.
3. [Deep Reinforcement Learning and DQN](docs/DEEP_RL_DQN.md) — neural Q-functions, DQN, replay memory, target networks, Double DQN, and game-agent workflows.
4. [Learning Glossary](docs/LEARNING_GLOSSARY.md) — concise definitions for quick revision.
5. [Experiment Guide](docs/EXPERIMENT_GUIDE.md) — a consistent protocol for recording and interpreting runs.

The explanations follow the supplied course PDFs and project assignment materials, with additional notation and clarifying examples to connect the concepts to implementation.

## Repository structure

```text
.
├── README.md
├── docs/
│   ├── RL_FUNDAMENTALS.md
│   ├── TABULAR_RL.md
│   ├── DEEP_RL_DQN.md
│   ├── LEARNING_GLOSSARY.md
│   ├── EXECUTION_STATUS.md
│   ├── EXPERIMENT_GUIDE.md
│   └── SETUP_GUIDE.md
├── projects/
│   ├── flappy_bird_dqn/
│   │   ├── README.md
│   │   ├── EXPERIMENT_RESULTS.md
│   │   ├── results/
│   │   └── assigned_implementation/
│   └── mario_double_dqn/
├── experiments/
│   └── EXPERIMENT_TEMPLATE.md
└── requirements/
```

## Projects

### Flappy Bird — Deep Q-Network (DQN)

The Flappy Bird project applies value-based reinforcement learning to a game-playing agent. The state-based network uses 12 input features, a 256-unit hidden layer, and two output action values (no-op and flap). The learning workflow uses experience replay, a target network, epsilon-greedy exploration, and reward-based updates.

**Execution update:** end-to-end training and a separate 100-episode greedy evaluation have now been completed locally. The evaluation achieved a mean reward of **1.455**, standard deviation **0.907**, maximum **5.6**, and minimum **-0.2**. The best training reward recorded was **5.6 at episode 13,604**.

- [Flappy Bird project guide](projects/flappy_bird_dqn/README.md)
- [Measured experiment results and limitations](projects/flappy_bird_dqn/EXPERIMENT_RESULTS.md)
- [Result artifact staging instructions](projects/flappy_bird_dqn/results/README.md)

The assigned implementation is the starting codebase for this project. Its inspected target calculation uses the maximum output from the target network, which is standard DQN rather than Double DQN. Preserve applicable original license and attribution notices when using or redistributing source files.

### Super Mario Bros — Double DQN

The Mario assignment applies Double DQN to a visual game environment. The reference workflow covers observation preprocessing, frame stacking, a convolutional Q-network, replay memory, epsilon-greedy action selection, and separate online/target networks.

[Read the Mario Double DQN project documentation](projects/mario_double_dqn/README.md).

The assignment points to the [PyTorch Mario RL tutorial](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html). Its original environment stack uses legacy Gym ecosystem packages, so dependency and operating-system compatibility should be checked before setup. Mario environment training and evaluation have not yet been completed.

## Experiments and reproducibility

Every experiment should record:
- Python and package versions.
- Operating system and device (CPU/GPU).
- Environment ID and wrappers.
- Hyperparameters and random seed.
- Number of training episodes/steps.
- Evaluation protocol and number of evaluation episodes.
- Mean and standard deviation of evaluation return.
- Checkpoint path and training-log path.
- Warnings, failures, and deviations from the implementation being followed.

Training results, reward curves, and checkpoints are added only after the corresponding run has actually completed. This keeps the repository technically honest and makes experiments reproducible.

See [Execution Notes](docs/EXECUTION_STATUS.md) and the [Setup Guide](docs/SETUP_GUIDE.md).

## Learning principle

The central idea of reinforcement learning is learning from interaction: **observe a state → choose an action → receive a reward → observe the next state → improve the decision-making strategy**. The sections in this repository build from that loop toward algorithms that learn value functions and neural-network-based game agents.
