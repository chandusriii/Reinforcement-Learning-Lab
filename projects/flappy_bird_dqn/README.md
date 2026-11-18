# Flappy Bird AI Agent Using Deep Q-Networks (DQN)

## Project Overview

This project trains a reinforcement learning agent to play Flappy Bird through interaction with the game environment. The agent observes a state, selects an action, receives a reward or penalty, and updates its action-value estimates.

The coursework implementation uses a standard Deep Q-Network (DQN). Its target calculation takes the maximum action value from the target network; this is standard DQN, not Double DQN.

## Model Architecture

The state-based neural network is designed with:

- **Input layer:** 180 LiDAR observations in the executed configuration (use_lidar=True). The network input dimension is read dynamically from the environment observation space.
- **Hidden layer:** 256 neurons with ReLU activation.
- **Output layer:** 2 Q-values, corresponding to the available actions (no-op and flap).

## Key Reinforcement Learning Components

- **Experience replay:** Stores previous transitions and samples batches for training.
- **Target network:** Provides more stable target-value estimates during learning.
- **Epsilon-greedy policy:** Balances exploration of new actions with exploitation of learned action values.
- **Reward-based learning:** Uses feedback from the environment to update the agent's action-value estimates.

## Technology Stack

- Python 3.11.9
- PyTorch 2.15.0+cpu
- Gymnasium 1.4.0
- flappy-bird-gymnasium 0.4.0
- NumPy 1.26.4
- Pygame 2.6.1
- Matplotlib 3.11.2
- PyYAML 6.0.3

The recorded run used CPU execution; CUDA was unavailable. These are the versions reported by the local environment and should be verified against a complete `pip freeze` export for exact reproduction.

## Project Workflow

1. Initialize the Flappy Bird environment.
2. Observe the 180-element LiDAR state representation used in this experiment.
3. Select an action using an epsilon-greedy policy during training.
4. Execute the action and collect the reward and next state.
5. Store the transition in the replay buffer.
6. Sample experiences and update the online Q-network.
7. Synchronize the target network at configured intervals.
8. Save the policy when a new best episode reward is observed.
9. Load the saved checkpoint and evaluate with exploration disabled.

## Assigned Source Implementation

The coursework source files are preserved in [`assigned_implementation/`](assigned_implementation/README.md). The original `agent.py`, `dqn.py`, `experience_replay.py`, `game_flappy_bird.py`, and `parameters.yaml` are kept separate from evaluation adaptations and experiment outputs.

The original `agent.py` was not modified to fix the evaluation print issue. A separate `agent_eval.py` copy was used for the 100-episode evaluation.

## Running the Project

Follow the [Windows setup guide](../../docs/SETUP_GUIDE.md) to create a compatible environment. From `assigned_implementation/`, train with:

```cmd
python agent.py flappybirdv0 --train
```

Run the saved agent with rendering using the original source:

```cmd
python agent.py flappybirdv0
```

For the separate 100-episode evaluation copy:

```cmd
python agent_eval.py flappybirdv0
```

Manual keyboard play:

```cmd
python game_flappy_bird.py
```

The source creates a `runs/` directory and saves the best checkpoint and best-reward log there. The published evaluation CSV is available in [`results/flappybirdv0_evaluation.csv`](results/flappybirdv0_evaluation.csv).

## Experiment Results

End-to-end training and a separate 100-episode greedy evaluation have been completed locally.

| Metric | Result |
|---|---:|
| Training episodes reached | At least 13,604 |
| Best training reward | **5.6** |
| Episode of best training reward | **13,604** |
| Evaluation episodes | **100** |
| Mean evaluation reward | **1.455** |
| Population standard deviation | **0.907** |
| Maximum evaluation reward | **5.6** |
| Minimum evaluation reward | **-0.2** |

The agent achieved a positive mean reward across the 100 evaluation episodes. Individual episode returns varied, so the best episode should not be interpreted as typical performance. The evaluation used the saved checkpoint without epsilon-greedy exploration.

See [Experiment Results](EXPERIMENT_RESULTS.md) for the protocol, interpretation, and limitations. The [results directory](results/README.md) contains the published evaluation CSV, training log, and both PNG visualizations.

## Warnings and Limitations

- Gymnasium emitted warnings that observations returned by environment reset/step were outside the declared observation space. The run and evaluation completed, but the mismatch has not yet been diagnosed.
- No random seed was recorded, so this is a single-run result rather than a multi-seed performance estimate.
- The source training log records best-reward improvements, not every episode reward or loss.
- The evaluation metrics describe this checkpoint and environment setup; they do not establish performance across other seeds or package versions.

## Learning Outcomes

- Understanding how DQN applies neural networks to action-value estimation.
- Working with state, action, reward, and next-state transitions.
- Understanding experience replay and target networks.
- Applying exploration-versus-exploitation strategies.
- Separating training from greedy policy evaluation.
- Reporting mean, standard deviation, maximum, and minimum episode returns.

## Attribution

This project uses the implementation assigned for the coursework as its starting codebase. Original authorship and applicable license notices should be retained when redistributing or modifying upstream source files.
