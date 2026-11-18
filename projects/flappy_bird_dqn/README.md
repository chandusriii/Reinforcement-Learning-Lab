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

Use Python 3.11. From the repository root, install the pinned environment dependencies and then install the matching PyTorch build separately for your machine:

```cmd
python -m pip install -r requirements/flappy-bird.txt
```

Choose the PyTorch CPU/CUDA build from the [official installer selector](https://docs.pytorch.org/get-started/locally/), then follow the [Windows setup guide](../../docs/SETUP_GUIDE.md). From `assigned_implementation/`, the recommended seeded runner trains for a finite budget (20,000 episodes by default; at most 10,000 steps per episode):

```cmd
python agent_eval.py flappybirdv0 --train --seed 42 --episodes 20000
```

It saves the best policy weights as `runs/flappybirdv0.pt`, the latest weights snapshot, a full resumable checkpoint (`runs/flappybirdv0_resume.pt`), and per-episode metrics (return, steps, environment steps, epsilon, loss, and clipped observation count). The full checkpoint contains policy/target networks, optimizer, replay memory, exploration state, RNG states, and completed episode. Resume with the same seed and environment settings:

```cmd
python agent_eval.py flappybirdv0 --train --seed 42 --episodes 20000 --resume
```

Here `--episodes` is the number of additional episodes. The runner checkpoints every 500 episodes by default and retains the most recent completed checkpoint. A normal run with no `--resume` starts a fresh experiment. For a reproducible, headless greedy evaluation:

```cmd
python agent_eval.py flappybirdv0 --seed 10042 --episodes 100
```

Use `--render` to watch an evaluation. Greedy evaluation writes per-episode reward and step counts to `runs/flappybirdv0_evaluation_seed_10042.csv`. The original coursework `agent.py` remains unchanged and is retained for source comparison; the bounded runner is recommended for new experiments.

To measure the reported Gymnasium observation-space warning rather than suppress it, run this diagnostic from the same folder:

```cmd
python diagnose_observations.py --episodes 10 --max-steps 2000 --seed 42
```

It measures the raw environment range before correction, counts the out-of-Box values, and verifies that every observation returned by the bounded adapter satisfies the declared Box. The third-party environment itself is not modified.

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

- The raw LiDAR observation-space mismatch has been reproduced in GitHub Actions. Across 150 sampled observations, 2,944 individual values exceeded the declared [0, 1] Box; the maximum observed value was 1.0021062783. The environment's ray-intersection output therefore exceeds its advertised normalized upper bound. The seeded `agent_eval.py` adapter disables the premature passive checker and clips observations to the declared bounds before model input. The diagnostic records the raw values before correction and verifies that all values passed through the adapter satisfy the Box. The upstream mismatch remains documented rather than hidden.
- The historical 13,604-episode training run and 100-episode evaluation were not recorded with a seed. The new runner supports seeded, resumable long experiments; its train/resume/evaluate pipeline is covered by CI, but this is not a new long-run performance result.
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
