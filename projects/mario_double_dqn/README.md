# Super Mario Bros — Double Deep Q-Network (Double DQN)

## Assignment and research basis

The supplied assignment asks for a Mario-playing reinforcement-learning agent using **Double Deep Q-Networks (Double DQN)** and points to the official PyTorch tutorial. The implementation in this folder follows that tutorial's core design: visual preprocessing, frame skipping and stacking, a convolutional Q-network, experience replay, epsilon-greedy action selection, and separate online and target networks.

The algorithm follows Hado van Hasselt, Arthur Guez, and David Silver, [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461), AAAI 2016, DOI [10.1609/AAAI.V30I1.10295](https://doi.org/10.1609/AAAI.V30I1.10295). Double DQN selects the next action with the online network and evaluates that action with the target network, reducing the overestimation associated with using one estimator for both operations.

Primary implementation documentation: [PyTorch — Train a Mario-playing RL Agent](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html).

## Implementation

The project contains a complete, modular training and evaluation implementation:

| File | Responsibility |
|---|---|
| `code/environment.py` | Super Mario Bros environment creation, two-action Joypad wrapper, frame skipping, grayscale conversion, resize to 84 × 84, and four-frame stacking |
| `code/mario_dqn.py` | Tutorial-style CNN, separate online/target Q-networks, disk-backed replay memory, epsilon-greedy agent, Double DQN target, optimization, and checkpoints |
| `code/train.py` | Training loop, reproducible seeds, episode metrics, best/latest checkpoints, and graceful save on interruption |
| `code/evaluate.py` | Separate greedy evaluation using a saved checkpoint, per-episode CSV, mean and standard deviation |
| `code/smoke_test.py` | Environment reset/step and observation-shape check |
| `code/test_core.py` | CPU unit tests for the CNN, frozen target network, learning update, and checkpoint round-trip |
| `requirements-mario.txt` | Isolated legacy Gym/NES dependency set |

### Observation and action design

The environment is reduced to grayscale 84 × 84 frames. Four consecutive frames are stacked, giving the network an input tensor of shape **(4, 84, 84)**. The action space follows the supplied tutorial:

- Action 0: walk right.
- Action 1: jump right.

The wrapper repeats each selected action for four frames and accumulates the reward. Replay observations are stored as uint8 in a disk-backed circular buffer to avoid keeping the entire image replay history in RAM.

### Double DQN update

For each sampled transition, the online network chooses the next action:

`a* = argmax_a Q_online(s', a; theta)`

The target network evaluates that action:

`y = r + gamma * (1 - terminated) * Q_target(s', a*; theta_target)`

The online network is optimized with Smooth L1 loss. Target-network parameters are frozen between periodic synchronization steps. Truncated time-limit transitions are recorded separately from true terminal transitions so the target does not incorrectly suppress bootstrapping at a time limit.

## Setup

Use a dedicated **Python 3.10** environment for the legacy Gym/NES emulator stack. Do not install these dependencies into the Flappy Bird environment.

From this folder:

```bash
python -m venv .venv
```

Activate the environment, install a PyTorch and TorchVision pair supported by the machine from the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install the project dependencies:

```bash
python -m pip install -r requirements-mario.txt
```

The NES emulator package may require native build tools on Windows if a compatible wheel is unavailable. The upstream PyTorch tutorial itself uses a legacy Gym stack and explicitly warns that Gym is unmaintained; this isolated environment is intentional.

## Run

Run commands from `projects/mario_double_dqn/`.

First run the environment smoke test:

```bash
python code/smoke_test.py
```

Run the CPU core tests:

```bash
python code/test_core.py
```

Start training (the tutorial recommends a long training horizon; 40,000 episodes is the long-run setting):

```bash
python code/train.py --episodes 40000 --seed 42
```

For a short integration check, limit the run:

```bash
python code/train.py --episodes 1 --max-steps 200 --seed 42
```

Evaluate a saved policy separately from exploratory training:

```bash
python code/evaluate.py --checkpoint checkpoints/mario/best_mario_double_dqn.pt --episodes 20
```

Add `--render` to the smoke test, training, or evaluation command to open the game window.

## Outputs and reproducibility

Training writes `checkpoints/mario/training_metrics.csv`, a best checkpoint, and a latest checkpoint. Replay observations are written to `replay_memory/`. Evaluation writes a separate CSV under `evaluation/` by default. These generated files are ignored by Git; do not commit emulator ROMs or game assets.

The training CSV records episode return, length, epsilon, mean loss, mean Q-value, flag completion, and elapsed time. The evaluation CSV records episode return, length, and whether the flag was reached. Keep training and evaluation data separate.

## Validation status

The CNN forward pass, Double DQN target calculation, frozen target-network gradients, and online-network backward pass were validated on a CPU PyTorch runtime during implementation. The committed `test_core.py` provides repeatable unit tests for these components and checkpoint handling.

**End-to-end emulator smoke testing, Mario training, and gameplay evaluation are environment-dependent and are not represented as completed results until those commands have run successfully on a machine with the NES/Gym dependencies installed.** No gameplay score, trained checkpoint, or success-rate claim is fabricated.

## Further reading

- [Official PyTorch Mario tutorial](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
- [Double DQN research paper (AAAI 2016)](https://arxiv.org/abs/1509.06461)
- [Deep RL and DQN notes](../../docs/DEEP_RL_DQN.md)
- [Experiment guide](../../docs/EXPERIMENT_GUIDE.md)
- [Repository execution status](../../docs/EXECUTION_STATUS.md)
