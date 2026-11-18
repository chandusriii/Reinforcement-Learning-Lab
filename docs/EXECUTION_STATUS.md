# Execution Status

Last updated: 2026-11-18

This page separates code/architecture checks from actual environment training.

## Summary

| Work item | Status | Evidence / limitation |
|---|---|---|
| RL fundamentals material | Reviewed | Source notes were reviewed in the learning session; content organization is pending |
| SARSA / CliffWalking | Preliminary algorithm run reported | A recreated CliffWalking environment was used because Gymnasium was unavailable in that execution environment; this is not a rerun of the uploaded notebook |
| Q-Learning / CliffWalking | Preliminary algorithm run reported | Same recreated-environment limitation |
| Flappy Bird DQN network | Preliminary shape/finite-output check reported | Does not validate environment integration, optimization, or gameplay |
| Mario Double DQN network | Preliminary shape/finite-output check reported | Does not validate environment integration, optimization, or gameplay |
| Flappy Bird game training | **Pending** | Environment dependencies could not be installed in the isolated execution environment |
| Mario game training | **Pending** | Emulator/environment dependencies could not be installed in the isolated execution environment |
| Model checkpoints / reward curves | **Not available** | No end-to-end game training has been completed |
| Repository documentation | Initialized | See README and setup/fundamentals documents |

## Previously reported tabular run

The earlier session reported these single-run values from a recreated CliffWalking environment:

| Algorithm | Episodes | Last-50 mean reward | Last-50 mean steps | Greedy evaluation reward | Greedy evaluation steps |
|---|---:|---:|---:|---:|---:|
| SARSA | 500 | -26.10 | 22.14 | -17 | 17 |
| Q-Learning | 500 | -44.30 | 16.58 | -13 | 13 |

These are historical, preliminary results from the prior session. They have not been independently reproduced in this repository. Do not use them as a controlled algorithm comparison: a single seed, evaluation setup, and implementation details may affect the result.

## What is required to close the pending work

### Flappy Bird
1. Create an isolated Python environment.
2. Install PyTorch, Gymnasium, Pygame, and the selected Flappy Bird Gymnasium package.
3. Run an environment smoke test: reset, sample actions, step, verify observation and reward.
4. Run a short training smoke test and verify loss/parameter updates.
5. Run a longer training job with a fixed seed and periodic checkpoints.
6. Evaluate with exploration disabled across multiple episodes; save per-episode returns and a summary.

### Mario
1. Create a separate environment from Flappy Bird's environment.
2. Resolve the exact versions of Gym, gym-super-mario-bros, nes-py, NumPy, PyTorch, TorchVision, TensorDict, and TorchRL.
3. Verify emulator startup and one reset/step before training.
4. Run a short training smoke test.
5. Train and evaluate using a documented protocol; save logs and checkpoints.

## Integrity rule

A network forward-pass test is not a training run. A successful environment reset is not evidence of a learned policy. Only mark a project trained after an end-to-end training run and a separate evaluation run have completed.