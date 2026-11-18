# Execution Status

Last updated: 2026-11-18

This page separates code/architecture checks from actual environment training and evaluation. Status is based on the runs completed and reported for this repository; preliminary experiments are not treated as reproduced results.

## Summary

| Work item | Status | Evidence / limitation |
|---|---|---|
| RL fundamentals material | Reviewed | Learning notes are present; further revision and examples remain useful |
| SARSA / CliffWalking | Preliminary algorithm run reported | Recreated environment; not yet reproduced from a committed notebook or controlled multi-seed experiment |
| Q-Learning / CliffWalking | Preliminary algorithm run reported | Recreated environment; not yet reproduced from a committed notebook or controlled multi-seed experiment |
| Flappy Bird DQN architecture | Inspected | 12 state features, 256-unit hidden layer, two action values |
| Flappy Bird environment setup | Completed locally | Windows 11, Python 3.11.9, CPU-only PyTorch run |
| Flappy Bird end-to-end training | **Completed locally** | Training log records a best reward of 5.6 at episode 13,604 |
| Flappy Bird checkpoint | **Saved locally** | `runs/flappybirdv0.pt`; checkpoint is saved when a new best episode reward is reached |
| Flappy Bird greedy evaluation | **Completed locally** | 100 episodes; mean 1.455, population standard deviation 0.907, max 5.6, min -0.2 |
| Flappy Bird evaluation data | **Saved locally** | `runs/flappybirdv0_evaluation.csv`; raw per-episode CSV still needs to be committed to the repository |
| Flappy Bird visualizations | **Generated locally** | Training/evaluation plot and evaluation plot with a 10-episode moving average; image files still need to be committed |
| Mario Double DQN architecture | Preliminary shape/finite-output check reported | Does not validate emulator integration, optimization, or gameplay |
| Mario game training and evaluation | Pending | Environment compatibility, smoke test, training, and separate evaluation remain |
| Repository artifacts | Partially updated | Results are documented; local CSV and PNG artifacts are not automatically uploaded by this documentation update |

## Flappy Bird DQN: completed local run

### Environment

| Item | Recorded value |
|---|---|
| Operating system | Windows 11 |
| Python | 3.11.9 |
| PyTorch | 2.15.0+cpu (as reported by the local environment) |
| TorchVision | 0.29.2 |
| Gymnasium | 1.4.0 |
| flappy-bird-gymnasium | 0.4.0 |
| Pygame | 2.6.1 |
| NumPy | 1.26.4 |
| Matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| Compute | CPU; CUDA unavailable |

These are the versions reported during setup. A fresh `pip freeze` export should be retained with the experiment if exact environment reproduction is required.

### Training and checkpoint

- Training was launched with `python agent.py flappybirdv0 --train`.
- The training loop ran for at least 13,604 episodes.
- The best reward in the training log was **5.6 at episode 13,604**.
- The source saves the policy checkpoint when a new best episode reward is observed.
- Local checkpoint: `projects/flappy_bird_dqn/assigned_implementation/runs/flappybirdv0.pt`.
- Local training log: `projects/flappy_bird_dqn/assigned_implementation/runs/flappybirdv0.log`.

The best-reward log records improvements, not every episode's return. The training curve therefore represents best-so-far reward events, not a per-episode reward series.

### Evaluation

A separate evaluation copy was used so the assigned `agent.py` remained unchanged. Evaluation loaded the saved checkpoint, disabled epsilon-greedy exploration, and ran 100 episodes.

| Metric | Result |
|---|---:|
| Episodes | 100 |
| Mean episode reward | **1.455** |
| Population standard deviation | **0.907** |
| Maximum episode reward | **5.6** |
| Minimum episode reward | **-0.2** |

Local evaluation artifacts:
- `runs/flappybirdv0_evaluation.csv` — per-episode rewards.
- `runs/flappybird_dqn_performance.png` — training best-reward events and evaluation rewards.
- `runs/flappybird_dqn_evaluation_v2.png` — evaluation rewards and 10-episode moving average.

The raw CSV and PNG files were generated on the Windows machine. They still need to be added to Git if they are to appear in the public repository; this status/documentation update does not upload local files.

### Warnings and limitations

- Gymnasium emitted warnings that observations returned by the environment reset/step methods were outside the declared observation space. Evaluation nevertheless completed. This discrepancy has not yet been diagnosed and should be investigated before treating the environment integration as fully clean.
- The original evaluation call initially failed because the source print statement referenced `epsilon` even when training was disabled. The issue was addressed in the separate `agent_eval.py` copy; the assigned `agent.py` was not changed.
- No random seed was recorded for this run. The results are a single-run measurement, not a multi-seed estimate.
- The source training log does not contain every episode reward, episode length, loss, or environment-step count. These metrics cannot be reconstructed from the best-reward log.
- The 100-episode evaluation is a single checkpoint evaluation. It demonstrates observed performance under this run's setup, not guaranteed generalization to other environment versions or seeds.

## Previously reported tabular run

The earlier session reported these single-run values from a recreated CliffWalking environment:

| Algorithm | Episodes | Last-50 mean reward | Last-50 mean steps | Greedy evaluation reward | Greedy evaluation steps |
|---|---:|---:|---:|---:|---:|
| SARSA | 500 | -26.10 | 22.14 | -17 | 17 |
| Q-Learning | 500 | -44.30 | 16.58 | -13 | 13 |

These are historical, preliminary results from the prior session. They have not been independently reproduced in this repository. Do not use them as a controlled algorithm comparison: a single seed, evaluation setup, and implementation details may affect the result.

## Remaining work

### Flappy Bird
1. Commit the local evaluation CSV and generated PNG visualizations.
2. Save a complete package lock/export from the working virtual environment.
3. Investigate the Gymnasium observation-space warnings.
4. Optionally repeat evaluation with documented seeds and a fixed protocol.
5. If further training comparisons are desired, record per-episode metrics, episode lengths, loss, and environment steps rather than only best-reward events.

### Tabular RL
1. Add/review the SARSA and Q-Learning notebooks in the repository.
2. Reproduce both algorithms from committed notebooks.
3. Compare them with matched environment settings and multiple seeds.
4. Reproduce Cliff Walking results and record the policy/evaluation protocol.

### Mario
1. Create a separate environment from Flappy Bird's environment.
2. Resolve compatible versions of Gym, Gym-Super-Mario-Bros, nes-py, NumPy, PyTorch, TorchVision, TensorDict, and TorchRL.
3. Verify emulator startup and one reset/step before training.
4. Run a short training smoke test.
5. Train and evaluate using a documented protocol; save logs and checkpoints.

## Integrity rule

A network forward-pass test is not a training run. A successful environment reset is not evidence of a learned policy. Only mark a project trained after an end-to-end training run and a separate evaluation run have completed.
