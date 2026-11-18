# Execution Status

Last updated: 2026-11-18

This page separates code/architecture checks from actual environment training and evaluation. Status is based on the runs completed and reported for this repository; preliminary experiments are not treated as reproduced results.

## Summary

| Work item | Status | Evidence / limitation |
|---|---|---|
| RL fundamentals material | Reviewed | Learning notes are present; further revision and examples remain useful |
| SARSA / CliffWalking | Source notebook committed; single run retained | 500-episode CliffWalking run and greedy evaluation are included in projects/tabular_rl/notebooks/SARSA.ipynb; no seed or multi-seed comparison |
| Q-Learning / CliffWalking | Source notebook committed; single run retained | 500-episode CliffWalking run and greedy evaluation are included in projects/tabular_rl/notebooks/Q_Learning.ipynb; no seed or multi-seed comparison |
| Flappy Bird DQN architecture | Inspected | 180 LiDAR observations in the executed configuration, 256-unit hidden layer, two action values |
| Flappy Bird environment setup | Completed locally | Windows 11, Python 3.11.9, CPU-only PyTorch run |
| Flappy Bird end-to-end training | **Completed locally** | Training log records a best reward of 5.6 at episode 13,604 |
| Flappy Bird checkpoint | **Saved locally** | `runs/flappybirdv0.pt`; checkpoint is saved when a new best episode reward is reached |
| Flappy Bird greedy evaluation | **Completed locally** | 100 episodes; mean 1.455, population standard deviation 0.907, max 5.6, min -0.2 |
| Flappy Bird evaluation data | **Published** | Evaluation CSV committed under `projects/flappy_bird_dqn/results/` |
| Flappy Bird visualizations | **Published** | Two PNG files are committed under `projects/flappy_bird_dqn/results/` |
| Repository artifacts | **Published** | Evaluation CSV, training log, and two PNG visualizations are available in the results directory |
| Mario Double DQN implementation | **Implemented** | Tutorial-aligned CNN, online/target networks, disk-backed replay, frame preprocessing, training, checkpointing, greedy evaluation, smoke test and core tests are committed under `projects/mario_double_dqn/code/` |
| Mario Double DQN core validation | **Passed in GitHub Actions** | Python compileall and `code/test_core.py` completed successfully on Python 3.11 / CPU PyTorch; [workflow run](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44291824310) |
| Mario emulator smoke test | **Passed in GitHub Actions** | Python 3.10, Gym 0.26.2, gym-super-mario-bros 7.4.0, nes-py 8.2.1; reset, step, and 4 × 84 × 84 observation checks passed |
| Mario 40-episode bounded integration run + 10-episode greedy evaluation | **Completed** | Mean training return 537.78; best 1,236; greedy evaluation mean 626.0, std 0.0; flag reached 0/10 |
| Mario 100-episode bounded faster-decay integration run + 10-episode greedy evaluation | **Completed** | Mean training return 532.24; best 1,346; greedy evaluation mean 231.0, std 0.0; flag reached 0/10 |
| Mario core unit tests | **Passed** | Includes deterministic Double DQN online-action selection / target-network evaluation and terminal masking; latest run: [GitHub Actions](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44291824310) |
| Mario experiment data | **Published** | 40-episode and 100-episode training CSVs, plus separate 10-episode greedy-evaluation CSVs, are committed under `projects/mario_double_dqn/results/` |


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
- Published training log: [`flappybirdv0.log`](../projects/flappy_bird_dqn/results/flappybirdv0.log).

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

Published artifacts:

- [Evaluation CSV](../projects/flappy_bird_dqn/results/flappybirdv0_evaluation.csv)
- [Training log](../projects/flappy_bird_dqn/results/flappybirdv0.log)
- [Figure 1](../projects/flappy_bird_dqn/results/Figure_1.png)
- [Figure 2](../projects/flappy_bird_dqn/results/fig%202.png)

The model checkpoint remains local and is intentionally not published.

### Warnings and limitations

- Gymnasium emitted warnings that observations returned by the environment reset/step methods were outside the declared observation space. Evaluation nevertheless completed. This discrepancy has not yet been diagnosed and should be investigated before treating the environment integration as fully clean.
- The original evaluation call initially failed because the source print statement referenced `epsilon` even when training was disabled. The issue was addressed in the separate `agent_eval.py` copy; the assigned `agent.py` was not changed.
- No random seed was recorded for this run. The results are a single-run measurement, not a multi-seed estimate.
- The source training log does not contain every episode reward, episode length, loss, or environment-step count. These metrics cannot be reconstructed from the best-reward log.
- The 100-episode evaluation is a single checkpoint evaluation. It demonstrates observed performance under this run's setup, not guaranteed generalization to other environment versions or seeds.

## Tabular RL recorded notebook results

The earlier session reported these single-run values from a recreated CliffWalking environment:

| Algorithm | Episodes | Last-50 mean reward | Last-50 mean steps | Greedy evaluation reward | Greedy evaluation steps |
|---|---:|---:|---:|---:|---:|
| SARSA | 500 | -22.20 | 20.22 | -17 | 17 |
| Q-Learning | 500 | -57.04 | 17.44 | -13 | 13 |

These values were recalculated from the recorded episode outputs and greedy-evaluation cells in the two supplied notebooks now committed under projects/tabular_rl/notebooks/. They are single-run observations: the notebooks do not set a random seed, and the values should not be presented as a controlled or multi-seed algorithm comparison.

## Remaining work

### Flappy Bird

1. Save a complete package lock/export from the working virtual environment.
2. Investigate the Gymnasium observation-space warnings.
3. Optionally repeat evaluation with documented seeds and a fixed protocol.
4. If further training comparisons are desired, record per-episode metrics, episode lengths, loss, and environment steps rather than only best-reward events.

### Tabular RL

1. Review the supplied SARSA and Q-Learning notebooks and preserve their original versions.
2. Add explicit random seeds and repeat both algorithms over the same set of seeds.
3. Record per-episode returns and lengths, then compare mean and variability across runs.
4. Keep the greedy evaluation protocol identical for both algorithms.

### Mario

1. The implementation, emulator smoke test, short training runs, and separate greedy evaluations are complete and recorded in `projects/mario_double_dqn/results/`.
2. The current 10-episode greedy evaluations did not reach the level flag. Continue the longer training schedule (the PyTorch tutorial recommends at least 40,000 episodes) before claiming a successfully completed Mario-playing agent.
3. When extending training, compare checkpoints with a fixed greedy evaluation protocol rather than selecting solely by exploratory episode reward.

## Integrity rule

A network forward-pass test is not a training run. A successful environment reset is not evidence of a learned policy. Only mark a project trained after an end-to-end training run and a separate evaluation run have completed.
