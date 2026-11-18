# Flappy Bird DQN — Experiment Results

## Run summary

A local end-to-end training run was completed on Windows 11, followed by a separate greedy evaluation using the saved policy checkpoint.

| Field | Recorded value |
|---|---|
| Algorithm | Standard Deep Q-Network (DQN) |
| Environment | `FlappyBird-v0` from `flappy-bird-gymnasium` |
| Operating system | Windows 11 |
| Python | 3.11.9 |
| PyTorch | 2.15.0+cpu |
| Gymnasium | 1.4.0 |
| flappy-bird-gymnasium | 0.4.0 |
| NumPy | 1.26.4 |
| Pygame | 2.6.1 |
| Matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| Device | CPU; CUDA unavailable |
| Training command | `python agent.py flappybirdv0 --train` |
| Evaluation command | `python agent_eval.py flappybirdv0` |

These package versions were read from the local setup record. For strict reproduction, retain the complete output of `python -m pip freeze` from the training environment.

## Training result

- The training run reached at least **13,604 episodes**.
- The best recorded episode reward was **5.6 at episode 13,604**.
- The source code saves the policy checkpoint whenever a new best episode reward is achieved.
- Local checkpoint: `assigned_implementation/runs/flappybirdv0.pt`.
- Local training log: `assigned_implementation/runs/flappybirdv0.log`.

The training log contains best-reward improvement events only. The training plot is therefore a best-so-far progression plot, not a complete per-episode reward curve.

## Evaluation protocol and result

The saved checkpoint was loaded in evaluation mode. Epsilon-greedy exploration was disabled; the policy selected actions greedily from the network's Q-values. The evaluation copy ran 100 episodes and recorded each episode return.

| Metric | Result |
|---|---:|
| Evaluation episodes | 100 |
| Mean episode reward | **1.455** |
| Population standard deviation | **0.907** |
| Maximum episode reward | **5.6** |
| Minimum episode reward | **-0.2** |

The positive mean indicates that the checkpoint obtained positive cumulative reward on average in this 100-episode evaluation. The standard deviation and reward plot show episode-to-episode variation. The maximum reward is a best case, not a measure of typical performance.

## Local artifacts

The following files were generated on the Windows machine:

| Artifact | Purpose |
|---|---|
| `assigned_implementation/runs/flappybirdv0.pt` | Saved DQN policy checkpoint |
| `assigned_implementation/runs/flappybirdv0.log` | Best-reward training log |
| `assigned_implementation/runs/flappybirdv0_evaluation.csv` | Per-episode rewards for 100 evaluation episodes |
| `assigned_implementation/runs/flappybird_dqn_performance.png` | Training best-reward events and evaluation rewards |
| `assigned_implementation/runs/flappybird_dqn_evaluation_v2.png` | Evaluation rewards with a 10-episode moving average |

The evaluation CSV and training log have been published under [`results/`](results/README.md). The two PNG visualizations remain local and were not found in the GitHub results directory at the time of verification. The model checkpoint remains local and is intentionally not included in the repository.

## Issues observed

1. **Gymnasium observation-space warnings:** the environment emitted warnings that observations returned by `reset()` and `step()` were outside the declared observation space. Training and evaluation completed, but the cause has not yet been diagnosed.
2. **Evaluation print bug in assigned source:** the original agent's episode print statement references `epsilon` even when `is_training=False`, although epsilon is initialized only during training. The assigned `agent.py` was preserved; the fix was made in a separate evaluation copy.
3. **No recorded random seed:** the run is a single-run result. Repeating with fixed and multiple seeds is needed to estimate variability more robustly.
4. **Incomplete training telemetry:** the original training log does not preserve every episode return, episode length, loss, or environment-step count. Those metrics cannot be recovered from the best-reward-only log.

## Interpretation and next experiments

The run demonstrates that the assigned DQN implementation can train and that its saved checkpoint can be evaluated independently. It does not establish that the policy is optimal or robust across seeds and environment versions.

Recommended follow-up work:

- Diagnose the observation-space warnings by inspecting the environment's declared bounds and returned observations.
- Repeat evaluation using documented seeds and retain all per-episode returns.
- Add periodic training metrics (episode return, episode length, epsilon, loss, and environment steps) in a separate experiment implementation.
- Compare configurations using matched training budgets and evaluation protocols.
- Keep the assigned coursework source unchanged; place all adaptations in separate files.
