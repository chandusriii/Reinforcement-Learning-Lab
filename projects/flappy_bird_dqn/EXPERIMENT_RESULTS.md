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
- Training log: [`results/flappybirdv0.log`](results/flappybirdv0.log).

The training log contains best-reward improvement events only. The training plot should therefore be interpreted as a best-so-far progression plot, not a complete per-episode reward curve.

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

## Published artifacts

The [results directory](results/README.md) contains the evaluation CSV, training log, and two PNG visualizations:

- [Evaluation rewards CSV](results/flappybirdv0_evaluation.csv)
- [Training log](results/flappybirdv0.log)
- [Figure 1](results/Figure_1.png)
- [Figure 2](results/fig%202.png)

The local model checkpoint remains in `assigned_implementation/runs/flappybirdv0.pt` and is intentionally not included in the repository.

## Issues observed

1. **Gymnasium observation-space mismatch:** this was reproduced and measured in GitHub Actions using the pinned Flappy Bird environment. The declared LiDAR Box is [0, 1], but a 3-episode diagnostic inspected 150 observations and found 2,944 individual values outside the Box, with a maximum of 1.0021062783. The updated separate `agent_eval.py` disables the premature passive checker and clips observations to the declared bounds before they enter the DQN. The raw diagnostic remains available in `assigned_implementation/diagnose_observations.py`; it does not clip the data. The original assigned `agent.py` remains unchanged and may still emit the upstream warnings.
2. **Evaluation print bug in assigned source:** the original agent's episode print statement references `epsilon` even when `is_training=False`, although epsilon is initialized only during training. The assigned `agent.py` was preserved; the fix was made in a separate evaluation copy.
3. **No recorded random seed:** the run is a single-run result. Repeating with fixed and multiple seeds is needed to estimate variability more robustly.
4. **Incomplete training telemetry:** the original training log does not preserve every episode return, episode length, loss, or environment-step count. Those metrics cannot be recovered from the best-reward-only log.

## Interpretation and next experiments

The run demonstrates that the assigned DQN implementation can train and that its saved checkpoint can be evaluated independently. It does not establish that the policy is optimal or robust across seeds and environment versions.

Recommended follow-up work:

- Run the updated seeded training/evaluation adapter on the local Windows setup and retain the new per-episode evaluation CSV.
- Export a complete `pip freeze` record from the local environment.
- Add periodic training metrics (episode return, episode length, epsilon, loss, and environment steps) in a separate experiment implementation.
- Compare configurations using matched training budgets and evaluation protocols.
- Keep the assigned coursework source unchanged; place all adaptations in separate files.
