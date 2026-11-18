# Flappy Bird result artifacts

The local training run writes output to `assigned_implementation/runs/`. That folder is excluded by the imported `.gitignore`, so selected files must be copied here before they can be committed.

From the repository root in Windows CMD, after the local run files exist:

```cmd
mkdir projects\flappy_bird_dqn\results
copy projects\flappy_bird_dqn\assigned_implementation\runs\flappybirdv0_evaluation.csv projects\flappy_bird_dqn\results\
copy projects\flappy_bird_dqn\assigned_implementation\runs\flappybird_dqn_performance.png projects\flappy_bird_dqn\results\
copy projects\flappy_bird_dqn\assigned_implementation\runs\flappybird_dqn_evaluation_v2.png projects\flappy_bird_dqn\results\
copy projects\flappy_bird_dqn\assigned_implementation\runs\flappybirdv0.log projects\flappy_bird_dqn\results\
```

The evaluation CSV and training log are already published in this directory. The two PNG visualizations are still pending upload. Once copied into this directory, commit and push them. The model checkpoint (`.pt`) is intentionally not included by default because model weights can add repository size; consider Git LFS or a release asset if sharing the checkpoint is required. The model checkpoint (`.pt`) is intentionally not included by default because model weights can add repository size; consider Git LFS or a release asset if sharing the checkpoint is required.

## Published files\n\n- [`flappybirdv0_evaluation.csv`](flappybirdv0_evaluation.csv) — 100 per-episode evaluation rewards.\n- [`flappybirdv0.log`](flappybirdv0.log) — best-reward training progress log.\n\nThe two plot files exist locally under `assigned_implementation/runs/` but have not yet been verified in this GitHub directory.\n\n## Recorded evaluation summary

- Episodes: 100
- Mean reward: 1.455
- Population standard deviation: 0.907
- Maximum reward: 5.6
- Minimum reward: -0.2

The measured protocol and caveats are documented in [EXPERIMENT_RESULTS.md](../EXPERIMENT_RESULTS.md).
