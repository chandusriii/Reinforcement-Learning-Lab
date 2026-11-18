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

Review the copied files, then commit and push the `results/` directory. The model checkpoint (`.pt`) is intentionally not included by default because model weights can add repository size; consider Git LFS or a release asset if sharing the checkpoint is required.

## Recorded evaluation summary

- Episodes: 100
- Mean reward: 1.455
- Population standard deviation: 0.907
- Maximum reward: 5.6
- Minimum reward: -0.2

The measured protocol and caveats are documented in [EXPERIMENT_RESULTS.md](../EXPERIMENT_RESULTS.md).
