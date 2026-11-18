# Execution Status

Last updated: 2026-11-18

This page separates code/architecture checks from actual environment training and evaluation. Status is based on the runs completed and reported for this repository; preliminary experiments are not treated as reproduced results.

## Summary

| Work item | Status | Evidence / limitation |
|---|---|---|
| RL fundamentals material | Reviewed | Learning notes are present; further revision and examples remain useful |
| SARSA / CliffWalking | **Multi-seed comparison completed** | 10 matched seeds × 500 training episodes and 20 greedy evaluations per seed; per-seed CSV and summary are committed under projects/tabular_rl/results/multi_seed/ |
| Q-Learning / CliffWalking | **Multi-seed comparison completed** | Same 10 seeds and evaluation protocol; all 200 greedy evaluation episodes reached the goal in this experiment |
| Flappy Bird DQN architecture | Inspected | 180 LiDAR observations in the executed configuration, 256-unit hidden layer, two action values |
| Flappy Bird environment setup | Completed locally; CI diagnostic passed | Windows 11, Python 3.11.9, CPU-only PyTorch run; reproducible environment diagnostic in GitHub Actions |
| Flappy Bird bounded runner validation | **Passed in GitHub Actions** | Pinned dependencies, train → resume → greedy evaluation, versioned checkpoint, replay persistence, and per-episode metrics passed; [latest workflow run](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44335603089) |
| Flappy Bird end-to-end training | **Completed locally** | Training log records a best reward of 5.6 at episode 13,604 |
| Flappy Bird long training | **In progress** | Initial seeded 20,000-episode run started in [GitHub Actions](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44334994493); training/evaluation results are pending |
| Flappy Bird checkpoint | **Saved locally** | `runs/flappybirdv0.pt`; checkpoint is saved when a new best episode reward is reached |
| Flappy Bird greedy evaluation | **Completed locally (historical run)** | 100 episodes; mean 1.455, population standard deviation 0.907, max 5.6, min -0.2; historical run was unseeded |
| Flappy Bird evaluation data | **Published** | Historical unseeded evaluation CSV committed under `projects/flappy_bird_dqn/results/`; updated seeded evaluator is implemented separately |
| Flappy Bird visualizations | **Published** | Two PNG files are committed under `projects/flappy_bird_dqn/results/` |
| Repository artifacts | **Published** | Evaluation CSV, training log, and two PNG visualizations are available in the results directory |
| Mario Double DQN implementation | **Implemented** | Tutorial-aligned CNN, online/target networks, disk-backed replay, frame preprocessing, training, checkpointing, greedy evaluation, smoke test and core tests are committed under `projects/mario_double_dqn/code/` |
| Mario Double DQN core validation | **Passed in GitHub Actions** | Compileall and CPU tests for Double DQN target selection, terminal masking, replay resume, evaluation-only checkpoint loading, and RNG restoration; [latest workflow run](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44335418978) |
| Mario emulator smoke test | **Passed in GitHub Actions** | Python 3.10, Gym 0.26.2, gym-super-mario-bros 7.4.0, nes-py 8.2.1; reset, step, and 4 × 84 × 84 observation checks passed; [resume workflow](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44332415242) also validates a real environment train → resume → evaluate cycle |
| Mario long training | **In progress** | Initial 1,000-episode seeded chunk started in [GitHub Actions](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44335214352); 40,000-episode target is not yet reached |
| Mario 40-episode bounded integration run + 10-episode greedy evaluation | **Completed** | Mean training return 537.78; best 1,236; greedy evaluation mean 626.0, std 0.0; flag reached 0/10 |
| Mario 100-episode bounded faster-decay integration run + 10-episode greedy evaluation | **Completed** | Mean training return 532.24; best 1,346; greedy evaluation mean 231.0, std 0.0; flag reached 0/10 |
| Mario core unit tests | **Passed** | Includes Double DQN action selection, terminal masking, checkpoint load, and persistent replay-buffer resume; [GitHub Actions](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44328956974) |
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

- A pinned CI diagnostic reproduced the raw LiDAR space mismatch: declared bounds [0, 1], while 150 sampled observations contained 2,944 values outside the Box, with maximum 1.0021062783. The separate seeded `agent_eval.py` now disables the passive checker and clips values to the declared bounds before DQN input. The raw diagnostic remains available; original assigned `agent.py` is preserved unchanged. [latest pinned-dependency diagnostic and training/resume/evaluation smoke workflow](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44335603089).
- The original evaluation call initially failed because the source print statement referenced `epsilon` even when training was disabled. The issue was addressed in the separate `agent_eval.py` copy; the assigned `agent.py` was not changed.
- The historical 13,604-episode run did not record a seed. The updated separate training/evaluation runner accepts a seed and has a finite episode budget; new seeded performance results have not yet been collected.
- The source training log does not contain every episode reward, episode length, loss, or environment-step count. These metrics cannot be reconstructed from the best-reward log.
- The 100-episode evaluation is a single checkpoint evaluation. It demonstrates observed performance under this run's setup, not guaranteed generalization to other environment versions or seeds.

## Tabular RL multi-seed comparison

A separate runner was added without modifying the supplied notebooks. A successful GitHub Actions run tested Gymnasium 1.4.0 on ten matched seeds (42–51), 500 training episodes per seed, and 20 greedy evaluation episodes per seed. Evaluation is capped at 2,000 steps to prevent an unlearned policy from looping forever.

| Metric | SARSA | Q-Learning |
|---|---:|---:|
| Last-50 training return, mean ± population SD across seeds | -24.39 ± 2.34 | -42.71 ± 9.62 |
| Greedy evaluation return, mean ± population SD across seeds | -215.30 ± 594.90 | -13.00 ± 0.00 |
| Greedy success rate averaged across seeds | 90% | 100% |

Q-Learning reached the goal in all 200 greedy evaluation episodes. SARSA reached the goal in 180/200; all 20 failures were from seed 49, whose greedy policy did not reach the goal within the 2,000-step cap. This is retained in the published raw data and not omitted from the mean.

- [Per-seed results](../projects/tabular_rl/results/multi_seed/per_seed.csv)
- [Summary statistics](../projects/tabular_rl/results/multi_seed/summary.csv)
- [Experiment metadata](../projects/tabular_rl/results/multi_seed/metadata.json)
- [Successful CI run](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44333090235)

## Tabular RL recorded notebook results

The earlier session reported these single-run values from a recreated CliffWalking environment:

| Algorithm | Episodes | Last-50 mean reward | Last-50 mean steps | Greedy evaluation reward | Greedy evaluation steps |
|---|---:|---:|---:|---:|---:|
| SARSA | 500 | -22.20 | 20.22 | -17 | 17 |
| Q-Learning | 500 | -57.04 | 17.44 | -13 | 13 |

These values were recalculated from the recorded episode outputs and greedy-evaluation cells in the two supplied notebooks now committed under projects/tabular_rl/notebooks/. They are single-run observations: the notebooks do not set a random seed, and the values should not be presented as a controlled or multi-seed algorithm comparison.

## Remaining work

### Flappy Bird

1. The first 20,000-episode seeded workflow run is underway. Inspect its result and uploaded artifact before deciding whether a continuation chunk is needed.
2. Export a complete `pip freeze` record from that machine.
3. The recorded historical best-reward log still lacks per-episode loss and telemetry; collect those metrics during any new training run.

### Tabular RL

1. The matched 10-seed comparison is complete and published. Seed 49 shows a SARSA greedy-policy failure; an additional SARSA training/tie-handling investigation is optional, not hidden from the current results.
2. Preserve the original 500-episode coursework notebooks alongside the new comparison output.

### Mario

1. Implementation, emulator smoke test, core tests, checkpointing, persistent replay, and safe resume logic are validated.
2. The current greedy evaluations reached the level flag 0/20 times. A long training run (the PyTorch tutorial recommends at least 40,000 episodes) has not been performed. Run that workload on the user's hardware or a suitable GPU runner, then evaluate across multiple seeds before claiming Mario completes the level.
3. Keep the latest checkpoint, training CSV, and replay-memory directory together when resuming; do not resume from the older short-run artifact checkpoints created before persistent replay was added.

## Integrity rule

A network forward-pass test is not a training run. A successful environment reset is not evidence of a learned policy. Only mark a project trained after an end-to-end training run and a separate evaluation run have completed.
