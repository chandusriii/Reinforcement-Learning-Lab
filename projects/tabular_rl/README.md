# Tabular Reinforcement Learning — CliffWalking

## Overview

This folder preserves the supplied SARSA and Q-Learning notebooks for Gymnasium's CliffWalking-v1 environment. Both methods use a tabular action-value function with 48 states and 4 actions.

The notebooks are retained as the original learning artifacts, including their training output and final greedy-policy evaluation. The summary statistics below were calculated from the recorded episode logs in those notebooks.

## Algorithms

**SARSA (on-policy TD control)** updates the current state-action value using the next action actually selected by the epsilon-greedy behaviour policy:

Q(s,a) <- Q(s,a) + alpha [r + gamma Q(s',a') - Q(s,a)]

**Q-Learning (off-policy TD control)** updates toward the maximum estimated next-state action value:

Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]

Both notebooks use gamma = 0.99, alpha = 0.5, epsilon = 0.1, and 500 training episodes.

## Recorded single-run results

| Metric | SARSA | Q-Learning |
|---|---:|---:|
| Training episodes | 500 | 500 |
| Last-50 mean episode return | -22.20 | -57.04 |
| Last-50 mean episode length | 20.22 | 17.44 |
| Greedy evaluation return | -17 | -13 |
| Greedy evaluation length | 17 | 13 |

Returns in CliffWalking are negative step costs, with additional penalty for falling from the cliff. These values describe the recorded run only. The notebooks do not set an explicit random seed, so a fresh execution can produce different results. Do not treat this as a multi-seed benchmark or a definitive algorithm ranking.

## Files

- [SARSA notebook](notebooks/SARSA.ipynb)
- [Q-Learning notebook](notebooks/Q_Learning.ipynb)

## Reproducible multi-seed comparison

A standalone experiment runner now compares SARSA and Q-Learning using the same ten seeds, matching the notebook hyperparameters (500 episodes, alpha 0.5, gamma 0.99, epsilon 0.1). It runs a separate 20-episode greedy evaluation for every trained seed, then writes per-seed metrics, aggregate means and population standard deviations, and run metadata.

Install the pinned dependencies and run it from the repository root:

    python -m pip install -r requirements/tabular-rl.txt
    python projects/tabular_rl/experiments/compare_seeds.py

You can change the protocol explicitly, for example:

    python projects/tabular_rl/experiments/compare_seeds.py --seeds 1 2 3 4 5 --episodes 500 --evaluation-episodes 20

Outputs are written to `projects/tabular_rl/results/multi_seed/`. A GitHub Actions workflow runs the ten-seed experiment. The committed [per-seed CSV](results/multi_seed/per_seed.csv), [summary CSV](results/multi_seed/summary.csv), and [metadata](results/multi_seed/metadata.json) are the actual output of [successful workflow run 44330224587](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44330224587). The original coursework notebooks remain unchanged.

### Measured ten-seed results

| Measure | SARSA | Q-Learning |
|---|---:|---:|
| Mean of last-50 training returns (mean ± population SD across seeds) | -24.39 ± 2.34 | -42.71 ± 9.62 |
| Mean greedy evaluation return across seeds | -215.30 ± 594.90 | -13.00 ± 0.00 |
| Mean greedy evaluation steps across seeds | 215.30 ± 594.90 | 13.00 ± 0.00 |
| Mean greedy success rate | 90% | 100% |

Each seed used 500 training episodes and 20 greedy evaluation episodes, with a 2,000-step cap. Q-Learning reached the goal in all 200 evaluation episodes. SARSA reached the goal in 180/200; all 20 failures occurred for seed 49, whose greedy policy did not reach the goal before the step cap. This outlier is retained in the published results rather than hidden or removed. The two algorithms have different training-return patterns and the evaluation result does not establish a universal algorithm ranking; it is specific to these settings and seeds.

Open either notebook in Jupyter to reproduce the original single-run coursework outputs. The notebooks include rendering during selected training episodes; on a headless machine, disable the human-rendering cells before running.

## Interpretation

SARSA is on-policy: its update accounts for the next action selected by the current exploratory policy. Q-Learning is off-policy: its update uses the greedy next-state action value. Their observed greedy evaluation returns here are close, while their recorded learning trajectories differ. A single unseeded run is not sufficient to establish that one method is generally better.

## Source integrity

The notebooks supplied for the coursework are preserved in their original form. The statistics above are derived from their recorded output; they are not a claim of a newly executed or multi-seed experiment.
