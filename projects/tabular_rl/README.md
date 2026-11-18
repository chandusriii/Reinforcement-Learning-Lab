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

## Reproduction

Install Gymnasium with its toy-text dependencies and NumPy in a Python environment that supports the package versions you choose:

    python -m pip install "gymnasium[toy-text]" numpy jupyter

Open either notebook in Jupyter and run the cells in order. The original notebooks include rendering during selected training episodes and a final greedy evaluation. On a headless machine, remove or disable the human-rendering cells before running.

For a stronger experiment, add explicit seeds, repeat each algorithm over the same set of seeds, keep environment settings matched, and report mean and variability across runs.

## Interpretation

SARSA is on-policy: its update accounts for the next action selected by the current exploratory policy. Q-Learning is off-policy: its update uses the greedy next-state action value. Their observed greedy evaluation returns here are close, while their recorded learning trajectories differ. A single unseeded run is not sufficient to establish that one method is generally better.

## Source integrity

The notebooks supplied for the coursework are preserved in their original form. The statistics above are derived from their recorded output; they are not a claim of a newly executed or multi-seed experiment.
