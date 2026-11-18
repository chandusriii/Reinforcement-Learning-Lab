# 05 — Experiment Guide

This guide defines how experiments in this repository are configured, executed, evaluated, and documented. It is intended to make results understandable and repeatable.

## 1. Before training

Record:
- Project and algorithm (SARSA, Q-Learning, DQN, Double DQN).
- Environment ID and version.
- Observation and action spaces.
- Preprocessing and wrappers.
- Python, framework, and package versions.
- Device (CPU/GPU).
- Random seed(s).
- Training budget (episodes, steps, or frames).

## 2. Configuration

Keep hyperparameters in one configuration file or a clearly defined configuration object. Record learning rate, discount factor, epsilon start/end/decay, replay capacity, batch size, target-network update interval, optimizer, and network architecture as applicable.

Change one or a small number of variables at a time when comparing configurations. This makes it easier to understand which change affected the result.

## 3. Training metrics

At minimum, record:
- Episode number.
- Episode return.
- Episode length.
- Epsilon (for exploratory policies).
- Loss (for neural value learning, when available).
- Environment steps and training updates.

Use a moving average for visualization, but retain raw episode data.

## 4. Evaluation

Evaluate separately from training:
- Use a fixed evaluation protocol.
- State whether exploration is disabled.
- Record the number of evaluation episodes.
- Report mean and standard deviation, not only the best episode.
- Use multiple seeds where practical.
- Include environment and wrapper versions.

For game projects, include task-specific measures such as score, survival time, level progress, or completion rate when available.

## 5. Artifacts

Organize outputs by project and experiment identifier:

```text
experiments/
├── logs/
│   └── <project>/<run-id>/
├── results/
│   └── <project>/<run-id>/
└── EXPERIMENT_TEMPLATE.md
```

A run folder may contain:
- config.json or config.yaml
- metrics.csv
- training_curve.png
- evaluation.json
- checkpoint metadata
- README.md describing observations

Do not commit large model weights or raw recordings without considering repository size. Use Git LFS or release assets when appropriate.

## 6. Interpreting results

A rising training reward does not automatically prove that the policy generalizes. Training returns may include exploration and can be noisy. Compare greedy evaluation performance and test across seeds or held-out conditions where possible.

Do not compare runs with different environments, reward definitions, episode limits, or evaluation protocols without explicitly noting the differences.

## 7. Experiment write-up template

For each run, answer:
1. What question were you testing?
2. Which algorithm and configuration were used?
3. What changed from the previous run?
4. What did the training curve show?
5. How did the agent perform during evaluation?
6. What failed or behaved unexpectedly?
7. What did you learn, and what is the next experiment?

Only enter measured values after running the experiment. Never fill a result field with an estimate or an illustrative number.
