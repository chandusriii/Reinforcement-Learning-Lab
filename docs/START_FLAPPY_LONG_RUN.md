# Flappy Bird long-run launch marker

This file triggers the initial 20,000-episode seeded Flappy Bird DQN workflow once.
The workflow uses seed 42, a maximum of 10,000 steps per episode, checkpoints every
500 episodes, and a separate 100-episode greedy evaluation. The generated checkpoint,
replay memory, per-episode metrics, and evaluation CSV are uploaded as the
`flappy-long-training-state` GitHub Actions artifact.

Do not modify this marker unless intentionally starting another initial long run.
