# Mario Double DQN — Experiment Results

## Scope

These are measured results from two short, real-emulator experiments. Both runs used the Super Mario Bros. World 1-1 environment, the committed preprocessing and Double DQN pipeline, and a separate greedy evaluation. The experiments verify that the emulator, training loop, checkpointing, and evaluation path work end-to-end. They do **not** establish that the policy has learned to complete the level.

## Environment and shared configuration

- Python 3.10
- Gym 0.26.2
- gym-super-mario-bros 7.4.0
- nes-py 8.2.1
- PyTorch 2.2.2, CPU
- Environment: `SuperMarioBros-1-1-v0`
- Actions: move right; move right + jump
- Observation: grayscale 84 × 84 pixels, four stacked frames
- Action repeat / frame skip: 4
- Random seed: 42 for training; 10042 onward for greedy evaluation
- Replay capacity: 2,048 transitions
- Batch size: 32
- Burn-in: 128 transitions
- Learning interval: every 3 environment actions
- Target-network synchronization: every 1,000 actions
- Maximum training/evaluation steps: 256 wrapped environment steps per episode
- Optimizer: Adam; learning rate 2.5e-4
- Discount factor: 0.9
- Loss: Smooth L1

The intentionally short 256-step episode cap makes these integration experiments bounded. It is not a full-length Mario training schedule.

## Results

| Metric | Tutorial-style schedule | Faster epsilon decay |
|---|---:|---:|
| Training episodes | 40 | 100 |
| Epsilon decay | 0.99999975 | 0.9995 |
| Mean training return | 537.78 | 532.24 |
| Training return standard deviation (population) | 247.99 | 275.85 |
| Best training return | 1,236 (episode 39) | 1,346 (episode 26) |
| Mean greedy evaluation return | 626.00 | 231.00 |
| Greedy evaluation standard deviation | 0.00 | 0.00 |
| Evaluation episodes | 10 | 10 |
| Level flag reached | 0/10 | 0/10 |

The faster-decay run reached the minimum exploration rate sooner but produced a lower greedy evaluation return in this specific experiment. This is evidence that the exploration schedule matters; two short, single-seed runs are not enough to conclude that one schedule is generally superior.

**Important:** none of the 20 greedy evaluation episodes reached the level flag. The reported episode rewards are progress/reward signals, not level-completion results. The trained checkpoint should therefore be described as an experimental Double DQN policy, not as an agent that reliably beats Mario.

## Published raw metrics

### Tutorial-style schedule — 40 episodes

- [Training metrics CSV](training_40ep.csv)
- [Greedy evaluation CSV — 10 episodes](evaluation_40ep_greedy10.csv)
- [GitHub Actions run](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44290496743)

### Faster epsilon-decay experiment — 100 episodes

- [Training metrics CSV](training_100ep_fast_epsilon.csv)
- [Greedy evaluation CSV — 10 episodes](evaluation_100ep_greedy10.csv)
- [GitHub Actions run](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44290512314)

The workflow artifacts contain the corresponding best/latest checkpoints and uploaded CSVs. GitHub Actions retains those binary artifacts for 30 days; they are not stored in Git to avoid committing large model weights and replay files.

## Validation

- Real NES emulator package installation succeeded in CI.
- Emulator smoke test passed: environment reset, action step, and four-frame observation shape.
- Core unit tests passed, including an explicit deterministic assertion that the online network selects the next action while the target network evaluates that selected action.
- Training completed and saved checkpoints in both experiment runs.
- Separate greedy evaluation completed and wrote the evaluation CSVs.

Links:
- [Core unit tests — GitHub Actions](https://github.com/chandusriii/Reinforcement-Learning-Lab/actions/runs/44290709856)
- [Mario implementation and usage](../README.md)

## Interpretation and next training stage

The implementation and end-to-end pipeline are validated. Gameplay mastery is not yet demonstrated. The PyTorch tutorial's longer training schedule is on the order of 40,000 episodes; the 40- and 100-episode experiments here are short integration and hyperparameter checks. A meaningful level-completion claim requires a much longer training run, longer episodes, and evaluation across multiple independent seeds.

Do not use the current evaluation scores as proof of a winning policy. Report the current status accurately as **Double DQN implementation and short-run training/evaluation completed; level completion not achieved**.

## References

- [PyTorch — Train a Mario-playing RL Agent](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
- Hado van Hasselt, Arthur Guez, and David Silver, [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461), AAAI 2016.
