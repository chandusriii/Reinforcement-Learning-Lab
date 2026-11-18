# Super Mario Bros — Double Deep Q-Network (Double DQN)

## Project overview

This project explores an AI agent that learns to play Super Mario Bros through reinforcement learning. The agent observes processed game frames, selects an available action, receives reward feedback from the environment, and improves its action-value estimates over repeated interactions.

The assignment specifies a Mario-playing agent using **Double Deep Q-Networks (Double DQN)** and links to the PyTorch tutorial as the implementation guide.

## Learning objectives

- Understand how a visual game environment can be converted into observations suitable for a neural network.
- Apply frame preprocessing, resizing, skipping, and stacking.
- Use a convolutional neural network to estimate action values from image observations.
- Understand experience replay and target networks.
- Implement epsilon-greedy action selection.
- Distinguish standard DQN from Double DQN.
- Track training behaviour and evaluate the learned policy.

## Architecture and learning workflow

1. **Observation preprocessing:** transform game frames into a consistent compact representation.
2. **Frame stacking:** combine recent observations to provide temporal context.
3. **Q-network:** use a convolutional neural network to estimate Q-values for available actions.
4. **Action selection:** use epsilon-greedy exploration during training.
5. **Experience replay:** store transitions and sample mini-batches for learning.
6. **Double DQN target:** select the next action using the online network and evaluate it using the target network.
7. **Optimization:** update online network parameters from the TD loss.
8. **Evaluation:** test the learned policy separately from exploratory training and record task performance.

### Double DQN target

`a* = argmax_a Q_online(s', a)`

`y = r + gamma Q_target(s', a*)`

For a true terminal transition, the bootstrap term is removed.

## Technology

- Python
- PyTorch
- Convolutional neural networks
- Reinforcement Learning / Double DQN
- Gym-based Super Mario environment (tutorial dependency stack)

## Reference implementation guide

The supplied assignment points to the [official PyTorch Mario RL tutorial](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html). The tutorial demonstrates frame preprocessing, stacked observations, replay memory, online and target networks, and Double DQN learning.

## Environment compatibility

The tutorial uses a legacy Gym-based dependency stack. Keep Mario in an isolated virtual environment and verify compatible versions before installation. The NES emulator dependency may require native build tools on Windows if a compatible wheel is unavailable. See the [setup guide](../../docs/SETUP_GUIDE.md).

Do not redistribute ROM files or copyrighted game assets.

## Experiment and evaluation

Record the environment version, wrappers, preprocessing settings, random seed, training steps, optimizer settings, epsilon schedule, replay-buffer configuration, target update schedule, episode returns, evaluation protocol, and checkpoint metadata.

The actual emulator training run and evaluation must be executed before adding performance claims, reward curves, or checkpoints. The project documentation describes the intended implementation and learning workflow; it does not claim unverified gameplay results.

## Further reading

- [Experiment guide](../../docs/EXPERIMENT_GUIDE.md)
- [Deep RL and DQN notes](../../docs/DEEP_RL_DQN.md)
- [Project references](../../docs/REFERENCES.md)
