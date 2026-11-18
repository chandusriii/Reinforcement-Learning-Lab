# Super Mario Bros — Double Deep Q-Network (Double DQN)

## Assignment and objective

The supplied assignment asks for a Mario-playing reinforcement-learning agent using **Double Deep Q-Networks (Double DQN)** and explicitly directs the student to complete the official PyTorch tutorial. The objective is to train an agent that learns to play through interaction with the game environment.

This repository treats the PyTorch tutorial as the implementation guide and the Double DQN research paper as the algorithmic foundation. The project is documented and scoped; emulator training and evaluation are not yet claimed as completed.

## Primary sources

1. **Implementation documentation:** [Train a Mario-playing RL Agent — PyTorch Tutorials](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html). The tutorial covers visual-observation preprocessing, frame skipping and stacking, convolutional Q-networks, replay memory, epsilon-greedy action selection, online and target networks, and training.
2. **Research paper:** Hado van Hasselt, Arthur Guez, and David Silver, [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461), AAAI 2016, DOI: [10.1609/AAAI.V30I1.10295](https://doi.org/10.1609/AAAI.V30I1.10295). The paper motivates Double DQN as a way to reduce the overestimation of action values associated with standard Q-learning/DQN.

## What the agent learns

The Mario environment produces image observations. These are transformed into a compact, consistent representation before being passed to a convolutional neural network. Recent frames are stacked to provide temporal context.

The agent maintains an online action-value network and a target network. It selects actions with an epsilon-greedy policy during training, stores transitions in replay memory, samples mini-batches, and updates the online network using a Double DQN target.

### Double DQN target

The online network selects the next action:  a* = argmax_a Q_online(s', a; theta)

The target network evaluates the selected action:  y = r + gamma * (1 - done) * Q_target(s', a*; theta_target)

For a true terminal transition, the bootstrap term is removed. Action selection and action evaluation are separated to reduce overestimation.

## Implementation workflow

1. Create and validate the Mario environment in an isolated Python environment.
2. Apply the tutorial's observation wrappers and confirm the final observation shape and action space.
3. Build the convolutional online and target Q-networks.
4. Implement epsilon-greedy action selection.
5. Store state, action, reward, next state, and termination information in replay memory.
6. Sample mini-batches and compute the Double DQN target.
7. Optimize the online network and periodically synchronize the target network.
8. Record episode return, episode length, exploration rate, loss, and environment steps.
9. Save checkpoints and training logs.
10. Evaluate the saved policy separately from exploratory training.

## Technology

- Python
- PyTorch and TorchVision
- Gym-based Super Mario environment
- NumPy
- Convolutional neural networks
- Experience replay and target networks
- Double DQN

## Environment and reproducibility notes

The PyTorch tutorial uses an older Gym ecosystem and pinned versions of supporting packages. Do not install those dependencies into the Flappy Bird environment. Use a dedicated virtual environment and verify compatibility among Python, Gym/Gymnasium, gym-super-mario-bros, nes-py, NumPy, PyTorch, TorchVision, TensorDict, and TorchRL before training.

First validate that the emulator imports, the environment resets, one action can be stepped, and the returned observation matches the model input. A successful import or network forward pass alone is not evidence that the game agent has learned.

Do not redistribute ROM files or copyrighted game assets.

## Experiment status

**Current status: documentation and implementation plan prepared; gameplay training and evaluation pending.**

No reward curve, success-rate claim, trained checkpoint, or gameplay-performance result is reported here. Once implementation is complete, record Python and dependency versions, operating system, compute device, environment ID, wrappers, observation shape, preprocessing, random seed, optimizer, learning rate, discount factor, epsilon schedule, replay capacity, batch size, target update schedule, training returns, episode lengths, loss, checkpoint path, and separate evaluation metrics.

## Related repository material

- [Deep RL and DQN notes](../../docs/DEEP_RL_DQN.md)
- [Setup guide](../../docs/SETUP_GUIDE.md)
- [Experiment guide](../../docs/EXPERIMENT_GUIDE.md)
- [Execution status](../../docs/EXECUTION_STATUS.md)

## Attribution

The implementation is based on the supplied coursework assignment and the PyTorch tutorial linked above. Cite the Double DQN paper when describing the algorithm, and retain applicable upstream notices when adapting tutorial code.
