# 03 — Deep Reinforcement Learning and DQN

## 1. Why Deep RL?

Tabular methods store a separate value for each state or state-action pair. This becomes impractical when the state space is large, continuous, or based on high-dimensional observations such as images.

Deep Reinforcement Learning combines RL with neural networks. A network can approximate a value function or policy from state features or observations. This section focuses on value-based methods and Deep Q-Networks.

## 2. Neural Q-function

A Q-network receives a state representation and outputs an estimated value for every available action:

`Q(s, ·; theta) = [Q(s,a_1;theta), ..., Q(s,a_n;theta)]`

- Input: state features or processed observation.
- Hidden layers: learn useful representations.
- Output: one Q-value per discrete action.
- theta: trainable network parameters.

The greedy action is argmax_a Q(s,a;theta). The network estimates action values; it does not directly output action probabilities.

## 3. Deep Q-Network (DQN)

DQN approximates the optimal action-value function using a neural network. A one-step target is:

`y = r + gamma max_(a') Q_target(s',a')`

A common squared TD loss is:

`L(theta) = E[(y - Q_online(s,a;theta))^2]`

For true terminal transitions, remove the discounted next-state term. Update online network parameters by minimizing the loss with gradient-based optimization.

## 4. Why DQN uses stabilizing techniques

Consecutive game observations are correlated. Training on adjacent correlated samples can make optimization unstable. The target can also move as the same network changes. Classic DQN addresses these issues with experience replay and a separate target network; epsilon-greedy supports exploration.

## 5. Experience replay

A replay buffer stores transitions such as (state, action, reward, next_state, done).

1. Collect a transition from the environment.
2. Store it in replay memory.
3. Sample a mini-batch randomly.
4. Compute target Q-values.
5. Calculate loss and update the online network.

Random mini-batches reduce dependence on the order of consecutive experiences and allow old experiences to be reused. A buffer commonly has a fixed capacity, replacing older entries when full.

## 6. Target network

DQN maintains:
- **Online network:** updated by gradient descent.
- **Target network:** used to compute target values and updated less frequently.

A hard synchronization copies online parameters to target parameters:

`theta_target <- theta_online`

This reduces how rapidly target values change during optimization.

## 7. Epsilon-greedy exploration

With probability epsilon, select a random action. Otherwise select the action with the highest online Q-value. Epsilon is often decayed over training subject to a minimum. During evaluation, use a greedy policy (typically epsilon = 0) unless the evaluation protocol says otherwise.

## 8. DQN training workflow

1. Reset environment and receive initial observation.
2. Select action using epsilon-greedy.
3. Step environment.
4. Store transition in replay memory.
5. Sample mini-batch once enough transitions exist.
6. Compute predicted Q-values for selected actions.
7. Compute target values using target network.
8. Calculate loss and update online network.
9. Synchronize target network at configured intervals.
10. Track episode return, length, and exploration rate.
11. Repeat and save checkpoints.

## 9. Standard DQN vs Double DQN

**Standard DQN:** the target network supplies the maximum next-state value:

`y = r + gamma max_(a') Q_target(s',a')`

**Double DQN:** online network selects the next action; target network evaluates it:

`a* = argmax_(a') Q_online(s',a')`

`y = r + gamma Q_target(s',a*)`

Separating selection and evaluation is designed to reduce overestimation from using the same estimates for both operations.

## 10. Flappy Bird DQN project

The assigned Flappy Bird implementation uses a state-based neural network:
- 12 input features.
- One hidden layer with 256 units and ReLU.
- 2 output action values: no-op and flap.
- Experience replay.
- Online and target networks.
- Epsilon-greedy action selection.
- Adam optimizer and squared-error loss.

**Algorithm detail:** the inspected target calculation takes the maximum output of the target network for the next state. This is standard DQN, not the Double DQN selection/evaluation split.

The supplied FlappyBird_DQN_Code.pdf points to the assigned implementation and notes that a runs folder is created by executing the code covered in the lecture. Preserve applicable upstream license and attribution notices when using its source.

## 11. Mario Double DQN project

The supplied assignment asks for a Mario-playing agent using Double DQN and points to the PyTorch Mario RL tutorial. The tutorial workflow includes:
- Frame preprocessing, skipping, resizing, and stacking.
- A convolutional neural network for visual observations.
- Replay memory.
- Epsilon-greedy action selection.
- Online and target networks.
- Double DQN action selection and evaluation.
- Training, logging, and checkpointing.

The tutorial's original environment stack uses legacy Gym ecosystem packages. Verify dependency and operating-system compatibility before treating it as a reproducible installation.

## 12. Implementation considerations

- Distinguish task termination from time-limit truncation.
- Do not bootstrap targets through true terminal states.
- Use identical preprocessing during training and evaluation.
- Check observation shape and action-space size before training.
- Set and record seeds where supported.
- Keep training and evaluation episodes separate.
- Save model weights and the configuration needed to interpret them.
- Track raw episode returns and moving averages.
- Never report a trained result until the environment and training loop have actually run.

## 13. Evaluation

Evaluate separately from exploratory training. Record evaluation episode count, mean and standard deviation of return, episode length, success rate or task score, evaluation exploration setting, checkpoint identifier, and environment/wrapper versions.

A reward curve helps reveal learning trends but does not by itself prove generalization. Evaluate across multiple seeds or held-out conditions when feasible.

## Source alignment

Based on the supplied Flappy Bird DQN code reference, Mario Double DQN assignment, and Deep RL concepts from the course materials. These notes explain implementation patterns without claiming either game has been trained end-to-end in this repository.
