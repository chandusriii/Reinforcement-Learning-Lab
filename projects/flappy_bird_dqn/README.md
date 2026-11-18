# Flappy Bird AI Agent Using Deep Q-Networks (DQN)

## Project Overview

This project focuses on building a reinforcement learning agent that learns to play Flappy Bird through interaction with the game environment. The agent observes the current game state, selects an action, receives a reward or penalty, and gradually learns a policy that improves its gameplay.

The implementation uses a Deep Q-Network (DQN) to estimate the value of the available actions.

## Model Architecture

The state-based neural network is designed with:

- **Input layer:** 12 state features from the game environment.
- **Hidden layer:** 256 neurons with ReLU activation.
- **Output layer:** 2 Q-values, corresponding to the available actions (no-op and flap).

## Key Reinforcement Learning Components

- **Experience replay:** Stores previous transitions and samples batches for training.
- **Target network:** Provides more stable target-value estimates during learning.
- **Epsilon-greedy policy:** Balances exploration of new actions with exploitation of learned action values.
- **Reward-based learning:** Uses feedback from the environment to update the agent's action-value estimates.

## Technology Stack

- Python
- PyTorch
- Gymnasium
- flappy-bird-gymnasium
- NumPy

## Project Workflow

1. Initialize the Flappy Bird environment.
2. Observe the 12-feature state representation.
3. Select an action using an epsilon-greedy policy.
4. Execute the action and collect the reward and next state.
5. Store the transition in the replay buffer.
6. Sample experiences and update the online Q-network.
7. Synchronize the target network at the configured intervals.
8. Evaluate the trained agent using a greedy policy.

## Running the Project

Follow the [Windows setup guide](../../docs/SETUP_GUIDE.md) to create the environment and install the dependencies. Training and evaluation commands will be documented here after the runnable training pipeline is integrated and verified.

## Experiment Status

**Implementation documentation and architecture review are in place. End-to-end game training and evaluation are still pending.**

No training reward curve, trained checkpoint, or gameplay performance result is claimed yet. These will be added after the training pipeline has been executed and its results recorded.

## Learning Outcomes

- Understanding how DQN applies neural networks to action-value estimation.
- Working with state, action, reward, and next-state transitions.
- Understanding experience replay and target networks.
- Applying exploration-versus-exploitation strategies.
- Evaluating reinforcement learning agents using recorded episode rewards.

## Attribution

This project uses the implementation assigned for the coursework as its starting codebase. Original authorship and applicable license notices should be retained when redistributing or modifying upstream source files.
