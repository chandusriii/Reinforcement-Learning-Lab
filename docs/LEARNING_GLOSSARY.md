# Reinforcement Learning — Quick Glossary

| Term | Meaning |
|---|---|
| Agent | Learner or decision-maker |
| Environment | World the agent interacts with |
| State | Information describing the current situation |
| Observation | Information returned to the agent |
| Action | A decision available to the agent |
| Reward | Immediate scalar feedback |
| Return | Accumulated future reward, often discounted |
| Policy | Rule or distribution used to choose actions |
| Episode | Sequence of interaction ending at termination or truncation |
| MDP | Mathematical framework for sequential decision-making |
| Markov property | Next state depends on current state and action, not full history |
| Value function | Expected return from a state |
| Q-function | Expected return from a state-action pair |
| Bellman equation | Recursive relationship defining value |
| Exploration | Trying actions to gather information |
| Exploitation | Choosing actions currently estimated to be best |
| Dynamic Programming | Planning with a known environment model |
| Monte Carlo | Learning from sampled episode returns |
| Temporal Difference | Learning from reward plus an estimated next value |
| Bootstrapping | Updating an estimate using another estimate |
| SARSA | On-policy TD control algorithm |
| Q-Learning | Off-policy TD control algorithm |
| DQN | Q-learning with a neural-network function approximator |
| Replay buffer | Stored transitions sampled during training |
| Target network | Slowly updated network used to calculate DQN targets |
| Double DQN | Online network selects action; target network evaluates it |
| Learning rate | Step size used in value or parameter updates |
| Discount factor | Weight applied to future rewards |
| Epsilon-greedy | Random action with probability epsilon, greedy otherwise |
| Checkpoint | Saved model parameters and metadata |
| Seed | Value used to control randomness |
| Evaluation | Measuring an agent separately from training updates |
| Reward curve | Plot of episode reward over training |
