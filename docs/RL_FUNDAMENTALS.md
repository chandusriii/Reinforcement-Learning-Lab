# Reinforcement Learning Fundamentals

These notes organize the concepts covered in the learning material.

## 1. Agent and environment

- **Agent:** the learner or decision-maker.
- **Environment:** the world with which the agent interacts.
- **State (s):** a representation of the current situation.
- **Action (a):** a choice made by the agent.
- **Reward (r):** scalar feedback returned by the environment.
- **Policy (π):** a rule or probability distribution for selecting actions.

The interaction loop is:

1. The agent observes state (s_t).
2. It selects action (a_t) using its policy.
3. The environment returns reward (r_{t+1}) and next state (s_{t+1}).
4. The agent updates its policy or value estimates.

## 2. Return and discount factor

The discounted return from time (t) is:

[
G_t = sum_{k=0}^{infty} gamma^k r_{t+k+1}
]

where (0 leq gamma leq 1) is the discount factor. A smaller value emphasizes immediate rewards; a larger value gives more weight to future rewards.

## 3. Markov Decision Process

An MDP is commonly represented by ((S,A,P,R,gamma)):
- (S): state space
- (A): action space
- (P(s'|s,a)): transition probability
- (R): reward function
- (gamma): discount factor

The Markov property says that the next state depends on the current state and action, rather than the full history, when the state is a sufficient representation.

## 4. State-value and action-value functions

[
V^pi(s) = mathbb{E}_pi[G_t mid S_t=s]
]

[
Q^pi(s,a) = mathbb{E}_pi[G_t mid S_t=s,A_t=a]
]

The value function evaluates a state under a policy. The action-value function evaluates taking an action in a state and then following the policy.

## 5. Bellman optimality equation

For the optimal action-value function:

[
Q^*(s,a) = mathbb{E}left[r_{t+1}+gammamax_{a'}Q^*(s_{t+1},a')mid s,aight]
]

This recursive relationship is the basis of tabular Q-Learning and the DQN target.

## 6. Exploration and exploitation

- **Exploration:** try actions to gather information.
- **Exploitation:** choose the action currently estimated to be best.
- **Epsilon-greedy:** with probability (epsilon), choose a random action; otherwise choose the action with the highest estimated value.

A decaying epsilon schedule starts with more exploration and gradually shifts toward exploitation.

## 7. SARSA

SARSA is an on-policy temporal-difference control method. Its update is:

[
Q(s_t,a_t) leftarrow Q(s_t,a_t) + alphaleft[r_{t+1}+gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)ight]
]

The next action (a_{t+1}) is selected using the current behavior policy.

## 8. Q-Learning

Q-Learning is an off-policy temporal-difference control method:

[
Q(s_t,a_t) leftarrow Q(s_t,a_t) + alphaleft[r_{t+1}+gammamax_a Q(s_{t+1},a)-Q(s_t,a_t)ight]
]

It uses the maximum next-state action value in its target, regardless of which action the behavior policy actually selects next.

## 9. Deep Q-Networks

DQN approximates the action-value function with a neural network (Q(s,a;	heta)). Two stabilizing techniques are central:

- **Experience replay:** store transitions and sample mini-batches to reduce temporal correlation.
- **Target network:** use a delayed copy of the Q-network to compute more stable targets.

A common DQN target is:

[
y = r + gamma(1-d)max_{a'}Q_{	ext{target}}(s',a')
]

where (d) indicates a terminal transition.

## 10. Double DQN distinction

Double DQN separates next-action selection from next-action evaluation:

[
a^* = argmax_{a'}Q_{	ext{online}}(s',a')
]

[
y = r + gamma(1-d)Q_{	ext{target}}(s',a^*)
]

This is different from standard DQN, which takes the maximum action value directly from the target network.

## Learning sequence

1. Understand the MDP and Bellman equations.
2. Implement tabular SARSA and Q-Learning.
3. Compare policies on the same environment, seeds, and evaluation protocol.
4. Learn replay buffers and target networks.
5. Implement DQN on a low-dimensional state environment.
6. Move to image observations and Double DQN for Mario.