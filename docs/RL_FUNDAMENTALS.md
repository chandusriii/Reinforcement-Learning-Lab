# 01 — Reinforcement Learning Fundamentals

## What is Reinforcement Learning?

Reinforcement Learning (RL) is a machine-learning approach in which an **agent learns to make decisions by interacting with an environment**. At each time step, the agent observes a state, chooses an action, receives a reward, and observes the next situation.

Unlike supervised learning, the agent is not given the correct action for every situation. It discovers useful behaviour through experience and feedback. The objective is to learn a policy that maximizes expected cumulative reward.

## The interaction loop

1. Environment provides the current state, S_t.
2. Agent selects action A_t using its policy.
3. Environment transitions to S_(t+1).
4. Environment returns reward R_(t+1).
5. Agent uses the transition to improve its policy or value estimates.
6. The process continues until the episode ends.

A transition is commonly represented as (state, action, reward, next_state, terminated).

## Core components

| Component | Meaning | Example |
|---|---|---|
| Agent | Learner or decision-maker | Flappy Bird controller |
| Environment | World being interacted with | Flappy Bird game |
| State | Information describing the situation | Bird position, velocity, pipe distances |
| Action | Choice made by the agent | Flap or do nothing |
| Reward | Scalar feedback | Progress reward or collision penalty |
| Policy | Rule for choosing actions | Epsilon-greedy policy |
| Episode | One sequence from reset to end | One game attempt |

An observation is the information returned to the agent. In a fully observable problem it can represent the full state; in a partially observable problem it may reveal only part of the underlying state.

## Rewards and returns

A reward is immediate feedback. Return measures accumulated reward over time:

`G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots`

Equivalently, G_t = R_(t+1) + gamma G_(t+1).

- G_t: return from time t.
- R_(t+1): next reward.
- gamma: discount factor, 0 <= gamma <= 1.

Gamma = 0 makes the agent care only about immediate reward. A value close to 1 gives future rewards more influence. Reward design matters: an agent optimizes the reward it receives, which may differ from the real-world goal if the reward is poorly designed.

## Policy

A policy defines how actions are selected.
- Deterministic: a = pi(s).
- Stochastic: pi(a|s) defines a probability distribution over actions.

A policy may be fixed or improved during learning.

## Markov Decision Process (MDP)

An MDP is a framework for sequential decision-making, commonly represented by (S, A, P, R, gamma):
- S: state set.
- A: action set.
- P(s'|s,a): probability of next state s' after action a in state s.
- R(s,a,s'): expected immediate reward for the transition.
- gamma: discount factor.

### Markov property

The Markov property states that the next state depends on the current state and action, not on the full history:

`P(S_(t+1) | S_t, A_t, history) = P(S_(t+1) | S_t, A_t)`

The state should contain enough information for this assumption to be reasonable.

### Grid-world intuition

An agent moves through a grid. Its cell is the state; up, down, left, and right are actions. Reaching a goal may give a positive reward, a cliff may give a large penalty, and each move may have a small cost. The best route depends on cumulative return, not only the next move.

## Value functions

The state-value function estimates expected return from a state under policy pi:

`V^pi(s) = E_pi[G_t | S_t = s]`

The action-value function estimates expected return after taking action a in state s and then following pi:

`Q^pi(s,a) = E_pi[G_t | S_t=s, A_t=a]`

Q-values allow the agent to compare actions. A greedy policy selects the action with the largest estimated Q-value.

## Bellman equations

Bellman equations express value recursively as immediate reward plus discounted future value.

For a policy:

`V^pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a) [R(s,a,s') + gamma V^pi(s')]`

For the optimal state value:

`V*(s) = max_a sum_s' P(s'|s,a) [R(s,a,s') + gamma V*(s')]`

The Bellman idea underlies Dynamic Programming, Temporal-Difference learning, SARSA, Q-Learning, and DQN.

## Exploration and exploitation

- **Exploration:** try actions to gather information and discover better choices.
- **Exploitation:** choose the action currently believed to be best.

### Epsilon-greedy

With probability epsilon, choose a random action. With probability 1-epsilon, choose argmax_a Q(s,a). Epsilon is often reduced over training: more exploration early, more exploitation later. Too little exploration can trap an agent in a poor policy; too much can prevent consistent use of learned information.

## RL method map

| Method | Main idea | Model required? | Update timing |
|---|---|---:|---|
| Dynamic Programming | Bellman backups with known transitions | Yes | Repeated planning sweeps |
| Monte Carlo | Learn from sampled episode returns | No | After return is available |
| Temporal Difference | Reward plus estimated next value | No | Step by step |
| SARSA | On-policy TD control | No | Uses sampled next action |
| Q-Learning | Off-policy TD control | No | Uses maximum next action value |
| DQN | Neural-network approximation of Q-values | No | Mini-batch gradient updates |

## Essential vocabulary

- Model-based RL: uses or learns environment dynamics.
- Model-free RL: learns values or policies directly from experience.
- On-policy: learns about the policy generating behaviour.
- Off-policy: learns about a target policy that can differ from the behaviour policy.
- Terminal state: task-defined end of an episode.
- Truncation: externally imposed cutoff, such as a time limit.
- Learning rate (alpha): controls update size.
- Discount factor (gamma): controls future reward weighting.

## Check your understanding

1. How is reward different from return?
2. Why are exploration and exploitation both necessary?
3. What should a state contain for the Markov property to be useful?
4. How do V(s) and Q(s,a) differ?
5. How can reward design change learned behaviour?

## Source alignment

Organized from the supplied **Deep_Learning-42 3.pdf**, covering the agent-environment interaction, components, MDP, grid example, and exploration–exploitation. Explanatory notation and examples are included to make the concepts easier to study and implement.
