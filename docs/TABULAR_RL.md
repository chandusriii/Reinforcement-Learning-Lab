# 02 — Tabular Reinforcement Learning

Tabular methods store value estimates in tables. They are useful for learning RL foundations because individual state and state-action values can be inspected.

## 1. Dynamic Programming (DP)

Dynamic Programming uses a known environment model: transition probabilities and expected rewards. It applies Bellman equations repeatedly.

**Policy evaluation** estimates values for a fixed policy:

`V_(k+1)(s) = sum_a pi(a|s) sum_s' P(s'|s,a)[R(s,a,s') + gamma V_k(s')]`

**Policy improvement** selects actions with the best expected immediate reward plus discounted next-state value.

**Value iteration** combines evaluation and greedy improvement:

`V_(k+1)(s) = max_a sum_s' P(s'|s,a)[R(s,a,s') + gamma V_k(s')]`

DP uses the model directly and is principled, but it can be computationally expensive for large state spaces and requires a model that may not be available.

## 2. Monte Carlo (MC)

Monte Carlo methods learn from complete sampled episodes. They do not require a transition model and do not bootstrap from an estimated next-state value.

For a visited state, calculate the return from the rewards that follow it, then update:

`V(S_t) <- V(S_t) + alpha [G_t - V(S_t)]`

MC is episode-based: the return is available after the episode (or relevant return segment) is complete. Estimates can have high variance because they depend on sampled trajectories.

## 3. Temporal-Difference (TD) learning

TD learning updates from observed rewards and estimated future values. It combines sampling from experience with bootstrapping from current estimates.

TD(0) state-value update:

`V(S_t) <- V(S_t) + alpha [R_(t+1) + gamma V(S_(t+1)) - V(S_t)]`

TD error:

`delta_t = R_(t+1) + gamma V(S_(t+1)) - V(S_t)`

The TD error measures the difference between the current estimate and a one-step target. TD updates step by step rather than waiting for a full episode. For a true terminal transition, the bootstrap term is zero.

## 4. SARSA — On-policy TD control

SARSA is named for the sequence **State → Action → Reward → Next State → Next Action**.

1. Observe current state S_t.
2. Choose A_t using the behaviour policy, often epsilon-greedy.
3. Execute A_t and receive reward R_(t+1) and next state S_(t+1).
4. Choose next action A_(t+1) using the same policy.
5. Update Q(S_t,A_t) using the action actually selected.

`Q(S_t,A_t) <- Q(S_t,A_t) + alpha [R_(t+1) + gamma Q(S_(t+1),A_(t+1)) - Q(S_t,A_t)]`

For terminal transitions, the target is only the immediate reward. SARSA is on-policy because its update uses the next action selected by the policy generating behaviour.

## 5. Q-Learning — Off-policy TD control

Q-Learning uses the maximum estimated next-state action value, regardless of which next action the behaviour policy actually takes:

`Q(S_t,A_t) <- Q(S_t,A_t) + alpha [R_(t+1) + gamma max_a Q(S_(t+1),a) - Q(S_t,A_t)]`

For terminal transitions, the target is only the immediate reward. Q-Learning is off-policy because behaviour may be epsilon-greedy while the update learns toward a greedy target policy.

## 6. SARSA vs Q-Learning

| Property | SARSA | Q-Learning |
|---|---|---|
| Type | TD control | TD control |
| Policy relationship | On-policy | Off-policy |
| Next-state term | Q(S_(t+1), A_(t+1)) | max_a Q(S_(t+1),a) |
| Uses sampled next action? | Yes | No |
| Learns toward | Behaviour policy | Greedy target policy |

In Cliff Walking, an exploratory action can move the agent close to the cliff. SARSA accounts for the next action actually selected, while Q-Learning's target assumes the best next action. This can produce different routes and risk profiles.

## 7. Cliff Walking

The supplied notes use a 4 × 12 grid:
- Start at the lower-left.
- Goal at the lower-right.
- Cliff cells occupy the bottom row between start and goal.
- Ordinary moves have a small negative step reward.
- Falling from the cliff gives a large negative reward and returns the agent to the start.

This illustrates that maximizing expected return is not always equivalent to choosing the shortest-looking path. The learned route depends on algorithm, exploration, and reward design.

A 4 × 12 grid has 48 grid positions. Actions are typically up, right, down, and left; verify the actual environment's action mapping rather than assuming an ordering.

## 8. Gymnasium interaction pattern

```python
observation, info = env.reset(seed=seed)

while True:
    action = agent.choose_action(observation)
    next_observation, reward, terminated, truncated, info = env.step(action)
    agent.learn(observation, action, reward, next_observation, terminated)
    observation = next_observation

    if terminated or truncated:
        break

env.close()
```

- reset() returns (observation, info).
- step(action) returns (observation, reward, terminated, truncated, info).
- terminated indicates a task-defined terminal state.
- truncated indicates an external cutoff, such as a time limit.
- Close the environment after use.

For bootstrapping methods, do not bootstrap through a true terminal state. Truncation is conceptually different; whether to bootstrap at truncation depends on the task and implementation.

## 9. Hyperparameters

| Parameter | Meaning |
|---|---|
| alpha | Learning rate; controls update size |
| gamma | Discount factor; controls future reward weight |
| epsilon | Probability of random exploratory action |
| Episodes | Number of complete attempts |
| Max steps | Per-episode safety/time limit |

Treat hyperparameters as experimental choices, not universal defaults. Record them with each run.

## 10. Experiment checklist

For SARSA and Q-Learning, use the same environment and comparable training budgets. Record episode return, episode length, moving-average return, greedy evaluation return, cliff falls (if available), learning rate, discount factor, epsilon schedule, and random seed.

One run is not enough to establish general superiority. Use multiple seeds for a more reliable comparison.

## Source alignment

Expanded from the supplied **Reinforcement_Learning part2.pdf**, especially its DP, Monte Carlo, TD, SARSA, Q-Learning, Cliff Walking, Gymnasium, and implementation slides. The original SARSA and Q-Learning notebooks should be preserved and reviewed alongside these explanations when added to this repository.
