# Flappy Bird — DQN

## Reference
- Upstream implementation reviewed: [shradha-khapra/RL_DQN](https://github.com/shradha-khapra/RL_DQN)
- Environment: [flappy-bird-gymnasium](https://pypi.org/project/flappy-bird-gymnasium/)
- Environment catalogue: [Gymnasium third-party environments](https://gymnasium.farama.org/v1.2.3/environments/third_party_environments/)

## Reviewed model design
The reference agent's state-based network uses 12 input features, one 256-unit ReLU hidden layer, and two output action values. The action space represents no-op and flap. The reference design uses epsilon-greedy action selection, replay memory, a target network, Adam, and a squared-error loss.

The inspected target calculation takes `max` over the target network's next-state outputs. That is standard DQN, not Double DQN.

## Run plan
1. Install the isolated dependencies using [the setup guide](../../docs/SETUP_GUIDE.md).
2. Verify the environment observation shape and action space.
3. Check that the reference network input dimension matches the environment observation.
4. Run a short training smoke test.
5. Train with periodic checkpointing.
6. Evaluate with epsilon set to zero over at least 20 episodes.
7. Save episode returns, moving-average returns, training configuration, and checkpoint metadata.

## Current status
**Not trained end-to-end.** Preliminary network shape/finite-output testing was reported in the prior session, but the game environment and training loop were not run in the assistant's isolated environment.

## Attribution
This project is based on learning from the upstream reference. Before copying or redistributing source files, review and preserve the upstream repository's license and notices.