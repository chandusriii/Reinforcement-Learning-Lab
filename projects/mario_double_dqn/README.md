# Super Mario Bros — Double DQN

## Reference
- [Official PyTorch Mario RL tutorial](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
- [gym-super-mario-bros on PyPI](https://pypi.org/project/gym-super-mario-bros/)
- [nes-py on PyPI](https://pypi.org/project/nes-py/)

## Learning architecture
The tutorial demonstrates a Double DQN agent for Super Mario Bros. It preprocesses game frames, stacks observations, stores transitions in replay memory, and uses online and target networks. Double DQN chooses the next action with the online network and evaluates that action with the target network.

## Compatibility warning
The official tutorial uses a legacy Gym-based dependency stack and explicitly warns that Gym is unmaintained and does not support NumPy 2.0. Keep Mario in a separate virtual environment. On Windows, the NES emulator dependency may require Visual Studio C++ Build Tools if a compatible wheel is unavailable.

See [the setup guide](../../docs/SETUP_GUIDE.md) before installation.

## Run plan
1. Resolve and record compatible package versions.
2. Import the environment and verify reset/step.
3. Verify image preprocessing and stacked-state shape.
4. Run a short training smoke test and confirm optimizer updates.
5. Train with periodic checkpoints and structured logs.
6. Evaluate without exploration over multiple episodes.
7. Save reward statistics, checkpoint metadata, and the exact environment configuration.

## Current status
**Not trained end-to-end.** Preliminary CNN output-shape/finite-output testing was reported in the prior session; emulator startup and gameplay training remain unverified.

## Attribution
Follow the PyTorch tutorial's attribution and any upstream project licenses when adapting code. Do not copy ROM files or redistribute copyrighted game assets.