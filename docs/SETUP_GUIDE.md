# Windows Setup Guide

Use separate virtual environments for the two game projects. Python 3.11 is a sensible first interpreter to try for the Flappy Bird Gymnasium project. Mario's reference tutorial uses older Gym-based packages, so dependency compatibility must be checked independently.

## 1. Flappy Bird environment

Open PowerShell in the repository:

```powershell
py -3.11 -m venv .venv-flappy
.\.venv-flappy\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install "numpy<2" gymnasium pygame matplotlib flappy-bird-gymnasium
```

The package documentation identifies the environment ID as `FlappyBird-v0`. Confirm the installed package's exact API before running the reference agent.

Smoke test:

```powershell
python -c "import gymnasium, flappy_bird_gymnasium; env=gymnasium.make('FlappyBird-v0'); obs,info=env.reset(); print('observation:', getattr(obs,'shape',None), 'info:', info); action=env.action_space.sample(); result=env.step(action); print('step tuple length:', len(result)); env.close()"
```

If the environment ID or observation format differs from the code you are adapting, stop and inspect the installed package documentation before changing the model input size.

## 2. Mario environment

The official PyTorch tutorial currently documents the legacy package combination below:

```powershell
py -3.11 -m venv .venv-mario
.\.venv-mario\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "numpy<2" torch torchvision
python -m pip install gym==0.26.2 gym-super-mario-bros==7.4.0 nes-py tensordict==0.3.0 torchrl==0.3.0
```

This is a starting compatibility attempt, **not a guaranteed working lockfile**. The official tutorial was last verified in 2024 and itself warns that Gym is unmaintained and does not support NumPy 2.0. On Windows, `nes-py` may require Visual Studio C++ Build Tools if a compatible wheel is unavailable. Do not mix the Mario dependencies into the Flappy Bird environment.

If installation fails, capture the complete error and package versions. Do not randomly upgrade Gym, NumPy, TorchRL, or nes-py together; that can make the environment harder to reproduce.

## 3. Record the environment

Run this inside each activated virtual environment:

```powershell
python --version
python -m pip freeze > requirements-installed.txt
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## 4. Training safety

Start with a short smoke run (for example, a few episodes) before a long training run. Verify that:
- the environment resets and steps without exceptions;
- observations have the expected shape and dtype;
- rewards and termination flags are finite/valid;
- the replay buffer receives transitions;
- a gradient update occurs;
- checkpoints can be saved and loaded.

Do not describe a smoke test as successful learning. Evaluate a trained model with exploration disabled and report multiple episodes.

## References
- [Flappy Bird Gymnasium package](https://pypi.org/project/flappy-bird-gymnasium/)
- [Gymnasium third-party environments](https://gymnasium.farama.org/v1.2.3/environments/third_party_environments/)
- [Official PyTorch Mario tutorial](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
- [nes-py package and Windows build notes](https://pypi.org/project/nes-py/)