# Assigned Flappy Bird DQN implementation

This directory preserves the source files supplied for the Flappy Bird DQN coursework implementation.

**Source repository:** https://github.com/shradha-khapra/RL_DQN  
**Imported branch:** `main`  
**Imported files:** `.gitignore`, `README.md`, `agent.py`, `dqn.py`, `experience_replay.py`, `game_flappy_bird.py`, and `parameters.yaml`.

These files have been copied as provided. The source repository did not expose a license file in its root listing when checked; do not assume a license grant. Confirm the author's permission or applicable license before redistributing the copied code publicly. This import is kept in a clearly named subdirectory so the assigned source remains distinct from our notes, experiments, and any future modifications.

## Run commands

From this directory, after installing compatible dependencies:

```powershell
python agent.py flappybirdv0 --train
```

To load the saved model for rendered play:

```powershell
python agent.py flappybirdv0
```

Manual keyboard play:

```powershell
python game_flappy_bird.py
```

The source creates the `runs/` directory and saves its best checkpoint and a log there.

## Execution notes

The source requires Python packages including PyTorch, Gymnasium, flappy-bird-gymnasium, PyYAML, and Pygame. Actual environment training must be run in a compatible local Python environment. Record the Python/package versions and measured run outputs before claiming training performance.
