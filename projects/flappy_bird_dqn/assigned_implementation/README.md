# Assigned Flappy Bird DQN implementation

This directory preserves the source files supplied for the Flappy Bird DQN coursework implementation.

**Source repository:** https://github.com/shradha-khapra/RL_DQN  
**Imported branch:** `main`  
**Imported source files:** `.gitignore`, `README.md`, `agent.py`, `dqn.py`, `experience_replay.py`, `game_flappy_bird.py`, and `parameters.yaml`.

The assigned source files are kept distinct from experiment adaptations. The original `agent.py` was not changed. The additional `agent_eval.py` is a separate evaluation copy that loads the saved checkpoint, runs 100 episodes without exploration, prints summary statistics, and writes per-episode rewards to a CSV file.

The source repository did not expose a license file in its root listing when checked; do not assume a license grant. Confirm the author's permission or applicable license before redistributing copied upstream code.

## Run commands

Run these commands from this directory after installing compatible dependencies.

Train the assigned implementation:

```cmd
python agent.py flappybirdv0 --train
```

Evaluate the saved checkpoint for 100 episodes:

```cmd
python agent_eval.py flappybirdv0
```

The evaluation script expects the checkpoint at `runs/flappybirdv0.pt`. It saves the per-episode evaluation data to `runs/flappybirdv0_evaluation.csv`.

Run the original agent in rendered mode only if you have separately addressed the epsilon-print issue in a local copy. The original assigned `agent.py` contains a print statement that references `epsilon` even when training is disabled; the evaluation copy handles this without modifying the assigned source.

Manual keyboard play:

```cmd
python game_flappy_bird.py
```

The source creates the `runs/` directory and saves its best checkpoint and best-reward log there. The imported `.gitignore` excludes `runs/`; copy selected result artifacts to the project's tracked `results/` directory if they should be committed.

## Execution notes

The source requires Python packages including PyTorch, Gymnasium, flappy-bird-gymnasium, PyYAML, and Pygame. The recorded local experiment used Python 3.11.9 and CPU-only PyTorch. See the [project experiment report](../EXPERIMENT_RESULTS.md) and [Windows setup guide](../../../docs/SETUP_GUIDE.md) for recorded versions and limitations.
