"""Evaluate a saved Mario policy with exploration disabled."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from agent import MarioAgent
from mario_env import make_mario_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.csv"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    env = make_mario_env()
    # The evaluation policy needs no replay storage or optimizer updates.
    agent = MarioAgent(
        env.action_space.n,
        args.checkpoint.parent / "evaluation_runtime",
        replay_capacity=1,
        batch_size=1,
        burn_in=10**12,
    )
    agent.load(args.checkpoint)
    returns = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["episode", "return", "steps", "flag_get"])
            writer.writeheader()
            for episode in range(1, args.episodes + 1):
                state, _ = env.reset(seed=args.seed + episode)
                episode_return = 0.0
                flag_get = False
                for step in range(1, 10_001):
                    action = agent.select_action(state, explore=False)
                    state, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward
                    flag_get = bool(info.get("flag_get", False))
                    if args.render:
                        env.render()
                    if terminated or truncated or flag_get:
                        break
                returns.append(episode_return)
                writer.writerow({
                    "episode": episode,
                    "return": round(episode_return, 4),
                    "steps": step,
                    "flag_get": int(flag_get),
                })
                print(
                    f"Evaluation {episode}/{args.episodes} | return={episode_return:.1f} "
                    f"| steps={step} | flag_get={flag_get}"
                )
    finally:
        env.close()
    print(
        f"Mean return: {np.mean(returns):.3f} | std: {np.std(returns):.3f} "
        f"| min: {np.min(returns):.3f} | max: {np.max(returns):.3f}"
    )
    print(f"Saved evaluation CSV: {args.output}")


if __name__ == "__main__":
    main()
