"""Evaluate a saved Mario policy without exploration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from environment import make_mario_env
from mario_dqn import MarioAgent


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mario Double DQN.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("evaluation/evaluation.csv"))
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("Episodes and max steps must be positive.")

    env = make_mario_env(render=args.render)
    agent = MarioAgent(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        replay_dir=args.output.parent / "evaluation_replay",
        replay_capacity=2,
        batch_size=2,
    )
    agent.load(args.checkpoint)
    agent.exploration_rate = 0.0
    returns = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "return", "steps", "flag_reached"])
            for episode in range(1, args.episodes + 1):
                state, _ = env.reset(seed=args.seed + episode)
                total_return = 0.0
                reached_flag = False
                for step in range(1, args.max_steps + 1):
                    action = agent.act(state, explore=False)
                    state, reward, terminated, truncated, info = env.step(action)
                    total_return += float(reward)
                    reached_flag = bool(info.get("flag_get", False))
                    if terminated or truncated or reached_flag:
                        break
                returns.append(total_return)
                writer.writerow([episode, round(total_return, 3), step, int(reached_flag)])
                print(
                    f"Evaluation {episode}/{args.episodes}: "
                    f"return={total_return:.1f}, steps={step}, flag={reached_flag}"
                )
    finally:
        env.close()

    print(f"Mean return: {np.mean(returns):.3f}")
    print(f"Std return: {np.std(returns):.3f}")
    print(f"Evaluation CSV: {args.output}")


if __name__ == "__main__":
    main()
