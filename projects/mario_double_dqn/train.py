"""Train the Mario Double DQN agent and persist metrics/checkpoints."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import random
import numpy as np
import torch

from agent import MarioAgent
from mario_env import make_mario_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40, help="Training episodes (tutorial demo defaults to 40)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.replay_capacity < 1:
        parser.error("--episodes, --max-steps and --replay-capacity must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = args.run_dir or Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_mario_env()
    agent = MarioAgent(env.action_space.n, run_dir, replay_capacity=args.replay_capacity)
    metrics_path = run_dir / "training_metrics.csv"
    best_reward = -float("inf")

    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["episode", "steps", "return", "epsilon", "mean_loss", "mean_q", "flag_get"],
            )
            writer.writeheader()
            for episode in range(1, args.episodes + 1):
                state, _ = env.reset(seed=args.seed + episode)
                episode_return = 0.0
                losses, q_values = [], []
                flag_get = False
                for step in range(1, args.max_steps + 1):
                    action = agent.select_action(state, explore=True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    flag_get = bool(info.get("flag_get", False))
                    # Do not bootstrap through true terminal states; time-limit truncations can bootstrap.
                    terminal_for_target = bool(terminated or flag_get)
                    agent.remember(state, action, reward, next_state, terminal_for_target)
                    learning_info = agent.learn()
                    if learning_info:
                        losses.append(learning_info["loss"])
                        q_values.append(learning_info["q_mean"])
                    state = next_state
                    episode_return += reward
                    if terminated or truncated or flag_get:
                        break
                row = {
                    "episode": episode,
                    "steps": step,
                    "return": round(episode_return, 4),
                    "epsilon": round(agent.epsilon, 6),
                    "mean_loss": round(float(np.mean(losses)), 6) if losses else "",
                    "mean_q": round(float(np.mean(q_values)), 6) if q_values else "",
                    "flag_get": int(flag_get),
                }
                writer.writerow(row)
                file.flush()
                print(
                    f"Episode {episode}/{args.episodes} | return={episode_return:.1f} "
                    f"| steps={step} | epsilon={agent.epsilon:.3f} | flag_get={flag_get}"
                )
                if episode_return > best_reward:
                    best_reward = episode_return
                    agent.save(run_dir / "best_mario_ddqn.pt")
            agent.save(run_dir / "mario_ddqn_final.pt")
    finally:
        agent.replay.flush()
        env.close()
    print(f"Training finished. Metrics: {metrics_path}")
    print(f"Best episode return: {best_reward:.2f}")


if __name__ == "__main__":
    main()
