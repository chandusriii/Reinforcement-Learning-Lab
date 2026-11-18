"""Train a Super Mario Bros agent with Double DQN."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from environment import make_mario_env
from mario_dqn import MarioAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Mario Double DQN agent.")
    parser.add_argument("--episodes", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/mario"))
    parser.add_argument("--replay-dir", type=Path, default=Path("replay_memory"))
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--burnin", type=int, default=10_000)
    parser.add_argument("--learn-every", type=int, default=3)
    parser.add_argument("--sync-every", type=int, default=10_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.episodes, args.max_steps, args.replay_capacity, args.batch_size, args.learn_every, args.sync_every) < 1 or args.burnin < 0:
        raise ValueError("Episode, step, buffer, batch and update settings must be positive; burn-in cannot be negative.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.checkpoint_dir / "training_metrics.csv"
    env = make_mario_env(render=args.render)
    agent = MarioAgent(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        replay_dir=args.replay_dir,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        burnin=args.burnin,
        learn_every=args.learn_every,
        sync_every=args.sync_every,
    )
    if args.resume:
        agent.load(args.resume)

    best_return = float("-inf")
    start_time = time.time()
    try:
        with log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow([
                    "episode", "return", "steps", "epsilon", "mean_loss",
                    "mean_q", "flag_reached", "elapsed_seconds"
                ])

            for episode in range(1, args.episodes + 1):
                state, _ = env.reset(seed=args.seed + episode)
                episode_return = 0.0
                losses = []
                q_values = []
                reached_flag = False

                for step in range(1, args.max_steps + 1):
                    action = agent.act(state, explore=True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    reached_flag = bool(info.get("flag_get", False))
                    terminal_for_target = bool(terminated or reached_flag)
                    episode_done = bool(terminated or truncated or reached_flag)

                    agent.remember(
                        state, action, reward, next_state, terminal_for_target
                    )
                    q_value, loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)
                        q_values.append(q_value)

                    episode_return += float(reward)
                    state = next_state
                    if episode_done:
                        break

                elapsed = time.time() - start_time
                mean_loss = float(np.mean(losses)) if losses else 0.0
                mean_q = float(np.mean(q_values)) if q_values else 0.0
                writer.writerow([
                    episode, round(episode_return, 3), step,
                    round(agent.exploration_rate, 6), round(mean_loss, 6),
                    round(mean_q, 6), int(reached_flag), round(elapsed, 2)
                ])
                file.flush()

                if episode_return > best_return:
                    best_return = episode_return
                    agent.save(args.checkpoint_dir / "best_mario_double_dqn.pt")

                if episode == 1 or episode % 10 == 0:
                    print(
                        f"Episode {episode}/{args.episodes} | return={episode_return:.1f} "
                        f"| steps={step} | epsilon={agent.exploration_rate:.3f} "
                        f"| best={best_return:.1f} | loss={mean_loss:.5f}"
                    )

                if episode % 100 == 0:
                    agent.save(args.checkpoint_dir / "latest_mario_double_dqn.pt")

    except KeyboardInterrupt:
        print("Training interrupted. Saving the latest model and metrics.")
    finally:
        agent.save(args.checkpoint_dir / "latest_mario_double_dqn.pt")
        env.close()

    print(f"Training log: {log_path}")
    print(f"Elapsed seconds: {time.time() - start_time:.1f}")


if __name__ == "__main__":
    main()
