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
    parser.add_argument("--episodes", type=int, default=40_000,
                        help="Number of new episodes to run (additional episodes when resuming).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/mario"))
    parser.add_argument("--replay-dir", type=Path, default=Path("replay_memory"))
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--burnin", type=int, default=10_000)
    parser.add_argument("--learn-every", type=int, default=3)
    parser.add_argument("--sync-every", type=int, default=10_000)
    parser.add_argument("--epsilon-decay", type=float, default=0.99999975)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from a matching checkpoint; use the same replay directory.")
    return parser.parse_args()


def _read_and_reconcile_metrics(path: Path, saved_episode: int) -> tuple[float, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot resume without the training metrics CSV: {path}"
        )
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if not fieldnames or "episode" not in fieldnames or "return" not in fieldnames:
            raise ValueError(f"Training metrics file has an invalid header: {path}")
        rows = [row for row in reader if int(row["episode"]) <= saved_episode]

    if saved_episode > 0 and not any(int(row["episode"]) == saved_episode for row in rows):
        raise ValueError(
            "Checkpoint episode is not present in training_metrics.csv; "
            "use the matching checkpoint and metrics file."
        )

    # A process can stop after writing metrics but before the next checkpoint.
    # Discard rows newer than the checkpoint so episode numbers never skip ahead.
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

    best_return = max((float(row["return"]) for row in rows), default=float("-inf"))
    return best_return, fieldnames


def main():
    args = parse_args()
    positive = (
        args.episodes, args.max_steps, args.replay_capacity,
        args.batch_size, args.learn_every, args.sync_every,
    )
    if min(positive) < 1 or args.burnin < 0:
        raise ValueError(
            "Episode, step, buffer, batch and update settings must be positive; "
            "burn-in cannot be negative."
        )
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")

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
        epsilon_decay=args.epsilon_decay,
        resume_replay=args.resume is not None,
    )

    if args.resume is not None:
        saved_episode = agent.load(args.resume)
        if saved_episode < 1:
            raise ValueError("The supplied checkpoint has no completed-episode marker.")
        best_return, fieldnames = _read_and_reconcile_metrics(log_path, saved_episode)
        start_episode = saved_episode + 1
        file_mode = "a"
    else:
        best_return = float("-inf")
        start_episode = 1
        file_mode = "w"
        fieldnames = [
            "episode", "return", "steps", "epsilon", "mean_loss",
            "mean_q", "flag_reached", "elapsed_seconds"
        ]

    if start_episode < 1:
        raise ValueError("Invalid starting episode.")

    start_time = time.time()
    last_completed_episode = start_episode - 1
    try:
        with log_path.open(file_mode, newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if file_mode == "w":
                writer.writerow(fieldnames)

            for episode in range(start_episode, start_episode + args.episodes):
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
                last_completed_episode = episode

                if episode_return > best_return:
                    best_return = episode_return
                    agent.save(
                        args.checkpoint_dir / "best_mario_double_dqn.pt",
                        episode=episode,
                    )
                    # Keep latest and replay metadata aligned with the same save point.
                    agent.save(
                        args.checkpoint_dir / "latest_mario_double_dqn.pt",
                        episode=episode,
                    )

                if episode == start_episode or episode % 10 == 0:
                    print(
                        f"Episode {episode} | return={episode_return:.1f} "
                        f"| steps={step} | epsilon={agent.exploration_rate:.3f} "
                        f"| best={best_return:.1f} | loss={mean_loss:.5f}"
                    )

                if episode % 100 == 0:
                    agent.save(
                        args.checkpoint_dir / "latest_mario_double_dqn.pt",
                        episode=episode,
                    )

    except KeyboardInterrupt:
        print("Training interrupted. Saving the latest completed episode.")
    finally:
        agent.save(
            args.checkpoint_dir / "latest_mario_double_dqn.pt",
            episode=last_completed_episode,
        )
        env.close()

    print(f"Training log: {log_path}")
    print(f"Last completed episode: {last_completed_episode}")
    print(f"Elapsed seconds in this session: {time.time() - start_time:.1f}")


if __name__ == "__main__":
    main()
