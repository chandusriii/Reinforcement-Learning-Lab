"""Reproducible multi-seed comparison of SARSA and Q-Learning on CliffWalking-v1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import gymnasium as gym
import numpy as np
import gymnasium


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    best = np.flatnonzero(q_values == np.max(q_values))
    return int(rng.choice(best))


def train_one(
    algorithm: str,
    seed: int,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    max_episode_steps: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    env = gym.make("CliffWalking-v1")
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    rows: list[dict[str, float]] = []

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=seed + episode)
            action = epsilon_greedy(q_table[state], epsilon, rng)
            total_reward = 0.0
            steps = 0

            while True:
                next_state, reward, terminated, truncated, _ = env.step(action)
                steps += 1
                episode_truncated = bool(truncated or steps >= max_episode_steps)
                episode_done = bool(terminated or episode_truncated)
                total_reward += float(reward)

                if algorithm == "sarsa":
                    if terminated:
                        target = float(reward)
                        next_action = None
                    else:
                        next_action = epsilon_greedy(q_table[next_state], epsilon, rng)
                        target = float(reward) + gamma * q_table[next_state, next_action]
                elif algorithm == "q_learning":
                    target = float(reward)
                    if not terminated:
                        target += gamma * float(np.max(q_table[next_state]))
                    next_action = None
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")

                q_table[state, action] += alpha * (target - q_table[state, action])
                if episode_done:
                    break
                state = next_state
                action = (
                    next_action
                    if algorithm == "sarsa"
                    else epsilon_greedy(q_table[state], epsilon, rng)
                )

            rows.append({
                "episode": episode,
                "return": total_reward,
                "steps": float(steps),
            })
    finally:
        env.close()

    return q_table, rows


def evaluate_greedy(
    q_table: np.ndarray, seed: int, episodes: int, max_episode_steps: int
) -> list[dict[str, float]]:
    env = gym.make("CliffWalking-v1")
    rng = np.random.default_rng(seed)
    results: list[dict[str, float]] = []
    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            steps = 0
            terminated = False
            while steps < max_episode_steps and not terminated:
                # Select randomly only among greedy (maximum-value) actions. This avoids
                # an arbitrary fixed action-order bias when learned values are tied.
                values = q_table[state]
                best_actions = np.flatnonzero(values == np.max(values))
                action = int(rng.choice(best_actions))
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            results.append({
                "episode": episode,
                "return": total_reward,
                "steps": float(steps),
                "success": float(bool(terminated)),
            })
    finally:
        env.close()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched-seed SARSA and Q-Learning CliffWalking experiments."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=2_000,
                        help="Hard cap to prevent non-terminating exploratory/greedy policies.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "multi_seed",
        help="Output directory (defaults to projects/tabular_rl/results/multi_seed).",
    )
    args = parser.parse_args()

    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain one or more unique integers.")
    if args.episodes < 1 or args.evaluation_episodes < 1 or args.max_episode_steps < 1:
        parser.error("Episode counts and max episode steps must be positive.")
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("--epsilon must be between 0 and 1.")
    if not 0.0 < args.alpha <= 1.0 or not 0.0 <= args.gamma <= 1.0:
        parser.error("--alpha must be in (0, 1] and --gamma must be in [0, 1].")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows: list[dict[str, float | int | str]] = []

    for algorithm in ("sarsa", "q_learning"):
        for seed in args.seeds:
            q_table, training = train_one(
                algorithm, seed, args.episodes, args.alpha, args.gamma, args.epsilon,
                args.max_episode_steps,
            )
            evaluation = evaluate_greedy(
                q_table,
                seed=seed * 10_000 + 50_000,
                episodes=args.evaluation_episodes,
                max_episode_steps=args.max_episode_steps,
            )
            tail = training[-min(50, len(training)):]
            per_seed_rows.append({
                "algorithm": algorithm,
                "seed": seed,
                "training_episodes": args.episodes,
                "last_50_mean_return": mean(row["return"] for row in tail),
                "last_50_mean_steps": mean(row["steps"] for row in tail),
                "greedy_mean_return": mean(row["return"] for row in evaluation),
                "greedy_mean_steps": mean(row["steps"] for row in evaluation),
                "greedy_success_rate": mean(row["success"] for row in evaluation),
                "evaluation_episodes": args.evaluation_episodes,
            })
            print(
                f"{algorithm:10s} seed={seed}: "
                f"train(last50)={per_seed_rows[-1]['last_50_mean_return']:.2f}, "
                f"greedy={per_seed_rows[-1]['greedy_mean_return']:.2f}, "
                f"success={per_seed_rows[-1]['greedy_success_rate']:.0%}"
            )

    per_seed_path = args.output_dir / "per_seed.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_seed_rows[0]))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    summary_rows = []
    metric_names = (
        "last_50_mean_return",
        "last_50_mean_steps",
        "greedy_mean_return",
        "greedy_mean_steps",
        "greedy_success_rate",
    )
    for algorithm in ("sarsa", "q_learning"):
        subset = [row for row in per_seed_rows if row["algorithm"] == algorithm]
        summary: dict[str, float | int | str] = {
            "algorithm": algorithm,
            "seed_count": len(subset),
            "training_episodes_per_seed": args.episodes,
            "evaluation_episodes_per_seed": args.evaluation_episodes,
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in subset]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = pstdev(values)
        summary_rows.append(summary)

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(fieldnames=list(summary_rows[0]), f=file)
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "environment": "CliffWalking-v1",
        "gymnasium_version": gymnasium.__version__,
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "evaluation_episodes_per_seed": args.evaluation_episodes,
        "max_episode_steps": args.max_episode_steps,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "standard_deviation": "population (ddof=0), across independent seeds",
        "evaluation": "greedy actions; seeded random tie-breaking among equal maximum Q-values",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved: {per_seed_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
