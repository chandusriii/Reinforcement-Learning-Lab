"""Measure Flappy Bird observations against the environment's declared Box."""

from __future__ import annotations

import argparse
import math

import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401 - registers the environment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect raw Flappy Bird observations for Box-space violations."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive.")
    return args


def split_step(result):
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, bool(terminated), bool(truncated), info
    observation, reward, done, info = result
    truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
    return observation, reward, bool(done) and not truncated, truncated, info


def main():
    args = parse_args()
    # Disable Gymnasium's automatic checker only so we can measure and report every
    # raw violation instead of stopping at the first warning. No values are clipped.
    env = gym.make(
        "FlappyBird-v0",
        use_lidar=True,
        normalize_obs=True,
        disable_env_checker=True,
    )
    space = env.observation_space
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"Expected Box observation space, found {type(space).__name__}")
    env.action_space.seed(args.seed)

    total_observations = 0
    violating_observations = 0
    violating_values = 0
    global_min = math.inf
    global_max = -math.inf
    first_violations = []

    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset(seed=args.seed + episode)
            for step in range(1, args.max_steps + 1):
                values = np.asarray(observation)
                if values.shape != space.shape:
                    raise ValueError(
                        f"Episode {episode}, step {step}: expected shape {space.shape}, "
                        f"received {values.shape}"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Episode {episode}, step {step}: non-finite observation detected."
                    )

                low = float(np.min(values))
                high = float(np.max(values))
                global_min = min(global_min, low)
                global_max = max(global_max, high)
                total_observations += 1

                outside = np.logical_or(values < space.low, values > space.high)
                if np.any(outside):
                    violating_observations += 1
                    violating_values += int(np.count_nonzero(outside))
                    if len(first_violations) < 10:
                        bad = values[outside]
                        first_violations.append({
                            "episode": episode,
                            "step": step,
                            "min_outside": float(np.min(bad)),
                            "max_outside": float(np.max(bad)),
                            "count": int(bad.size),
                        })

                action = env.action_space.sample()
                observation, _, terminated, truncated, _ = split_step(
                    env.step(action)
                )
                if terminated or truncated:
                    break
    finally:
        env.close()

    print(f"Environment: FlappyBird-v0 (use_lidar=True, normalize_obs=True)")
    print(f"Declared observation shape: {space.shape}, dtype={space.dtype}")
    print(f"Declared bounds: [{np.min(space.low)}, {np.max(space.high)}]")
    print(f"Episodes: {args.episodes}; maximum steps per episode: {args.max_steps}")
    print(f"Observations inspected: {total_observations}")
    print(f"Observed minimum: {global_min:.17g}")
    print(f"Observed maximum: {global_max:.17g}")
    print(f"Observations outside Box: {violating_observations}")
    print(f"Individual values outside Box: {violating_values}")
    if first_violations:
        print("First violations:")
        for item in first_violations:
            print(item)
    else:
        print("No observation-space violations were observed in this sample.")


if __name__ == "__main__":
    main()
