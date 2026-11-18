"""Measure raw Flappy Bird observations and verify the bounded agent adapter."""

from __future__ import annotations

import argparse

import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401 - registers the environment
from bounded_env import ClipObservationBounds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure raw LiDAR range and verify Box-bounded observations."
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
    truncated = (
        bool(info.get("TimeLimit.truncated", False))
        if isinstance(info, dict) else False
    )
    return observation, reward, bool(done) and not truncated, truncated, info


def main():
    args = parse_args()
    # The upstream raw values are measured inside the adapter before clipping.
    # Disable the passive checker so its first warning does not interrupt the run.
    raw_env = gym.make(
        "FlappyBird-v0",
        use_lidar=True,
        normalize_obs=True,
        disable_env_checker=True,
    )
    env = ClipObservationBounds(raw_env)
    space = env.observation_space
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"Expected Box observation space, found {type(space).__name__}")
    env.action_space.seed(args.seed)

    total_observations = 0
    returned_values_outside_box = 0

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
                if not space.contains(values):
                    returned_values_outside_box += int(
                        np.count_nonzero(
                            np.logical_or(values < space.low, values > space.high)
                        )
                    )
                    raise AssertionError(
                        "The observation adapter returned a value outside the Box."
                    )
                total_observations += 1

                action = env.action_space.sample()
                observation, _, terminated, truncated, _ = split_step(
                    env.step(action)
                )
                if terminated or truncated:
                    break
    finally:
        env.close()

    print("Environment: FlappyBird-v0 (use_lidar=True, normalize_obs=True)")
    print(f"Declared observation shape: {space.shape}, dtype={space.dtype}")
    print(f"Declared bounds: [{np.min(space.low)}, {np.max(space.high)}]")
    print(f"Episodes: {args.episodes}; maximum steps per episode: {args.max_steps}")
    print(f"Observations inspected: {total_observations}")
    print(f"Raw environment minimum: {env.raw_min:.17g}")
    print(f"Raw environment maximum: {env.raw_max:.17g}")
    print(f"Raw values clipped to Box: {env.clipped_values}")
    print(f"Returned values outside Box after adapter: {returned_values_outside_box}")
    print("PASS: every observation returned by the adapter satisfied the declared Box.")


if __name__ == "__main__":
    main()
