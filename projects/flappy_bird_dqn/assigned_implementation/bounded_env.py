"""Observation-space enforcement for Flappy Bird's normalized LiDAR input."""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class ClipObservationBounds(gym.Wrapper):
    """Clip raw observations to their declared Box bounds and record corrections.

    The upstream Flappy Bird LiDAR can exceed the normalized upper bound by a
    small amount because ray-intersection coordinates are rounded. This wrapper
    enforces the declared contract at the agent boundary and makes every
    correction measurable.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(self.observation_space, gym.spaces.Box):
            raise TypeError("ClipObservationBounds requires a Box observation space.")
        self.clipped_values = 0
        self.observation_values = 0
        self.raw_min = float("inf")
        self.raw_max = float("-inf")

    def _clip(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=self.observation_space.dtype)
        if values.shape != self.observation_space.shape:
            raise ValueError(
                f"Expected observation shape {self.observation_space.shape}; "
                f"received {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Environment returned a non-finite observation.")

        self.raw_min = min(self.raw_min, float(np.min(values)))
        self.raw_max = max(self.raw_max, float(np.max(values)))
        outside = np.logical_or(
            values < self.observation_space.low,
            values > self.observation_space.high,
        )
        self.clipped_values += int(np.count_nonzero(outside))
        self.observation_values += int(values.size)
        bounded = np.clip(
            values, self.observation_space.low, self.observation_space.high
        ).astype(self.observation_space.dtype, copy=False)

        if not self.observation_space.contains(bounded):
            raise RuntimeError("Clipped observation still violates its Box space.")
        return bounded

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation, info = result, {}
        return self._clip(observation), info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            return self._clip(observation), reward, terminated, truncated, info
        observation, reward, done, info = result
        return self._clip(observation), reward, done, info
