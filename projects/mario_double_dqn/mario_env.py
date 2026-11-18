"""Super Mario Bros environment and PyTorch-tutorial-style preprocessing."""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image


def _unpack_reset(result: Any):
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0], result[1]
    return result, {}


def _unpack_step(result: Any):
    if not isinstance(result, tuple):
        raise TypeError(f"Unexpected environment step result: {type(result)!r}")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated), bool(truncated), info
    if len(result) == 4:
        observation, reward, done, info = result
        truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
        terminated = bool(done) and not truncated
        return observation, float(reward), terminated, truncated, info
    raise ValueError(f"Expected 4- or 5-value step result, got {len(result)} values")


class MarioFrameStack:
    """Repeat each action, grayscale/resize frames, and stack four observations."""

    def __init__(self, env, skip: int = 4, size: int = 84, stack_size: int = 4):
        self.env = env
        self.skip = int(skip)
        self.size = int(size)
        self.stack_size = int(stack_size)
        self.action_space = env.action_space
        self.observation_space = None
        self.frames: deque[np.ndarray] = deque(maxlen=self.stack_size)

    def _process(self, frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(f"Expected RGB/RGBA observation, got shape {array.shape}")
        image = Image.fromarray(array.astype(np.uint8)).convert("L").resize(
            (self.size, self.size), getattr(Image, "Resampling", Image).BILINEAR
        )
        return np.asarray(image, dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        try:
            result = self.env.reset(seed=seed, options=options)
        except TypeError:
            result = self.env.reset()
        observation, info = _unpack_reset(result)
        processed = self._process(observation)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(processed.copy())
        return np.stack(self.frames, axis=0), info

    def step(self, action: int):
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}
        observation = None
        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = _unpack_step(self.env.step(int(action)))
            total_reward += reward
            if terminated or truncated:
                break
        if observation is None:
            raise RuntimeError("The Mario environment did not return an observation")
        if isinstance(info, dict) and info.get("flag_get", False):
            terminated = True
        self.frames.append(self._process(observation))
        return np.stack(self.frames, axis=0), total_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_mario_env():
    """Create the assignment's World 1-1 environment with two simple actions."""
    try:
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
    except ImportError as exc:
        raise RuntimeError(
            "Mario dependencies are missing. Install requirements-mario.txt in a dedicated venv."
        ) from exc

    # The coursework tutorial's Super Mario package uses the legacy Gym API.
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    return MarioFrameStack(env, skip=4, size=84, stack_size=4)
