"""Environment setup and tutorial-aligned visual preprocessing for Mario."""

from __future__ import annotations

from collections import deque
from typing import Optional

import gym
import numpy as np
from PIL import Image
from gym.spaces import Box


class MarioFrameStack(gym.Wrapper):
    """Repeat actions, sum rewards, grayscale/resize observations and stack 4 frames."""

    def __init__(self, env, frame_skip: int = 4, frame_stack: int = 4, image_size: int = 84):
        super().__init__(env)
        if frame_skip < 1 or frame_stack < 1 or image_size < 1:
            raise ValueError("Frame skip, stack count and image size must be positive.")
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.image_size = image_size
        self.frames = deque(maxlen=frame_stack)
        self.observation_space = Box(
            low=0, high=255, shape=(frame_stack, image_size, image_size), dtype=np.uint8
        )

    def _preprocess(self, observation: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.asarray(observation, dtype=np.uint8)).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    def reset(self, **kwargs):
        try:
            result = self.env.reset(**kwargs)
        except TypeError:
            seed = kwargs.get("seed")
            if seed is not None and hasattr(self.env, "seed"):
                self.env.seed(seed)
            result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation, info = result, {}
        frame = self._preprocess(observation)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        total_reward = 0.0
        observation = None
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.frame_skip):
            result = self.env.step(action)
            if len(result) == 5:
                observation, reward, terminated, truncated, info = result
            else:
                observation, reward, done, info = result
                terminated, truncated = bool(done), False
            total_reward += float(reward)
            if terminated or truncated:
                break

        if observation is None:
            raise RuntimeError("The Mario environment returned no observation.")
        self.frames.append(self._preprocess(observation))
        return np.stack(self.frames, axis=0), total_reward, bool(terminated), bool(truncated), info


def make_mario_env(render: bool = False, frame_skip: int = 4):
    """Create the two-action Super Mario Bros. level used by the PyTorch assignment."""
    try:
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
    except ImportError as exc:
        raise RuntimeError(
            "Mario dependencies are missing. Install requirements-mario.txt in a "
            "separate Python 3.10 virtual environment."
        ) from exc

    make_kwargs = {"render_mode": "human" if render else "rgb", "apply_api_compatibility": True}
    try:
        base_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", **make_kwargs)
    except TypeError:
        # Compatibility path for older Gym / gym-super-mario-bros combinations.
        base_env = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0", new_step_api=True
        )

    # Action 0: walk right. Action 1: jump right. Matches the assignment tutorial.
    base_env = JoypadSpace(base_env, [["right"], ["right", "A"]])
    return MarioFrameStack(base_env, frame_skip=frame_skip, frame_stack=4, image_size=84)
