"""Disk-backed replay memory to avoid holding the full image buffer in RAM."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, directory: str | Path, capacity: int = 10_000,
                 state_shape: tuple[int, int, int] = (4, 84, 84)):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.states = np.lib.format.open_memmap(
            self.directory / "states.npy", mode="w+", dtype=np.uint8,
            shape=(self.capacity, *state_shape)
        )
        self.next_states = np.lib.format.open_memmap(
            self.directory / "next_states.npy", mode="w+", dtype=np.uint8,
            shape=(self.capacity, *state_shape)
        )
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminals = np.zeros(self.capacity, dtype=np.uint8)
        self.position = 0
        self.size = 0

    def add(self, state, action: int, reward: float, next_state, terminal: bool) -> None:
        state_array = np.asarray(state, dtype=np.uint8)
        next_array = np.asarray(next_state, dtype=np.uint8)
        if state_array.shape != self.state_shape or next_array.shape != self.state_shape:
            raise ValueError(
                f"Expected state shape {self.state_shape}; received {state_array.shape} and {next_array.shape}"
            )
        index = self.position
        self.states[index] = state_array
        self.next_states[index] = next_array
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.terminals[index] = bool(terminal)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        if self.size < batch_size:
            raise ValueError(f"Need {batch_size} transitions, only {self.size} available")
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        states = torch.from_numpy(np.asarray(self.states[indices])).to(device=device)
        next_states = torch.from_numpy(np.asarray(self.next_states[indices])).to(device=device)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.long, device=device)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device)
        terminals = torch.as_tensor(self.terminals[indices], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, terminals

    def flush(self) -> None:
        self.states.flush()
        self.next_states.flush()
