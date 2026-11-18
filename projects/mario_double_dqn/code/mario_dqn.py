"""Double DQN agent and replay buffer for the Mario assignment."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn


class MarioNet(nn.Module):
    """Tutorial-style CNN with separate online and target Q-networks."""

    def __init__(self, input_dim: Sequence[int], action_dim: int) -> None:
        super().__init__()
        channels, height, width = map(int, input_dim)
        if (height, width) != (84, 84):
            raise ValueError(f"Expected 84x84 frames, received {height}x{width}.")
        if channels != 4:
            raise ValueError(f"Expected four stacked frames, received {channels}.")
        if action_dim < 2:
            raise ValueError("The action space must contain at least two actions.")

        self.online = self._build_network(channels, action_dim)
        self.target = self._build_network(channels, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _build_network(channels: int, action_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, states: torch.Tensor, network: str = "online") -> torch.Tensor:
        if states.ndim == 3:
            states = states.unsqueeze(0)
        states = states.to(dtype=torch.float32)
        if states.numel() and states.max() > 1.0:
            states = states / 255.0
        if network == "online":
            return self.online(states)
        if network == "target":
            return self.target(states)
        raise ValueError("network must be 'online' or 'target'.")

    @torch.no_grad()
    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()


class ReplayBuffer:
    """Disk-backed circular replay memory; uint8 frames keep storage compact."""

    def __init__(
        self,
        capacity: int = 20_000,
        state_shape: Sequence[int] = (4, 84, 84),
        directory: str | Path = "replay_memory",
    ) -> None:
        if capacity < 2:
            raise ValueError("Replay capacity must be at least two.")
        self.capacity = int(capacity)
        self.state_shape = tuple(map(int, state_shape))
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.states = np.lib.format.open_memmap(
            self.directory / "states.npy", mode="w+", dtype=np.uint8,
            shape=(self.capacity, *self.state_shape)
        )
        self.next_states = np.lib.format.open_memmap(
            self.directory / "next_states.npy", mode="w+", dtype=np.uint8,
            shape=(self.capacity, *self.state_shape)
        )
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        state_array = np.asarray(state, dtype=np.uint8)
        next_array = np.asarray(next_state, dtype=np.uint8)
        if state_array.shape != self.state_shape or next_array.shape != self.state_shape:
            raise ValueError(
                f"Expected state shape {self.state_shape}; got "
                f"{state_array.shape} and {next_array.shape}."
            )
        self.states[self.position] = state_array
        self.next_states[self.position] = next_array
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.terminated[self.position] = bool(terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        if self.size < batch_size:
            raise ValueError("Not enough transitions in replay memory.")
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        states = torch.as_tensor(np.asarray(self.states[indices]), device=device)
        next_states = torch.as_tensor(np.asarray(self.next_states[indices]), device=device)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.long, device=device)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device)
        terminated = torch.as_tensor(
            self.terminated[indices], dtype=torch.float32, device=device
        )
        return states, actions, rewards, next_states, terminated

    def flush(self) -> None:
        self.states.flush()
        self.next_states.flush()


class MarioAgent:
    """Epsilon-greedy Double DQN learner following the supplied PyTorch tutorial."""

    def __init__(
        self,
        state_dim: Sequence[int] = (4, 84, 84),
        action_dim: int = 2,
        device: Optional[str] = None,
        replay_dir: str | Path = "replay_memory",
        replay_capacity: int = 20_000,
        batch_size: int = 32,
        burnin: int = 10_000,
        learn_every: int = 3,
        sync_every: int = 10_000,
        save_every: int = 500_000,
        gamma: float = 0.9,
        learning_rate: float = 2.5e-4,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99999975,
        epsilon_min: float = 0.1,
    ) -> None:
        self.state_dim = tuple(map(int, state_dim))
        self.action_dim = int(action_dim)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.net = MarioNet(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.online.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(replay_capacity, self.state_dim, replay_dir)

        self.batch_size = int(batch_size)
        self.burnin = int(burnin)
        self.learn_every = int(learn_every)
        self.sync_every = int(sync_every)
        self.save_every = int(save_every)
        self.gamma = float(gamma)
        self.exploration_rate = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.curr_step = 0

    @torch.no_grad()
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.exploration_rate:
            action = random.randrange(self.action_dim)
        else:
            tensor = torch.as_tensor(
                np.asarray(state), dtype=torch.uint8, device=self.device
            ).unsqueeze(0)
            action = int(self.net(tensor, network="online").argmax(dim=1).item())

        if explore:
            self.exploration_rate = max(
                self.epsilon_min, self.exploration_rate * self.epsilon_decay
            )
        self.curr_step += 1
        return action

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.memory.add(state, action, reward, next_state, terminated)

    @torch.no_grad()
    def compute_td_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        # Double DQN: online selects a'; target evaluates that same a'.
        next_actions = self.net(next_states, network="online").argmax(dim=1)
        next_values = self.net(next_states, network="target").gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)
        return rewards + self.gamma * (1.0 - terminated) * next_values

    def learn(self):
        if self.curr_step > 0 and self.curr_step % self.sync_every == 0:
            self.net.sync_target()
        if self.curr_step > 0 and self.curr_step % self.save_every == 0:
            self.memory.flush()
        if self.curr_step < self.burnin or self.curr_step % self.learn_every != 0:
            return None, None
        if self.memory.size < self.batch_size:
            return None, None

        states, actions, rewards, next_states, terminated = self.memory.sample(
            self.batch_size, self.device
        )
        q_values = self.net(states, network="online").gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        td_target = self.compute_td_target(rewards, next_states, terminated)

        loss = self.loss_fn(q_values, td_target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(q_values.detach().mean().item()), float(loss.detach().item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.memory.flush()
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "exploration_rate": self.exploration_rate,
                "curr_step": self.curr_step,
                "gamma": self.gamma,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if tuple(checkpoint["state_dim"]) != self.state_dim:
            raise ValueError("Checkpoint observation shape does not match this agent.")
        if int(checkpoint["action_dim"]) != self.action_dim:
            raise ValueError("Checkpoint action count does not match this environment.")
        self.net.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.exploration_rate = float(checkpoint["exploration_rate"])
        self.curr_step = int(checkpoint["curr_step"])
