"""Double DQN agent and persistent disk-backed replay buffer for Mario."""

from __future__ import annotations

import json
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
        was_integer = not torch.is_floating_point(states)
        states = states.to(dtype=torch.float32)
        if was_integer or (states.numel() and states.max() > 1.0):
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
    """Disk-backed circular replay memory with explicit, validated resume metadata."""

    ARRAY_FILES = {
        "states": "states.npy",
        "next_states": "next_states.npy",
        "actions": "actions.npy",
        "rewards": "rewards.npy",
        "terminated": "terminated.npy",
    }

    def __init__(
        self,
        capacity: int = 20_000,
        state_shape: Sequence[int] = (4, 84, 84),
        directory: str | Path = "replay_memory",
        resume: bool = False,
    ) -> None:
        if capacity < 2:
            raise ValueError("Replay capacity must be at least two.")
        self.capacity = int(capacity)
        self.state_shape = tuple(map(int, state_shape))
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.directory / "metadata.json"
        expected_shapes = {
            "states": (self.capacity, *self.state_shape),
            "next_states": (self.capacity, *self.state_shape),
            "actions": (self.capacity,),
            "rewards": (self.capacity,),
            "terminated": (self.capacity,),
        }
        dtypes = {
            "states": np.uint8,
            "next_states": np.uint8,
            "actions": np.int64,
            "rewards": np.float32,
            "terminated": np.bool_,
        }

        if resume:
            missing = [
                filename for filename in (*self.ARRAY_FILES.values(), self.metadata_path.name)
                if not (self.directory / filename).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "Cannot resume replay memory; missing files: " + ", ".join(missing)
                )
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if int(metadata.get("capacity", -1)) != self.capacity:
                raise ValueError("Replay capacity differs from the saved replay buffer.")
            if tuple(metadata.get("state_shape", ())) != self.state_shape:
                raise ValueError("Observation shape differs from the saved replay buffer.")
            self.position = int(metadata["position"])
            self.size = int(metadata["size"])
            self.saved_curr_step = int(metadata["curr_step"])
            if not 0 <= self.position < self.capacity:
                raise ValueError("Saved replay-buffer position is invalid.")
            if not 0 <= self.size <= self.capacity:
                raise ValueError("Saved replay-buffer size is invalid.")
            if self.saved_curr_step < 0:
                raise ValueError("Saved replay-buffer step count is invalid.")
            mode = "r+"
        else:
            self.position = 0
            self.size = 0
            self.saved_curr_step = 0
            mode = "w+"

        for name, filename in self.ARRAY_FILES.items():
            array = np.lib.format.open_memmap(
                self.directory / filename,
                mode=mode,
                dtype=dtypes[name],
                shape=expected_shapes[name],
            )
            if tuple(array.shape) != expected_shapes[name] or array.dtype != dtypes[name]:
                raise ValueError(f"Replay array {filename} has an incompatible shape or dtype.")
            setattr(self, name, array)

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

    def flush(self, curr_step: Optional[int] = None) -> None:
        for name in self.ARRAY_FILES:
            getattr(self, name).flush()
        if curr_step is not None:
            self.saved_curr_step = int(curr_step)
        metadata = {
            "capacity": self.capacity,
            "state_shape": list(self.state_shape),
            "position": self.position,
            "size": self.size,
            "curr_step": self.saved_curr_step,
        }
        temporary = self.metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.metadata_path)


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
        resume_replay: bool = False,
    ) -> None:
        self.state_dim = tuple(map(int, state_dim))
        self.action_dim = int(action_dim)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.net = MarioNet(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.online.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(
            replay_capacity, self.state_dim, replay_dir, resume=resume_replay
        )
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
        self.last_saved_episode = 0

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
        next_actions = self.net(next_states, network="online").argmax(dim=1)
        next_values = self.net(next_states, network="target").gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)
        return rewards + self.gamma * (1.0 - terminated) * next_values

    def learn(self):
        if self.curr_step > 0 and self.curr_step % self.sync_every == 0:
            self.net.sync_target()
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

    def save(self, path: str | Path, episode: Optional[int] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.memory.flush(self.curr_step)
        self.last_saved_episode = (
            self.last_saved_episode if episode is None else int(episode)
        )
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "exploration_rate": self.exploration_rate,
                "curr_step": self.curr_step,
                "gamma": self.gamma,
                "episode": self.last_saved_episode,
                "replay_position": self.memory.position,
                "replay_size": self.memory.size,
            },
            path,
        )

    def load(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if tuple(checkpoint["state_dim"]) != self.state_dim:
            raise ValueError("Checkpoint observation shape does not match this agent.")
        if int(checkpoint["action_dim"]) != self.action_dim:
            raise ValueError("Checkpoint action count does not match this environment.")
        required = {"episode", "replay_position", "replay_size"}
        if not required.issubset(checkpoint):
            raise ValueError(
                "This checkpoint predates resumable replay memory. Start a new run "
                "or use a checkpoint created by the persistent-replay version."
            )
        if (
            int(checkpoint["curr_step"]) != self.memory.saved_curr_step
            or int(checkpoint["replay_position"]) != self.memory.position
            or int(checkpoint["replay_size"]) != self.memory.size
        ):
            raise ValueError(
                "Checkpoint and replay-memory files are from different save points. "
                "Resume only from the latest checkpoint using the same replay directory."
            )
        self.net.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.exploration_rate = float(checkpoint["exploration_rate"])
        self.curr_step = int(checkpoint["curr_step"])
        self.last_saved_episode = int(checkpoint["episode"])
        return self.last_saved_episode
