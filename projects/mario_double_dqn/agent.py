"""Double DQN Mario agent: epsilon-greedy actions, replay, and TD learning."""
from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import torch
from torch import nn

from model import DoubleDQN
from replay_buffer import ReplayBuffer


class MarioAgent:
    def __init__(self, action_count: int, run_dir: str | Path, device: str | None = None,
                 replay_capacity: int = 10_000, batch_size: int = 32,
                 gamma: float = 0.9, learning_rate: float = 2.5e-4,
                 burn_in: int = 10_000, learn_every: int = 3,
                 target_sync_every: int = 10_000, save_every: int = 50_000,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.99999975):
        if action_count < 1 or batch_size < 1 or learn_every < 1 or target_sync_every < 1:
            raise ValueError("Action count, batch size, and update intervals must be positive")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.action_count = int(action_count)
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.burn_in = int(burn_in)
        self.learn_every = int(learn_every)
        self.target_sync_every = int(target_sync_every)
        self.save_every = int(save_every)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.steps = 0
        self.net = DoubleDQN(input_channels=4, action_count=self.action_count).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.online.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(self.run_dir / "replay", capacity=replay_capacity)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            action = random.randrange(self.action_count)
        else:
            state_tensor = torch.as_tensor(
                np.asarray(state), dtype=torch.uint8, device=self.device
            ).unsqueeze(0)
            action = int(self.net.online(state_tensor).argmax(dim=1).item())
        if explore:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1
        return action

    def remember(self, state, action, reward, next_state, terminal) -> None:
        self.replay.add(state, action, reward, next_state, terminal)

    def learn(self):
        if self.steps > 0 and self.steps % self.target_sync_every == 0:
            self.net.sync_target()
        if self.save_every > 0 and self.steps > 0 and self.steps % self.save_every == 0:
            self.save(self.run_dir / f"mario_step_{self.steps}.pt")
        if self.steps < self.burn_in or self.steps % self.learn_every != 0:
            return None
        if self.replay.size < self.batch_size:
            return None

        states, actions, rewards, next_states, terminals = self.replay.sample(
            self.batch_size, self.device
        )
        q_estimate = self.net.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online network selects; target network evaluates.
            next_actions = self.net.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.net.target(next_states).gather(1, next_actions).squeeze(1)
            q_target = rewards + (1.0 - terminals) * self.gamma * next_q
        loss = self.loss_fn(q_estimate, q_target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return {"loss": float(loss.item()), "q_mean": float(q_estimate.detach().mean().item())}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.replay.flush()
        torch.save({
            "online": self.net.online.state_dict(),
            "target": self.net.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "action_count": self.action_count,
            "gamma": self.gamma,
        }, path)

    def load(self, path: str | Path, load_optimizer: bool = False) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.net.online.load_state_dict(checkpoint["online"])
        self.net.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon_min))
        self.steps = int(checkpoint.get("steps", 0))
