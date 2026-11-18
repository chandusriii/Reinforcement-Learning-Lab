"""Convolutional online/target Q-networks following the PyTorch Mario tutorial."""
from __future__ import annotations

import torch
from torch import nn


class MarioQNetwork(nn.Module):
    def __init__(self, input_channels: int = 4, action_count: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 4 or tuple(observations.shape[-2:]) != (84, 84):
            raise ValueError(f"Expected (batch, channels, 84, 84), got {tuple(observations.shape)}")
        return self.head(self.features(observations.float() / 255.0))


class DoubleDQN(nn.Module):
    """Independent online and frozen target networks, periodically synchronized."""

    def __init__(self, input_channels: int = 4, action_count: int = 2):
        super().__init__()
        self.online = MarioQNetwork(input_channels, action_count)
        self.target = MarioQNetwork(input_channels, action_count)
        self.sync_target()
        self.target.requires_grad_(False)

    @torch.no_grad()
    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.online(observations)
