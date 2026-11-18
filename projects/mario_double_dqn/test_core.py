"""Offline tests for environment adaptation, model, replay buffer, and agent."""
import tempfile
from pathlib import Path
import numpy as np
import torch

from model import DoubleDQN
from replay_buffer import ReplayBuffer
from agent import MarioAgent
from mario_env import MarioFrameStack


class FakeActionSpace:
    n = 2


class FakeEnv:
    def __init__(self):
        self.action_space = FakeActionSpace()
        self.counter = 0

    def reset(self, **kwargs):
        self.counter = 0
        return np.zeros((96, 96, 3), dtype=np.uint8), {"reset": True}

    def step(self, action):
        self.counter += 1
        frame = np.full((96, 96, 3), self.counter, dtype=np.uint8)
        return frame, 0.25, self.counter >= 3, False, {"flag_get": False}

    def close(self):
        pass

    def render(self):
        return None


def main():
    torch.manual_seed(7)
    wrapped = MarioFrameStack(FakeEnv(), skip=4, size=84, stack_size=4)
    initial, reset_info = wrapped.reset(seed=11)
    assert initial.shape == (4, 84, 84) and reset_info["reset"]
    next_state, reward, terminated, truncated, _ = wrapped.step(0)
    assert next_state.shape == (4, 84, 84) and reward == 0.75
    assert terminated and not truncated
    wrapped.close()

    model = DoubleDQN(input_channels=4, action_count=2)
    sample = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    output = model.online(sample)
    assert output.shape == (2, 2), output.shape
    assert torch.isfinite(output).all()
    assert not any(parameter.requires_grad for parameter in model.target.parameters())

    with tempfile.TemporaryDirectory() as directory:
        buffer = ReplayBuffer(Path(directory) / "buffer", capacity=16)
        state = np.zeros((4, 84, 84), dtype=np.uint8)
        next_state = np.ones((4, 84, 84), dtype=np.uint8)
        for index in range(8):
            buffer.add(state, index % 2, float(index), next_state, index == 7)
        batch = buffer.sample(4, torch.device("cpu"))
        assert batch[0].shape == (4, 4, 84, 84)
        assert batch[1].shape == (4,)
        assert batch[2].shape == (4,)

    with tempfile.TemporaryDirectory() as directory:
        agent = MarioAgent(
            action_count=2,
            run_dir=directory,
            replay_capacity=16,
            batch_size=2,
            burn_in=0,
            learn_every=1,
            target_sync_every=100,
        )
        state = np.zeros((4, 84, 84), dtype=np.uint8)
        next_state = np.full((4, 84, 84), 8, dtype=np.uint8)
        for _ in range(4):
            action = agent.select_action(state)
            agent.remember(state, action, 1.0, next_state, False)
            info = agent.learn()
        assert info is not None and np.isfinite(info["loss"])
        checkpoint = Path(directory) / "test.pt"
        agent.save(checkpoint)
        assert checkpoint.exists()
    print(
        "PASS: environment preprocessing/API adapter, CNN forward, target freeze, "
        "replay sampling, DDQN update, checkpoint save"
    )


if __name__ == "__main__":
    main()
