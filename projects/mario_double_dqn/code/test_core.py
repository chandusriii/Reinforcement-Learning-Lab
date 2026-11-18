"""Fast CPU tests for the network, Double DQN target and optimizer step."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mario_dqn import MarioAgent, MarioNet


class FixedQNetworks:
    def __call__(self, states, network):
        if network == "online":
            # Online network chooses action 1.
            return torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        if network == "target":
            # Target network values action 0 much higher, but must evaluate action 1.
            return torch.tensor([[100.0, 5.0]], dtype=torch.float32)
        raise ValueError(network)


class MarioCoreTests(unittest.TestCase):
    def test_network_output_and_target_freeze(self):
        network = MarioNet((4, 84, 84), 2)
        sample = torch.zeros((2, 4, 84, 84), dtype=torch.uint8)
        output = network(sample, "online")
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(not p.requires_grad for p in network.target.parameters()))

    def test_double_dqn_selects_online_and_evaluates_target(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = MarioAgent(
                replay_dir=Path(directory) / "replay",
                replay_capacity=8,
                batch_size=2,
            )
            agent.net = FixedQNetworks()
            rewards = torch.tensor([1.0])
            next_states = torch.zeros((1, 4, 84, 84), dtype=torch.uint8)
            not_terminal = torch.tensor([0.0])
            target = agent.compute_td_target(rewards, next_states, not_terminal)
            self.assertAlmostEqual(float(target.item()), 1.0 + 0.9 * 5.0, places=5)
            terminal = torch.tensor([1.0])
            terminal_target = agent.compute_td_target(rewards, next_states, terminal)
            self.assertAlmostEqual(float(terminal_target.item()), 1.0, places=5)

    def test_double_dqn_learning_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = MarioAgent(
                replay_dir=Path(directory) / "replay",
                replay_capacity=8,
                batch_size=4,
                burnin=1,
                learn_every=1,
                sync_every=2,
            )
            state = np.zeros((4, 84, 84), dtype=np.uint8)
            next_state = np.full((4, 84, 84), 15, dtype=np.uint8)
            for index in range(4):
                action = agent.act(state, explore=True)
                agent.remember(state, action, 1.0, next_state, index == 3)
                agent.learn()
            q_value, loss = agent.learn()
            self.assertIsNotNone(loss)
            self.assertTrue(np.isfinite(loss))
            checkpoint = Path(directory) / "agent.pt"
            agent.save(checkpoint, episode=5)
            restored = MarioAgent(
                replay_dir=Path(directory) / "replay",
                replay_capacity=8,
                batch_size=4,
                resume_replay=True,
            )
            saved_episode = restored.load(checkpoint)
            self.assertEqual(saved_episode, 5)
            self.assertEqual(restored.curr_step, agent.curr_step)
            self.assertEqual(restored.memory.size, agent.memory.size)
            self.assertTrue(np.isfinite(q_value))


if __name__ == "__main__":
    unittest.main()
