"""Fast CPU tests for the network, Double DQN target and optimizer step."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mario_dqn import MarioAgent, MarioNet


class MarioCoreTests(unittest.TestCase):
    def test_network_output_and_target_freeze(self):
        network = MarioNet((4, 84, 84), 2)
        sample = torch.zeros((2, 4, 84, 84), dtype=torch.uint8)
        output = network(sample, "online")
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(not p.requires_grad for p in network.target.parameters()))

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
            agent.save(checkpoint)
            restored = MarioAgent(
                replay_dir=Path(directory) / "replay_restored",
                replay_capacity=8,
                batch_size=4,
            )
            restored.load(checkpoint)
            self.assertEqual(restored.curr_step, agent.curr_step)
            self.assertTrue(np.isfinite(q_value))


if __name__ == "__main__":
    unittest.main()
