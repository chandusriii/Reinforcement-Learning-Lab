"""Experience replay with validated checkpoint serialization."""

from collections import deque
import random


class ReplayMemory:
    """FIFO replay memory used by the Flappy Bird DQN."""

    def __init__(self, maxlen, seed=None):
        if int(maxlen) < 1:
            raise ValueError("Replay-memory capacity must be positive.")
        self.maxlen = int(maxlen)
        self.memory = deque([], maxlen=self.maxlen)
        self.rng = random.Random(seed)

    def append(self, new_exp):
        self.memory.append(new_exp)

    def sample(self, sample_size):
        return self.rng.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

    def state_dict(self):
        return {
            "maxlen": self.maxlen,
            "memory": list(self.memory),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state):
        if int(state["maxlen"]) != self.maxlen:
            raise ValueError(
                "Saved replay-memory capacity does not match the configured capacity."
            )
        items = state["memory"]
        if len(items) > self.maxlen:
            raise ValueError("Saved replay memory exceeds configured capacity.")
        self.memory = deque(items, maxlen=self.maxlen)
        self.rng.setstate(state["rng_state"])
