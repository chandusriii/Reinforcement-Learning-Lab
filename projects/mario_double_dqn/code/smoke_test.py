"""Validate environment startup and the 4x84x84 observation contract."""

import argparse

from environment import make_mario_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = make_mario_env(render=args.render)
    try:
        state, info = env.reset(seed=123)
        assert state.shape == (4, 84, 84), f"Unexpected observation shape: {state.shape}"
        assert state.dtype.name == "uint8"
        for _ in range(args.steps):
            state, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert state.shape == (4, 84, 84)
            if terminated or truncated:
                state, info = env.reset()
        print("Mario environment smoke test passed: reset, step, and observation shape.")
        print(f"Observation: {state.shape}; actions: {env.action_space.n}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
