"""Seeded evaluation/training copy; the assigned agent.py is preserved unchanged."""

import flappy_bird_gymnasium  # noqa: F401 - registers the environment
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random
import csv
import numpy as np

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)



class ClipObservationBounds(gym.Wrapper):
    """Clip tiny upstream LiDAR range overshoots to the declared Box space."""

    def __init__(self, env):
        super().__init__(env)
        self.clipped_values = 0
        self.observation_values = 0

    def _clip(self, observation):
        values = np.asarray(observation, dtype=self.observation_space.dtype)
        outside = np.logical_or(
            values < self.observation_space.low,
            values > self.observation_space.high,
        )
        self.clipped_values += int(np.count_nonzero(outside))
        self.observation_values += int(values.size)
        return np.clip(
            values, self.observation_space.low, self.observation_space.high
        ).astype(self.observation_space.dtype, copy=False)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation, info = result, {}
        return self._clip(observation), info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            return self._clip(observation), reward, terminated, truncated, info
        observation, reward, done, info = result
        return self._clip(observation), reward, done, info


def split_step(result):
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, bool(terminated), bool(truncated), info
    observation, reward, done, info = result
    truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
    return observation, reward, bool(done) and not truncated, truncated, info


class Agent:
    def __init__(self, param_set):
        self.param_set = param_set
        with open("parameters.yaml", "r", encoding="utf-8") as file:
            all_param_set = yaml.safe_load(file)
            if param_set not in all_param_set:
                raise KeyError(f"Unknown parameter set: {param_set}")
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False, seed=42, episodes=100, max_steps_per_episode=10_000):
        random.seed(seed)
        torch.manual_seed(seed)
        raw_env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None,
            use_lidar=True,
            normalize_obs=True,
            disable_env_checker=True,
        )
        env = ClipObservationBounds(raw_env)
        env.action_space.seed(seed)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy_dqn = DQN(num_states, num_actions).to(device)
        evaluation_rewards = []

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")
            episode_range = range(episodes)
        else:
            if not os.path.isfile(self.MODEL_FILE):
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.MODEL_FILE}. Train the agent first."
                )
            policy_dqn.load_state_dict(
                torch.load(self.MODEL_FILE, map_location=device, weights_only=True)
            )
            policy_dqn.eval()
            episode_range = range(episodes)

        try:
            for episode in episode_range:
                try:
                    reset_result = env.reset(seed=seed + episode)
                except TypeError:
                    # Compatibility fallback for older Gymnasium builds.
                    reset_result = env.reset()
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    reset_observation = reset_result[0]
                else:
                    reset_observation = reset_result
                state = torch.as_tensor(
                    reset_observation, dtype=torch.float32, device=device
                )
                episode_reward = 0.0
                terminated = False
                truncated = False
                episode_steps = 0

                while (
                    not (terminated or truncated)
                    and episode_reward < self.reward_threshold
                    and episode_steps < max_steps_per_episode
                ):
                    if is_training and random.random() < epsilon:
                        action = torch.tensor(
                            env.action_space.sample(), dtype=torch.long, device=device
                        )
                    else:
                        with torch.no_grad():
                            action = policy_dqn(
                                state.unsqueeze(dim=0)
                            ).squeeze().argmax()

                    next_state, reward, terminated, truncated, _ = split_step(
                        env.step(action.item())
                    )
                    episode_reward += float(reward)
                    episode_steps += 1
                    reward_tensor = torch.tensor(
                        reward, dtype=torch.float32, device=device
                    )
                    next_state_tensor = torch.as_tensor(
                        next_state, dtype=torch.float32, device=device
                    )

                    if is_training:
                        memory.append((
                            state, action, next_state_tensor,
                            reward_tensor, bool(terminated)
                        ))
                        steps += 1

                    state = next_state_tensor

                epsilon_text = f"{epsilon:.6f}" if is_training else "N/A"
                print(
                    f"episode={episode + 1} return={episode_reward:.3f} "
                    f"steps={episode_steps} epsilon={epsilon_text}"
                )

                if not is_training:
                    evaluation_rewards.append(episode_reward)

                if is_training:
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    if episode_reward > best_reward:
                        with open(self.LOG_FILE, "a", encoding="utf-8") as file:
                            file.write(
                                f"best reward = {episode_reward} for episode={episode + 1}\n"
                            )
                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        best_reward = episode_reward

                if is_training and (episode + 1) % 100 == 0:
                    torch.save(
                        policy_dqn.state_dict(),
                        os.path.join(RUNS_DIR, f"{self.param_set}_latest.pt"),
                    )

                if is_training and len(memory) >= self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                    if steps >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0
        finally:
            env.close()
            if env.observation_values:
                print(
                    "Observation range enforcement: clipped "
                    f"{env.clipped_values} of {env.observation_values} values "
                    "to the declared Box bounds."
                )

        if is_training:
            torch.save(
                policy_dqn.state_dict(),
                os.path.join(RUNS_DIR, f"{self.param_set}_latest.pt"),
            )
            print("Latest checkpoint saved to runs/")

        if not is_training:
            average_reward = sum(evaluation_rewards) / len(evaluation_rewards)
            variance = sum(
                (reward - average_reward) ** 2 for reward in evaluation_rewards
            ) / len(evaluation_rewards)
            output_file = os.path.join(
                RUNS_DIR, f"{self.param_set}_evaluation_seed_{seed}.csv"
            )
            with open(output_file, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["episode", "seed", "reward"])
                writer.writerows(
                    (index, seed + index - 1, reward)
                    for index, reward in enumerate(evaluation_rewards, start=1)
                )

            print("\n========== EVALUATION SUMMARY ==========")
            print(f"Episodes: {len(evaluation_rewards)}")
            print(f"Seed range: {seed}–{seed + episodes - 1}")
            print(f"Mean reward: {average_reward:.3f}")
            print(f"Population standard deviation: {variance ** 0.5:.3f}")
            print(f"Maximum reward: {max(evaluation_rewards):.3f}")
            print(f"Minimum reward: {min(evaluation_rewards):.3f}")
            print(f"Results saved to {output_file}")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float32, device=device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(
                next_states
            ).max(dim=1)[0]

        current_q = policy_dqn(states).gather(
            dim=1, index=actions.unsqueeze(dim=1)
        ).squeeze()
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate the assigned Flappy Bird DQN."
    )
    parser.add_argument("hyperparameters", help="Key from parameters.yaml")
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episodes to train/evaluate (default: 20,000 training; 100 evaluation)")
    parser.add_argument("--max-steps", type=int, default=10_000,
                        help="Maximum steps in one episode to prevent unbounded loops")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    episodes = args.episodes if args.episodes is not None else (20_000 if args.train else 100)
    if episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive.")

    agent = Agent(param_set=args.hyperparameters)
    agent.run(
        is_training=args.train,
        render=args.render,
        seed=args.seed,
        episodes=episodes,
        max_steps_per_episode=args.max_steps,
    )
