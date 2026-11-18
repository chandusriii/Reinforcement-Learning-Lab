"""Seeded, bounded Flappy Bird DQN training and greedy evaluation.

The original assigned agent.py is intentionally preserved. This runner adds
observation-bound enforcement, reproducible seeds, per-episode telemetry, and
full resumable training checkpoints for long experiments.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import flappy_bird_gymnasium  # noqa: F401 - registers the environment
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from bounded_env import ClipObservationBounds
from dqn import DQN
from experience_replay import ReplayMemory


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

RUNS_DIR = Path("runs")
CHECKPOINT_VERSION = 1


def split_step(result):
    """Normalize Gym/Gymnasium four- and five-value step APIs."""
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, bool(terminated), bool(truncated), info
    observation, reward, done, info = result
    truncated = (
        bool(info.get("TimeLimit.truncated", False))
        if isinstance(info, dict) else False
    )
    return observation, reward, bool(done) and not truncated, truncated, info


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_metrics_for_resume(path: Path, completed_episode: int, expected_seed: int):
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot resume: missing metrics file {path}. Keep it with the checkpoint."
        )
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames
        required = {"episode", "seed", "return", "steps"}
        if not header or not required.issubset(header):
            raise ValueError("Existing training CSV has an incompatible header.")
        rows = []
        for row in reader:
            if int(row["episode"]) <= completed_episode:
                rows.append(row)

    if completed_episode > 0 and not any(
        int(row["episode"]) == completed_episode for row in rows
    ):
        raise ValueError(
            "Checkpoint episode is not present in the CSV. Use matching files."
        )
    if any(int(row["seed"]) != expected_seed + int(row["episode"]) - 1 for row in rows):
        raise ValueError("Training CSV seed sequence does not match the requested seed.")

    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return header, rows


class Agent:
    def __init__(self, param_set: str):
        self.param_set = param_set
        with Path("parameters.yaml").open("r", encoding="utf-8") as file:
            configurations = yaml.safe_load(file)
        if param_set not in configurations:
            raise KeyError(f"Unknown parameter set: {param_set}")
        params = configurations[param_set]

        self.alpha = float(params["alpha"])
        self.gamma = float(params["gamma"])
        self.epsilon_init = float(params["epsilon_init"])
        self.epsilon_min = float(params["epsilon_min"])
        self.epsilon_decay = float(params["epsilon_decay"])
        self.replay_memory_size = int(params["replay_memory_size"])
        self.mini_batch_size = int(params["mini_batch_size"])
        self.reward_threshold = float(params["reward_threshold"])
        self.network_sync_rate = int(params["network_sync_rate"])
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.log_path = RUNS_DIR / f"{self.param_set}.log"
        self.best_model_path = RUNS_DIR / f"{self.param_set}.pt"
        self.latest_model_path = RUNS_DIR / f"{self.param_set}_latest.pt"
        self.resume_path = RUNS_DIR / f"{self.param_set}_resume.pt"

    def _save_resume_checkpoint(
        self, policy_dqn, target_dqn, optimizer, memory, episode,
        epsilon, steps_since_sync, total_env_steps, best_reward,
        best_policy_state, seed,
    ) -> None:
        checkpoint = {
            "format_version": CHECKPOINT_VERSION,
            "param_set": self.param_set,
            "seed": int(seed),
            "episode": int(episode),
            "policy": policy_dqn.state_dict(),
            "target": target_dqn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epsilon": float(epsilon),
            "steps_since_sync": int(steps_since_sync),
            "total_env_steps": int(total_env_steps),
            "best_reward": float(best_reward),
            "best_policy": best_policy_state,
            "replay_memory": memory.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temporary = self.resume_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, self.resume_path)

        # Weights-only latest snapshot remains convenient for inference.
        torch.save(policy_dqn.state_dict(), self.latest_model_path)

    def _load_resume_checkpoint(
        self, path, policy_dqn, target_dqn, optimizer, memory, requested_seed
    ):
        checkpoint = torch.load(
            path, map_location=DEVICE, weights_only=False
        )
        if checkpoint.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported Flappy training-checkpoint version.")
        if checkpoint.get("param_set") != self.param_set:
            raise ValueError("Checkpoint hyperparameter set does not match.")
        if int(checkpoint["seed"]) != int(requested_seed):
            raise ValueError("Resume seed does not match the checkpoint seed.")

        policy_dqn.load_state_dict(checkpoint["policy"])
        target_dqn.load_state_dict(checkpoint["target"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        memory.load_state_dict(checkpoint["replay_memory"])

        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])

        return (
            int(checkpoint["episode"]),
            float(checkpoint["epsilon"]),
            int(checkpoint["steps_since_sync"]),
            int(checkpoint["total_env_steps"]),
            float(checkpoint["best_reward"]),
            checkpoint["best_policy"],
        )

    def run(
        self,
        is_training: bool = True,
        render: bool = False,
        seed: int = 42,
        episodes: int = 100,
        max_steps_per_episode: int = 10_000,
        checkpoint_every: int = 500,
        resume: bool = False,
    ):
        if episodes < 1 or max_steps_per_episode < 1:
            raise ValueError("Episode count and maximum steps must be positive.")
        if checkpoint_every < 1:
            raise ValueError("Checkpoint interval must be positive.")

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        seed_everything(seed)
        raw_env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None,
            use_lidar=True,
            normalize_obs=True,
            disable_env_checker=True,
        )
        env = ClipObservationBounds(raw_env)
        env.action_space.seed(seed)

        num_states = int(env.observation_space.shape[0])
        num_actions = int(env.action_space.n)
        policy_dqn = DQN(num_states, num_actions).to(DEVICE)
        self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

        training_file = None
        training_path = RUNS_DIR / f"{self.param_set}_training_seed_{seed}.csv"

        if is_training:
            memory = ReplayMemory(self.replay_memory_size, seed=seed)
            target_dqn = DQN(num_states, num_actions).to(DEVICE)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            epsilon = self.epsilon_init
            steps_since_sync = 0
            total_env_steps = 0
            best_reward = float("-inf")
            best_policy_state = None
            completed_episode = 0

            if resume:
                if not self.resume_path.is_file():
                    raise FileNotFoundError(
                        f"Resume checkpoint not found: {self.resume_path}"
                    )
                (
                    completed_episode, epsilon, steps_since_sync,
                    total_env_steps, best_reward, best_policy_state,
                ) = self._load_resume_checkpoint(
                    self.resume_path, policy_dqn, target_dqn,
                    self.optimizer, memory, seed,
                )
                if best_policy_state is None:
                    raise ValueError("Resume checkpoint is missing the best-policy weights.")
                # Restore the independent best-policy snapshot without replacing
                # the latest online weights used to continue training.
                torch.save(best_policy_state, self.best_model_path)
                header, _ = read_metrics_for_resume(
                    training_path, completed_episode, seed
                )
                training_file = training_path.open(
                    "a", newline="", encoding="utf-8"
                )
                training_writer = csv.writer(training_file)
                start_episode = completed_episode + 1
            else:
                header = [
                    "episode", "seed", "return", "steps", "total_env_steps",
                    "epsilon", "mean_loss", "clipped_observation_values",
                ]
                training_file = training_path.open(
                    "w", newline="", encoding="utf-8"
                )
                training_writer = csv.writer(training_file)
                training_writer.writerow(header)
                training_file.flush()
                start_episode = 1

            episode_range = range(start_episode, start_episode + episodes)
        else:
            if not self.best_model_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.best_model_path}. Train first."
                )
            policy_dqn.load_state_dict(
                torch.load(
                    self.best_model_path, map_location=DEVICE, weights_only=True
                )
            )
            policy_dqn.eval()
            episode_range = range(1, episodes + 1)
            evaluation_rewards = []
            evaluation_steps = []

        session_start = time.time()
        try:
            for episode in episode_range:
                observation, _ = env.reset(seed=seed + episode - 1)
                state = torch.as_tensor(
                    observation, dtype=torch.float32, device=DEVICE
                )
                episode_reward = 0.0
                episode_steps = 0
                episode_losses = []

                while (
                    episode_steps < max_steps_per_episode
                    and episode_reward < self.reward_threshold
                ):
                    if is_training and random.random() < epsilon:
                        action = random.randrange(num_actions)
                    else:
                        with torch.no_grad():
                            action = int(
                                policy_dqn(state.unsqueeze(0)).argmax(dim=1).item()
                            )

                    next_observation, reward, terminated, truncated, _ = split_step(
                        env.step(action)
                    )
                    next_state = torch.as_tensor(
                        next_observation, dtype=torch.float32, device=DEVICE
                    )
                    episode_reward += float(reward)
                    episode_steps += 1

                    if is_training:
                        reward_tensor = torch.tensor(
                            float(reward), dtype=torch.float32, device=DEVICE
                        )
                        action_tensor = torch.tensor(
                            action, dtype=torch.long, device=DEVICE
                        )
                        memory.append((
                            state.detach(), action_tensor, next_state.detach(),
                            reward_tensor, bool(terminated),
                        ))
                        total_env_steps += 1
                        steps_since_sync += 1

                    state = next_state
                    if terminated or truncated:
                        break

                if is_training:
                    if len(memory) >= self.mini_batch_size:
                        batch = memory.sample(self.mini_batch_size)
                        loss = self.optimize(batch, policy_dqn, target_dqn)
                        episode_losses.append(loss)
                    if steps_since_sync >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps_since_sync = 0

                    mean_loss = (
                        sum(episode_losses) / len(episode_losses)
                        if episode_losses else 0.0
                    )
                    training_writer.writerow([
                        episode, seed + episode - 1,
                        round(episode_reward, 4), episode_steps,
                        total_env_steps, round(epsilon, 6),
                        round(mean_loss, 6), env.clipped_values,
                    ])
                    training_file.flush()

                    epsilon = max(
                        epsilon * self.epsilon_decay, self.epsilon_min
                    )
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        best_policy_state = {
                            key: value.detach().cpu().clone()
                            for key, value in policy_dqn.state_dict().items()
                        }
                        torch.save(best_policy_state, self.best_model_path)
                        with self.log_path.open("a", encoding="utf-8") as log:
                            log.write(
                                f"best reward = {episode_reward:.4f} "
                                f"for episode={episode}\n"
                            )

                    if (
                        episode % checkpoint_every == 0
                        or episode == start_episode + episodes - 1
                    ):
                        self._save_resume_checkpoint(
                            policy_dqn, target_dqn, self.optimizer, memory,
                            episode, epsilon, steps_since_sync, total_env_steps,
                            best_reward, best_policy_state, seed,
                        )
                else:
                    evaluation_rewards.append(episode_reward)
                    evaluation_steps.append(episode_steps)

                if episode == start_episode if is_training else episode == 1:
                    print(f"Started episode {episode}")
                if episode % 100 == 0 or (
                    not is_training and episode == episodes
                ):
                    print(
                        f"Episode {episode} | return={episode_reward:.2f} "
                        f"| steps={episode_steps} "
                        f"| epsilon={epsilon if is_training else 0.0:.4f}"
                    )

        except KeyboardInterrupt:
            print(
                "Training interrupted. The most recent complete checkpoint is "
                "preserved; resume from the last saved checkpoint."
            )
            raise
        finally:
            if training_file is not None:
                training_file.close()
            env.close()
            if env.observation_values:
                print(
                    "Observation range enforcement: clipped "
                    f"{env.clipped_values} of {env.observation_values} values."
                )

        if is_training:
            print(f"Training metrics: {training_path}")
            print(f"Resume checkpoint: {self.resume_path}")
            print(f"Latest weights: {self.latest_model_path}")
        else:
            output_file = RUNS_DIR / (
                f"{self.param_set}_evaluation_seed_{seed}.csv"
            )
            with output_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["episode", "seed", "reward", "steps"])
                writer.writerows(
                    (index, seed + index - 1, reward, length)
                    for index, (reward, length) in enumerate(
                        zip(evaluation_rewards, evaluation_steps), start=1
                    )
                )
            avg = float(np.mean(evaluation_rewards))
            std = float(np.std(evaluation_rewards))
            print(f"Episodes: {len(evaluation_rewards)}")
            print(f"Mean reward: {avg:.3f} ± {std:.3f} (population SD)")
            print(f"Mean steps: {float(np.mean(evaluation_steps)):.1f}")
            print(f"Evaluation CSV: {output_file}")

        print(f"Session elapsed seconds: {time.time() - session_start:.1f}")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(
            terminations, dtype=torch.float32, device=DEVICE
        )

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(
                next_states
            ).max(dim=1)[0]

        current_q = policy_dqn(states).gather(
            dim=1, index=actions.unsqueeze(1)
        ).squeeze(1)
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or evaluate the seeded Flappy Bird DQN."
    )
    parser.add_argument("hyperparameters", help="Key from parameters.yaml")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    args.episodes = (
        args.episodes if args.episodes is not None
        else (20_000 if args.train else 100)
    )
    if args.episodes < 1 or args.max_steps < 1 or args.checkpoint_every < 1:
        parser.error("Episodes, max steps, and checkpoint interval must be positive.")
    if args.resume and not args.train:
        parser.error("--resume is available only with --train.")
    return args


if __name__ == "__main__":
    args = parse_args()
    agent = Agent(args.hyperparameters)
    agent.run(
        is_training=args.train,
        render=args.render,
        seed=args.seed,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
