"""Train a geometry-aware DDPG agent and evaluate on held-out CSI draws."""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from isac_environment import Config, ISACEnvironment


def network(input_size, output_size):
    return nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, output_size))


class DDPG:
    def __init__(self, state_size, action_size, capacity, seed, lr, tau):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.actor = nn.Sequential(network(state_size, action_size), nn.Tanh())
        self.critic = network(state_size + action_size, 1)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.tau = tau
        self.replay = np.empty((capacity, 2 * state_size + action_size + 1), dtype=np.float32)
        self.state_size = state_size
        self.action_size = action_size
        self.seen = 0

    @torch.no_grad()
    def act(self, state):
        return self.actor(torch.from_numpy(state)).numpy()

    def remember(self, state, action, reward, next_state):
        self.replay[self.seen % len(self.replay)] = np.concatenate(
            (state, action, [reward], next_state))
        self.seen += 1

    def update(self, batch_size):
        indices = self.rng.choice(min(self.seen, len(self.replay)), batch_size, replace=False)
        batch = torch.from_numpy(self.replay[indices])
        states, actions, rewards, next_states = batch.split(
            (self.state_size, self.action_size, 1, self.state_size), dim=1)
        with torch.no_grad():
            next_inputs = torch.cat((next_states, self.target_actor(next_states)), dim=1)
            targets = rewards + 0.99 * self.target_critic(next_inputs)

        self.critic_optimizer.zero_grad()
        predictions = self.critic(torch.cat((states, actions), dim=1))
        nn.functional.mse_loss(predictions, targets).backward()
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        self.actor_optimizer.zero_grad()
        policy_inputs = torch.cat((states, self.actor(states)), dim=1)
        (-self.critic(policy_inputs).mean()).backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        with torch.no_grad():
            for target, source in ((self.target_actor, self.actor),
                                   (self.target_critic, self.critic)):
                for target_weight, weight in zip(target.parameters(), source.parameters()):
                    target_weight.lerp_(weight, self.tau)


def evaluate(agent, config, seed, trials, random_policy=False):
    # Restart the same CSI sequence for each policy for a paired comparison.
    env = ISACEnvironment(config, seed)
    state = env.reset()
    rng = np.random.default_rng(seed)
    measurements = []
    for _ in range(trials):
        action = rng.uniform(-1, 1, env.action_size) if random_policy else agent.act(state)
        rate_bound, sensing_bound = env.link_upper_bounds()
        state, reward, info = env.step(action)
        rate_ok = np.all(info["rates"] >= config.min_rate)
        sensing_ok = np.all(info["sensing_w"] >= config.min_sensing_w)
        rate_impossible = np.any(rate_bound < config.min_rate)
        sensing_impossible = np.any(sensing_bound < config.min_sensing_w)
        measurements.append((reward, info["sum_rate"], rate_ok, sensing_ok, rate_ok and sensing_ok,
                             rate_impossible, sensing_impossible))
    return np.mean(measurements, axis=0)


def train(config, seed, steps, lr, tau, progress=False):
    env = ISACEnvironment(config, seed)
    state = env.reset()
    agent = DDPG(state.size, env.action_size, min(steps, 10000),
                 seed, lr, tau)
    rng = np.random.default_rng(seed)
    rewards = []
    for step in range(steps):
        if step < 128:
            action = rng.uniform(-1, 1, env.action_size)
        else:
            action = np.clip(agent.act(state) + rng.normal(0, 0.2, env.action_size), -1, 1)
        next_state, reward, _ = env.step(action)
        agent.remember(state, action, reward, next_state)
        state = next_state
        rewards.append(reward)
        if agent.seen >= 64:
            agent.update(64)
        if progress and (step + 1) % 100 == 0:
            print(f"Step {step + 1}: mean reward {np.mean(rewards[-100:]):.4f}", flush=True)
    return agent, rewards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--antennas", type=int, default=16)
    parser.add_argument("--ris-side", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--tau", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=Path, default=Path("run"))
    args = parser.parse_args()
    if args.steps < 1 or args.eval_trials < 1:
        parser.error("--steps and --eval-trials must be positive")
    torch.set_num_threads(1)
    config = Config(antennas=args.antennas, rows=args.ris_side, cols=args.ris_side)
    agent, rewards = train(config, args.seed, args.steps, args.lr, args.tau, progress=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "rewards.csv", rewards, header="reward", comments="")
    torch.save(agent.actor.state_dict(), args.output_dir / "actor.pt")
    for label, random_policy in (("DDPG", False), ("Random action", True)):
        reward, rate, rate_ok, sensing_ok, joint_ok, rate_impossible, sensing_impossible = evaluate(
            agent, config, args.seed + 100_000, args.eval_trials, random_policy)
        print(f"{label}: reward={reward:.4f}, sum rate={rate:.4f} bit/s/Hz, "
              f"all users pass={rate_ok:.1%}, all targets pass={sensing_ok:.1%}, "
              f"joint pass={joint_ok:.1%}")
    print(f"Upper bounds below threshold for at least one user in {rate_impossible:.1%} "
          f"of draws and at least one target in {sensing_impossible:.1%} of draws.")


if __name__ == "__main__":
    main()


