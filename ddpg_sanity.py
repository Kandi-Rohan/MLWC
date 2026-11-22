"""Run the DDPG pipeline on a controlled, demonstrably feasible channel."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from isac_environment import Config, ISACEnvironment
from train_ddpg import train


CONTROLLED = Config(antennas=3, rows=2, cols=3, users=2, targets=1, groups=2)
EVAL_SEED = 20261122


class FeasibleEnvironment(ISACEnvironment):
    """Two orthogonal user paths and a direct target, with varying fading."""

    def reset(self):
        # Each column of the RIS can steer a different BS antenna direction.
        self.bs_ris = 8e-4 * np.tile(np.eye(3, dtype=complex), (2, 1))
        strengths = self.rng.uniform(0.9, 1.1, 2)
        self.offset = self.rng.choice([-np.pi / 4, 0.0, np.pi / 4], size=2)
        self.ris_users = np.zeros((2, 6), dtype=complex)
        for k in range(2):
            self.ris_users[k, [k, k + 3]] = 1.3e-3 * strengths[k] * np.exp(
                1j * np.array([0, self.offset[k]]))
        self.ris_targets = np.zeros((1, 6), dtype=complex)
        self.direct = self.rng.uniform(1.35e-3, 1.65e-3) * np.array(
            [[1, 1, 0]], dtype=complex) / np.sqrt(2)
        # Controlled links are stronger, so scale the synthetic CSI near unit magnitude.
        return self._observe() * np.float32(0.1)

    def feasible_action(self):
        """Align the two RIS cells per user and split power between their beams."""
        geometry = np.array([[0.5, 1/6, 1, 1/3, 0.5],
                             [0.5, 1/2, 1, 1/3, 0.5],
                             [0.5, 5/6, 1, 1/3, 0.5]])
        beam = np.diag([np.sqrt(self.config.power_w / 2)] * 2 + [0])
        phases = np.zeros(6)
        phases[3:5] = np.mod(self.offset, 2 * np.pi)
        return np.concatenate((2 * geometry.ravel() - 1, beam.ravel(),
                               np.zeros(beam.size), phases / np.pi - 1))


def evaluate(policy, draws):
    env = FeasibleEnvironment(CONTROLLED, EVAL_SEED)
    state = env.reset()
    samples = []
    for _ in range(draws):
        state, reward, info = env.step(policy(env, state))
        users = np.all(info["rates"] >= CONTROLLED.min_rate)
        sensing = np.all(info["sensing_w"] >= CONTROLLED.min_sensing_w)
        samples.append((reward, info["sum_rate"], users, sensing, users and sensing))
    return np.mean(samples, axis=0)


def print_metrics(name, metrics):
    reward, rate, users, sensing, joint = metrics
    print(f"| {name} | {reward:.4f} | {rate:.4f} | "
          f"{users:.1%} | {sensing:.1%} | {joint:.1%} |")


def plot_checkpoints(path, checkpoints, seed_histories, reference):
    history = np.asarray(seed_histories)
    fig, (reward_ax, pass_ax) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                             layout="constrained")
    for ax, field, scale in ((reward_ax, 0, 1), (pass_ax, 4, 100)):
        values = history[:, :, field] * scale
        for seed_values in values:
            ax.plot(checkpoints, seed_values, lw=1, color="#4C78A8", alpha=0.28)
        ax.plot(checkpoints, values.mean(axis=0), lw=2.4, color="#245B8A",
                label="DDPG mean across seeds")
        ax.axhline(reference[field] * scale, color="#B45817", ls="--",
                   label="Constructed feasible action")
        ax.grid(alpha=0.22)
    reward_ax.legend(loc="lower right")
    reward_ax.set_title("DDPG on the controlled channel")
    reward_ax.set_ylabel("Held-out mean reward")
    pass_ax.set_ylabel("Joint threshold pass (%)")
    pass_ax.set_xlabel("Training steps")
    pass_ax.set_ylim(-4, 104)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--every", type=int, default=200)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--plot", type=Path, default=Path("ddpg_sanity.png"))
    args = parser.parse_args()
    if min(args.steps, args.every, args.draws) < 1:
        parser.error("Steps, checkpoint interval and draw count must be positive.")
    torch.set_num_threads(1)
    reference = evaluate(lambda env, state: env.feasible_action(), args.draws)
    if reference[4] != 1:
        raise RuntimeError("The constructed action failed the held-out feasibility check")
    rng = np.random.default_rng(EVAL_SEED + 1)
    random = evaluate(lambda env, state: rng.uniform(-1, 1, env.action_size), args.draws)
    checkpoints = sorted(set(range(0, args.steps + 1, args.every)) | {args.steps})
    histories = []
    for seed in args.seeds:
        history = []

        def on_step(step, agent):
            if step in checkpoints:
                stats = evaluate(lambda env, state: agent.act(state), args.draws)
                history.append(stats)
                print(f"Seed {seed}, step {step:4}: reward {stats[0]:.3f}, "
                      f"joint pass {stats[4]:.1%}", flush=True)

        train(CONTROLLED, seed, args.steps, args.lr, args.tau,
              environment=FeasibleEnvironment, on_step=on_step)
        histories.append(history)

    history = np.asarray(histories)
    print("\n| Policy | Reward | Sum rate (bit/s/Hz) | All users pass | Sensing passes | Joint pass |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    print_metrics("Random action", random)
    print_metrics("Constructed feasible action", reference)
    for seed, seed_history in zip(args.seeds, history):
        print_metrics(f"Seed {seed}, untrained", seed_history[0])
        print_metrics(f"Seed {seed}, final", seed_history[-1])
    print_metrics("DDPG final, mean across seeds", history[:, -1].mean(axis=0))
    plot_checkpoints(args.plot, checkpoints, history, reference)


if __name__ == "__main__":
    main()
