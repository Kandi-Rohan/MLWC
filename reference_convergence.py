"""Check DDPG reward trajectories on the original 36-element channel model."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_policies import TRAINING_SETTINGS
from isac_environment import Config
from train_ddpg import evaluate, train


def plot_rewards(path, training, held_out, checkpoints):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True,
                             layout="constrained")
    training = np.asarray(training)
    starts = np.arange(0, training.shape[1], 100)
    ends = np.minimum(starts + 100, training.shape[1])
    means = np.column_stack([training[:, start:end].mean(axis=1)
                             for start, end in zip(starts, ends)])
    values = (means, held_out[:, :, 0])
    for ax, x, curves, title in zip(
            axes, (ends, checkpoints), values,
            ("Training, 100-step blocks", "Held-out evaluation")):
        for curve in curves:
            ax.plot(x, curve, lw=1, color="#4C78A8", alpha=0.28)
        ax.plot(x, curves.mean(axis=0), lw=2.2, color="#245B8A", label="Mean across seeds")
        ax.axhline(-30, color="#777777", ls="--", lw=1, label="Reward of −30")
        ax.set_title(title)
        ax.set_xlabel("Training steps")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean reward (zoomed scale)")
    axes[0].legend(fontsize=8)
    fig.suptitle("DDPG reward on the 36-element reference channel")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--every", type=int, default=400)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--plot", type=Path, default=Path("reference_convergence.png"))
    args = parser.parse_args()
    if min(args.steps, args.every, args.draws) < 1:
        parser.error("Steps, checkpoint interval, and draws must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or args.eval_seed in args.seeds:
        parser.error("Use distinct training seeds and a separate evaluation seed.")

    torch.set_num_threads(1)
    config = Config()
    lr, tau = TRAINING_SETTINGS["Paper training settings"]
    checkpoints = set(range(0, args.steps + 1, args.every)) | {args.steps}
    training = []
    held_out = []
    for seed in args.seeds:
        history = []

        def on_step(step, agent):
            if step in checkpoints:
                result = evaluate(agent, config, args.eval_seed, args.draws)
                history.append(result)
                print(f"Seed {seed}, step {step:4} | reward {result[0]:.9f} | "
                      f"sum rate {result[1]:.6g} | joint pass {result[4]:.1%}", flush=True)

        _, rewards = train(config, seed, args.steps, lr, tau, on_step=on_step)
        training.append(rewards)
        held_out.append(history)

    held_out = np.asarray(held_out)
    print("\n| Checkpoint | Mean held-out reward | Mean sum rate (bit/s/Hz) | Joint pass |")
    print("| ---: | ---: | ---: | ---: |")
    for step, mean in zip(sorted(checkpoints), held_out.mean(axis=0)):
        print(f"| {step} | {mean[0]:.9f} | {mean[1]:.6g} | {mean[4]:.1%} |")
    plot_rewards(args.plot, training, held_out, sorted(checkpoints))


if __name__ == "__main__":
    main()
