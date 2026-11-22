"""Run a Figure 2(b)-style BS antenna sweep with a fixed 36-element RIS."""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_policies import (AO_NAMES, TRAINING_SETTINGS, add_experiment_arguments,
                              validate_experiment_arguments, train_actors)
from imperfect_csi import evaluate_conditions
from isac_environment import Config, ISACEnvironment


ANTENNAS = (4, 8, 12, 16, 20, 24, 28, 32)
METHODS = (*AO_NAMES, "DDPG pilot")
CSI_NAMES = ("Perfect CSI", "1% per-link relative error energy")


def plot_rates(path, antenna_counts, results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, layout="constrained")
    styles = (("#005A8D", "o", "-"), ("#BD5916", "s", "-"),
              ("#257859", "^", "-"), ("#71468D", "D", "--"))
    for i, (ax, label) in enumerate(zip(axes, CSI_NAMES)):
        for j, (method, (color, marker, line)) in enumerate(zip(METHODS, styles)):
            ax.plot(antenna_counts, results[:, i, j, 0], label=method, color=color,
                    marker=marker, linestyle=line, linewidth=1.8, markersize=4)
        ax.set_yscale("log")
        ax.set_xticks(antenna_counts)
        ax.set_title(label)
        ax.set_xlabel("Number of BS antennas")
        ax.grid(axis="y", alpha=0.22, which="both")
    axes[0].set_ylabel("Mean sum rate (bit/s/Hz, log scale)")
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle("BS antenna sweep with a 36-element RIS")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.set_defaults(steps=320)
    parser.add_argument("--antennas", type=int, nargs="+", default=ANTENNAS)
    parser.add_argument("--noise-seed", type=int, default=200_011)
    parser.add_argument("--plot", type=Path, default=Path("antenna_sweep.png"))
    args = parser.parse_args()
    validate_experiment_arguments(parser, args)
    antenna_counts = sorted(set(args.antennas))
    if min(antenna_counts) < 1:
        parser.error("Antenna counts must be positive.")

    torch.set_num_threads(1)
    rates_and_thresholds = []
    settings = {"Faster training settings": TRAINING_SETTINGS["Faster training settings"]}
    for antennas in antenna_counts:
        config = replace(Config(), antennas=antennas)
        print(f"\n{antennas} BS antennas", flush=True)
        profiles = train_actors(config, args.seeds, args.steps, training_settings=settings)
        env = ISACEnvironment(config, args.eval_seed)
        scores, _, _ = evaluate_conditions(
            env, profiles, args.draws, args.iterations, 0.01, args.noise_seed,
            error_modes=(True,), antenna_width=max(antenna_counts))
        rates_and_thresholds.append(scores)
        for label, condition in zip(CSI_NAMES, scores):
            for method, (rate, _, users, sensing, joint) in zip(METHODS, condition):
                print(f"{label} | {method} | {rate:.6g} bit/s/Hz | "
                      f"users {users:.1%} | sensing {sensing:.1%} | joint {joint:.1%}",
                      flush=True)

    plot_rates(args.plot, antenna_counts, np.asarray(rates_and_thresholds))


if __name__ == "__main__":
    main()
