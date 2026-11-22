"""Shared evaluation and plotting for the BS and RIS array-size sweeps."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_policies import AO_NAMES, TRAINING_SETTINGS, add_experiment_arguments, train_actors
from imperfect_csi import evaluate_conditions
from isac_environment import ISACEnvironment


METHODS = (*AO_NAMES, "DDPG pilot")
CSI_NAMES = ("Perfect CSI", "1% per-link relative error energy")


def add_sweep_arguments(parser):
    add_experiment_arguments(parser)
    parser.set_defaults(steps=320)
    parser.add_argument("--noise-seed", type=int, default=200_011)


def run_sweep(configs, args, *, antenna_width=None, ris_shape=None):
    torch.set_num_threads(1)
    results = []
    key = "Faster training settings"
    settings = {key: TRAINING_SETTINGS[key]}
    for config in configs:
        print(f"\n{config.antennas} BS antennas, {config.rows * config.cols} RIS elements", flush=True)
        profiles = train_actors(config, args.seeds, args.steps, training_settings=settings)
        env = ISACEnvironment(config, args.eval_seed)
        scores, _, _ = evaluate_conditions(
            env, profiles, args.draws, args.iterations, 0.01, args.noise_seed,
            error_modes=(True,), antenna_width=antenna_width, ris_shape=ris_shape)
        results.append(scores)
        for label, condition in zip(CSI_NAMES, scores):
            for method, (rate, _, users, sensing, joint) in zip(METHODS, condition):
                print(f"{label} | {method} | {rate:.6g} bit/s/Hz | "
                      f"users {users:.1%} | sensing {sensing:.1%} | joint {joint:.1%}",
                      flush=True)
    return np.asarray(results)


def plot_rates(path, sizes, results, *, x_label, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, layout="constrained")
    styles = (("#005A8D", "o", "-"), ("#BD5916", "s", "-"),
              ("#257859", "^", "-"), ("#71468D", "D", "--"))
    for i, (ax, label) in enumerate(zip(axes, CSI_NAMES)):
        for j, (method, (color, marker, line)) in enumerate(zip(METHODS, styles)):
            ax.plot(sizes, results[:, i, j, 0], label=method, color=color,
                    marker=marker, linestyle=line, linewidth=1.8, markersize=4)
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_title(label)
        ax.set_xlabel(x_label)
        ax.grid(axis="y", alpha=0.22, which="both")
    axes[0].set_ylabel("Mean sum rate (bit/s/Hz, log scale)")
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
