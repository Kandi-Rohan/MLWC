"""Visualize RIS allocations on the Figure 2(d) antenna and surface sizes."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch

from ao_baseline import greedy_partition, optimize
from compare_policies import TRAINING_SETTINGS, add_experiment_arguments, train_actors
from compare_policies import validate_experiment_arguments
from isac_environment import Config, ISACEnvironment
from reproduction_pilot import heuristic_action


CONFIG = Config(rows=8, cols=8)
SETTINGS = "Faster training settings"
METHODS = ("Fixed partition + AO", "Greedy assignment + AO", "DDPG after training")
GROUP_LABELS = ("CU group 1", "CU group 2", "CU group 3", "Sensing")
COLORS = ("#386CB0", "#E08214", "#389273", "#9053A0")


def baselines(env):
    fixed, beam, phases = heuristic_action(env, shared=False)
    greedy, greedy_phases = greedy_partition(env, beam)
    return (fixed, beam, phases), (greedy, beam, greedy_phases)


def describe(grid, info):
    counts = np.bincount(grid.ravel(), minlength=len(GROUP_LABELS))
    weakest = np.min(info["sensing_w"]) / CONFIG.min_sensing_w
    return counts, weakest, info["sum_rate"], np.all(info["rates"] >= CONFIG.min_rate)


@torch.inference_mode()
def evaluate(actors, draws, iterations, eval_seed):
    env = ISACEnvironment(CONFIG, eval_seed)
    statistics = {name: [] for name in (*METHODS, "DDPG before training")}
    layouts = {name: set() for name in statistics}
    examples = []
    for draw in range(draws):
        state = torch.from_numpy(env.reset())
        evaluated = []
        for grid, beam, phases in baselines(env):
            beam, phases, _ = optimize(env, grid, beam, phases, iterations)
            evaluated.append((grid, env.evaluate(grid, beam, phases)))
        before, after = (env.evaluate_action(actor(state).numpy()) for actor in actors)
        evaluated.extend(((after["partition"], after), (before["partition"], before)))

        for method, (grid, info) in zip(statistics, evaluated):
            statistics[method].append(describe(grid, info))
            layouts[method].add(grid.tobytes())
        if draw < 2:
            examples.append([(grid, *describe(grid, info)[:2]) for grid, info in evaluated[:3]])
    return statistics, layouts, examples


def report(statistics, layouts):
    print("\n| Method | Distinct layouts | Mean sensing cells | Sensing cell range | "
          "Mean sum rate (bit/s/Hz) | All users pass | All targets pass | Joint pass |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, records in statistics.items():
        counts, weakest, rate, user_pass = zip(*records)
        sensing = np.asarray(counts)[:, -1]
        target_pass = np.asarray(weakest) >= 1
        print(f"| {name} | {len(layouts[name])} | {sensing.mean():.2f} | "
              f"{sensing.min()} to {sensing.max()} | {np.mean(rate):.6g} | {np.mean(user_pass):.1%} | "
              f"{target_pass.mean():.1%} | {np.mean(target_pass & user_pass):.1%} |")


def plot_layouts(path, examples):
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 7.1), sharex=True, sharey=True,
                             layout="constrained")
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(len(GROUP_LABELS) + 1) - 0.5, cmap.N)
    for draw, (row, example) in enumerate(zip(axes, examples), start=1):
        for method, ax, (grid, counts, weakest) in zip(METHODS, row, example):
            ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_xticks(np.arange(8), np.arange(1, 9))
            ax.set_yticks(np.arange(8), np.arange(1, 9))
            ax.set_xticks(np.arange(7) + 0.5, minor=True)
            ax.set_yticks(np.arange(7) + 0.5, minor=True)
            ax.grid(which="minor", color="white", alpha=0.38, linewidth=0.75)
            ax.tick_params(which="minor", bottom=False, left=False)
            if draw == 1:
                ax.set_title(method, fontsize=11)
            allocation = " / ".join(str(n) for n in counts)
            ax.set_xlabel(f"Cells by group  {allocation}\n"
                          f"Weakest target  {100 * weakest:.2g}% of required power", fontsize=9)
        row[0].set_ylabel(f"Held-out draw {draw}\nRIS row")
    handles = [Patch(facecolor=color, label=label) for label, color in zip(GROUP_LABELS, COLORS)]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False)
    fig.suptitle("RIS element assignments on two matched channels", fontsize=16, weight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.set_defaults(seeds=[0])
    parser.add_argument("--plot", type=Path, default=Path("partition_visualization.png"))
    args = parser.parse_args()
    validate_experiment_arguments(parser, args)
    if len(args.seeds) != 1 or args.draws < 2:
        parser.error("Use one training seed and at least two held-out draws.")
    seed = args.seeds[0]
    torch.set_num_threads(1)
    profiles = train_actors(CONFIG, [seed], args.steps,
                            training_settings={SETTINGS: TRAINING_SETTINGS[SETTINGS]})
    print(f"Evaluating {args.draws} held-out channels at 64 RIS elements", flush=True)
    statistics, layouts, examples = evaluate(
        profiles[SETTINGS][seed], args.draws, args.iterations, args.eval_seed)
    report(statistics, layouts)
    plot_layouts(args.plot, examples)


if __name__ == "__main__":
    main()
