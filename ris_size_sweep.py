"""Run a Figure 2(c)-style RIS-size sweep at 16 BS antennas."""

import argparse
from dataclasses import replace
from pathlib import Path

from array_sweep import add_sweep_arguments, plot_rates, run_sweep
from compare_policies import validate_experiment_arguments
from isac_environment import Config


RIS_SIDES = (3, 4, 5, 6, 7, 8, 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_sweep_arguments(parser)
    parser.add_argument("--ris-sides", type=int, nargs="+", default=RIS_SIDES)
    parser.add_argument("--plot", type=Path, default=Path("ris_size_sweep.png"))
    args = parser.parse_args()
    validate_experiment_arguments(parser, args)
    sides = sorted(set(args.ris_sides))
    if min(sides) < 2:
        parser.error("RIS side lengths must be at least two.")

    configs = [replace(Config(), rows=side, cols=side) for side in sides]
    results = run_sweep(configs, args, ris_shape=(sides[-1], sides[-1]))
    plot_rates(args.plot, [side ** 2 for side in sides], results,
               x_label="Number of RIS elements", title="RIS size sweep with 16 BS antennas")


if __name__ == "__main__":
    main()
