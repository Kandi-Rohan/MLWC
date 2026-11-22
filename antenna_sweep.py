"""Run a Figure 2(b)-style BS antenna sweep with a fixed 36-element RIS."""

import argparse
from dataclasses import replace
from pathlib import Path

from array_sweep import add_sweep_arguments, plot_rates, run_sweep
from compare_policies import validate_experiment_arguments
from isac_environment import Config


ANTENNAS = (4, 8, 12, 16, 20, 24, 28, 32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_sweep_arguments(parser)
    parser.add_argument("--antennas", type=int, nargs="+", default=ANTENNAS)
    parser.add_argument("--plot", type=Path, default=Path("antenna_sweep.png"))
    args = parser.parse_args()
    validate_experiment_arguments(parser, args)
    antenna_counts = sorted(set(args.antennas))
    if min(antenna_counts) < 1:
        parser.error("Antenna counts must be positive.")

    configs = [replace(Config(), antennas=count) for count in antenna_counts]
    results = run_sweep(configs, args, antenna_width=antenna_counts[-1])
    plot_rates(args.plot, antenna_counts, results, x_label="Number of BS antennas",
               title="BS antenna sweep with a 36-element RIS")


if __name__ == "__main__":
    main()
