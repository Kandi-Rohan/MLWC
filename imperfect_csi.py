"""Compare perfect and noisy CSI decisions on the same true channel draws."""

import argparse
from copy import copy

import numpy as np
import torch

from compare_policies import (AO_NAMES, add_experiment_arguments, print_row,
                              score_policies, train_actors, validate_experiment_arguments)
from isac_environment import Config, ISACEnvironment


LINK_NAMES = ("bs_ris", "ris_users", "ris_targets", "direct")
CONDITIONS = ("Perfect CSI", "Raw additive error", "Per-link relative error")


def noisy_channel_view(true_env, errors, variance, relative):
    """Apply complex Gaussian errors in raw units or after per-link normalization."""
    estimated = copy(true_env)
    error_energy = []
    for name, standard_error in zip(LINK_NAMES, errors):
        channel = getattr(true_env, name)
        if relative:
            # Normalize the full BS-RIS link, and each separate user or target link.
            axis = None if name == "bs_ris" else -1
            rms = np.sqrt(np.mean(np.abs(channel)**2, axis=axis, keepdims=True))
        else:
            rms = 1
        error = np.sqrt(variance) * rms * standard_error
        setattr(estimated, name, channel + error)
        error_energy.append(np.vdot(error, error).real)
    state = estimated._observe()
    # Keep the association decided from the estimate when propagating on true CSI.
    scoring = copy(true_env)
    scoring.user_group = estimated.user_group
    return estimated, scoring, state, error_energy


def evaluate_conditions(env, profiles, draws, iterations, variance, noise_seed):
    rng = np.random.default_rng(noise_seed)
    scores = []
    channel_energy = []
    error_energy = []
    group_changes = []
    for _ in range(draws):
        state = env.reset()
        channels = [getattr(env, name) for name in LINK_NAMES]
        # Reuse the same standard complex draws in both noisy conditions.
        errors = [(rng.normal(size=h.shape) + 1j * rng.normal(size=h.shape)) / np.sqrt(2)
                  for h in channels]
        channel_energy.append([np.vdot(h, h).real for h in channels])
        conditions = [(env, env, state, [0] * len(channels))]
        conditions.extend(noisy_channel_view(env, errors, variance, relative)
                          for relative in (False, True))
        group_changes.append([not np.array_equal(design.user_group, env.user_group)
                              for design, _, _, _ in conditions])
        error_energy.append([energy for _, _, _, energy in conditions])
        draw_scores = []
        for design, scoring, observation, _ in conditions:
            ao, ddpg = score_policies(design, scoring, observation, profiles, iterations)
            final_ddpg = [np.mean([pair[1] for pair in runs.values()], axis=0)
                          for runs in ddpg.values()]
            draw_scores.append(ao + final_ddpg)
        scores.append(draw_scores)
    return (np.mean(scores, axis=0), np.sum(error_energy, axis=0) / np.sum(channel_energy, axis=0),
            np.mean(group_changes, axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--variance", type=float, default=0.01)
    parser.add_argument("--noise-seed", type=int, default=200_011)
    args = parser.parse_args()
    validate_experiment_arguments(parser, args)
    if not np.isfinite(args.variance) or args.variance < 0:
        parser.error("Variance must be finite and nonnegative.")
    torch.set_num_threads(1)
    config = Config()
    profiles = train_actors(config, args.seeds, args.steps)
    print(f"Evaluating {args.draws} paired channels with variance {args.variance:g}", flush=True)
    env = ISACEnvironment(config, args.eval_seed)
    scores, nmse, group_changes = evaluate_conditions(
        env, profiles, args.draws, args.iterations, args.variance, args.noise_seed)
    methods = (*AO_NAMES, *(f"DDPG, {label.lower()}" for label in profiles))

    for label, condition, mismatch in zip(CONDITIONS, scores, group_changes):
        print(f"\n{label} | Group labels changed on {mismatch:.1%} of draws")
        print("| Policy | Mean sum rate (bit/s/Hz) | Mean reward | All users pass | "
              "All targets pass | Joint pass |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for method, result in zip(methods, condition):
            print_row(method, result)
    print("\nSquared estimation error divided by true channel energy:")
    for label, errors in zip(CONDITIONS[1:], nmse[1:]):
        print(f"{label}: " + ", ".join(f"{name}={value:.3g}"
                                       for name, value in zip(LINK_NAMES, errors)))


if __name__ == "__main__":
    main()
