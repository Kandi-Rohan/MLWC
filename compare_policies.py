"""Score DDPG and all three AO baselines on paired held-out channels."""

import argparse
from copy import deepcopy

import numpy as np
import torch

from ao_baseline import baseline_actions, metrics, optimize
from isac_environment import Config, ISACEnvironment
from train_ddpg import train


TRAINING_SETTINGS = {
    "Paper training settings": (1e-6, 1e-8),
    "Faster training settings": (3e-4, 0.005),
}
AO_NAMES = ("Fixed partition AO", "Shared RIS AO", "Greedy assignment AO")


def train_actors(config, seeds, steps):
    profiles = {}
    for label, (lr, tau) in TRAINING_SETTINGS.items():
        runs = {}
        for seed in seeds:
            print(f"Training {label}, seed {seed}, {steps} steps", flush=True)
            initial_actor = None

            def save_initial(step, agent):
                nonlocal initial_actor
                if step == 0:
                    initial_actor = deepcopy(agent.actor).eval()

            agent, rewards = train(config, seed, steps, lr, tau, on_step=save_initial)
            runs[seed] = (initial_actor, agent.actor.eval())
            print(f"Mean training reward over final 100 steps: {np.mean(rewards[-100:]):.6f}",
                  flush=True)
        profiles[label] = runs
    return profiles


@torch.inference_mode()
def score_policies(design_env, scoring_env, state, profiles, iterations):
    """Choose on design CSI and evaluate on the scoring CSI of the same draw."""
    state = torch.from_numpy(state)
    config = scoring_env.config
    ao_scores = []
    for grid, beam, phases in baseline_actions(design_env):
        beam, phases, _ = optimize(design_env, grid, beam, phases, iterations)
        ao_scores.append(metrics(config, scoring_env.evaluate(grid, beam, phases)))
    ddpg_scores = {
        label: {
            seed: [metrics(config, scoring_env.evaluate_action(actor(state).numpy()))
                   for actor in actors]
            for seed, actors in runs.items()
        }
        for label, runs in profiles.items()
    }
    return ao_scores, ddpg_scores


def paired_evaluation(config, profiles, draws, iterations, eval_seed):
    env = ISACEnvironment(config, eval_seed)
    ao_scores = []
    ddpg_scores = {
        label: {seed: [] for seed in runs}
        for label, runs in profiles.items()
    }
    upper_bound_pass = []
    for _ in range(draws):
        # Every method is scored before the environment advances to the next draw.
        state = env.reset()
        rate_bound, sensing_bound = env.link_upper_bounds()
        upper_bound_pass.append([np.all(rate_bound >= config.min_rate),
                                 np.all(sensing_bound >= config.min_sensing_w)])
        ao_draw, ddpg_draw = score_policies(env, env, state, profiles, iterations)
        ao_scores.append(ao_draw)

        for label, runs in ddpg_draw.items():
            for seed, scores in runs.items():
                ddpg_scores[label][seed].append(scores)
    return np.asarray(ao_scores), ddpg_scores, np.asarray(upper_bound_pass)


def print_row(label, scores):
    rate, reward, users, sensing, joint = scores
    print(f"| {label} | {rate:.6g} | {reward:.6f} | {users:.1%} | "
          f"{sensing:.1%} | {joint:.1%} |")


def add_experiment_arguments(parser):
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--eval-seed", type=int, default=100_000)


def validate_experiment_arguments(parser, args):
    if min(args.steps, args.draws, args.iterations) < 1:
        parser.error("Steps, draws, and AO iterations must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or args.eval_seed in args.seeds:
        parser.error("Use distinct training seeds and a separate evaluation seed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    args = parser.parse_args()
    validate_experiment_arguments(parser, args)
    torch.set_num_threads(1)
    config = Config()
    profiles = train_actors(config, args.seeds, args.steps)
    print(f"Evaluating {args.draws} paired channels with seed {args.eval_seed}", flush=True)
    ao_scores, ddpg_scores, bounds = paired_evaluation(
        config, profiles, args.draws, args.iterations, args.eval_seed)

    print("\n| Policy | Mean sum rate (bit/s/Hz) | Mean reward | All users pass | "
          "All targets pass | Joint pass |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label, scores in zip(AO_NAMES, ao_scores.mean(axis=0)):
        print_row(label, scores)
    for label, runs in ddpg_scores.items():
        scores = np.asarray(list(runs.values()))
        starts, finals = scores.mean(axis=(0, 1))
        print_row(f"{label}, untrained", starts)
        for seed, seed_scores in runs.items():
            print_row(f"{label}, seed {seed}", np.mean(seed_scores, axis=0)[1])
        print_row(f"{label}, final mean across seeds", finals)

    user_bound, sensing_bound = bounds.mean(axis=0)
    print(f"All individual upper bounds pass for users on {user_bound:.1%} of draws, "
          f"targets on {sensing_bound:.1%}, and both on {np.all(bounds, axis=1).mean():.1%}.")


if __name__ == "__main__":
    main()
