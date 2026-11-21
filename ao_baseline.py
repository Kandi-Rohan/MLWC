"""Interference-aware alternating optimization for fixed and shared RIS baselines."""

import argparse
from dataclasses import replace

import numpy as np

from isac_environment import Config, ISACEnvironment
from reproduction_pilot import heuristic_action


IMPROVEMENT_TOL = 1e-12


def beam_gradient(env, user_channels, target_channels, beam):
    """Derivative of Eq. (8)'s feasible-power objective with respect to conj(beam)."""
    c = env.config
    user_streams = user_channels @ beam
    target_streams = target_channels @ beam
    rates, sensing, signal, interference, score = env.stream_metrics(
        user_streams, target_streams)
    noise_plus_interference = interference + c.noise_w
    total = noise_plus_interference + signal
    users = np.arange(c.users)
    slope = (-(signal / noise_plus_interference) / total)[:, None] * user_streams
    slope[users, users] = user_streams[users, users] / total
    slope *= ((1 + 10 * (rates < c.min_rate)) / np.log(2))[:, None]
    sensing_slope = 10 * (sensing < c.min_sensing_w)[:, None] * target_streams
    gradient = user_channels.conj().T @ slope + target_channels.conj().T @ sensing_slope
    return gradient, score


def update_beam(env, channels, beam):
    """Take one projected ascent step with a backtracking line search."""
    users, targets = channels
    gradient, score = beam_gradient(env, users, targets, beam)
    norm = np.linalg.norm(gradient)
    if norm == 0:
        return beam
    radius = np.sqrt(env.config.power_w)
    direction = gradient / norm
    for step in radius * 0.5 ** np.arange(16):
        candidate = beam + step * direction
        candidate /= max(1, np.linalg.norm(candidate) / radius)
        candidate_score = env.stream_metrics(users @ candidate, targets @ candidate)[-1]
        if candidate_score > score + IMPROVEMENT_TOL:
            return candidate
    return beam


def update_phases(env, links, beam, phases, levels):
    """Sweep the elements and choose the best discrete phase at each coordinate."""
    user_links, target_links = links
    coefficients = np.exp(1j * phases)
    element_streams = env.bs_ris @ beam
    reflected = coefficients[:, None] * element_streams
    user_streams = user_links @ reflected
    target_streams = env.direct.conj() @ beam + target_links @ reflected
    candidates = np.exp(1j * levels)
    for m, coeff in enumerate(coefficients):
        if not np.any(user_links[:, m]) and not np.any(target_links[:, m]):
            continue
        delta = (candidates - coeff)[:, None, None]
        user_trials = user_streams + delta * np.outer(user_links[:, m], element_streams[m])
        target_trials = target_streams + delta * np.outer(target_links[:, m], element_streams[m])
        scores = env.stream_metrics(user_trials, target_trials)[-1]
        best = np.argmax(scores)
        current = int(np.rint(phases[m] / (2 * np.pi) * len(levels))) % len(levels)
        if scores[best] > scores[current] + IMPROVEMENT_TOL:
            phases[m] = levels[best]
            user_streams, target_streams = user_trials[best], target_trials[best]
    return phases


def optimize(env, grid, beam, phases, iterations=6):
    """Alternate beam and phase updates while retaining a monotone reward history."""
    levels = 2 * np.pi * np.arange(2 ** env.config.bits) / 2 ** env.config.bits
    user_links, target_links = env.reflected_links(grid)
    history = [env.evaluate(grid, beam, phases)["reward"]]
    for _ in range(iterations):
        reflected = np.exp(1j * phases)[:, None] * env.bs_ris
        channels = (user_links @ reflected, env.direct.conj() + target_links @ reflected)
        candidate_beam = update_beam(env, channels, beam)
        candidate_phases = update_phases(
            env, (user_links, target_links), candidate_beam, phases.copy(), levels)
        score = env.evaluate(grid, candidate_beam, candidate_phases)["reward"]
        if score <= history[-1] + IMPROVEMENT_TOL:
            break
        beam, phases = candidate_beam, candidate_phases
        history.append(score)
    return beam, phases, np.asarray(history)


def metrics(config, info):
    users_pass = np.all(info["rates"] >= config.min_rate)
    targets_pass = np.all(info["sensing_w"] >= config.min_sensing_w)
    return [info["sum_rate"], info["reward"], users_pass, targets_pass, users_pass and targets_pass]


def compare(side, draws, iterations, seed):
    config = replace(Config(), rows=side, cols=side)
    env = ISACEnvironment(config, seed)
    samples = []
    for _ in range(draws):
        env.reset()
        outcomes = []
        for shared in (False, True):
            grid, beam, phases = heuristic_action(env, shared)
            outcomes.append(metrics(config, env.evaluate(grid, beam, phases)))
            beam, phases, _ = optimize(env, grid, beam, phases, iterations)
            outcomes.append(metrics(config, env.evaluate(grid, beam, phases)))
        samples.append(outcomes)
    result = np.asarray(samples)
    print(f"\n{side ** 2} RIS elements, {draws} paired draws, up to {iterations} AO iterations")
    print("| Method | Mean sum rate | Mean reward | All users pass | All targets pass | Joint pass |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    names = ("Fixed heuristic", "Fixed AO", "Shared heuristic", "Shared AO")
    for name, (rate, score, users, targets, joint) in zip(names, result.mean(axis=0)):
        print(f"| {name} | {rate:.6g} | {score:.6g} | {users:.1%} | {targets:.1%} | {joint:.1%} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ris-sides", nargs="+", type=int, default=[6])
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=100_000)
    args = parser.parse_args()
    if min(args.ris_sides) < 2 or args.draws < 1 or args.iterations < 1:
        parser.error("RIS sides must be at least 2, and draws and iterations must be positive.")
    for side in args.ris_sides:
        compare(side, args.draws, args.iterations, args.seed)


if __name__ == "__main__":
    main()
