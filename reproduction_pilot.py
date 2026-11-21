"""Check the link budget and run a paired RIS-size pilot for Figure 2(c)."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from isac_environment import Config, ISACEnvironment
from RIS_Phase_rate import quantize
from train_ddpg import train


def align_phases(paths, direct, bits, initial):
    """Alternating MRT and quantized phase alignment for one receiver."""
    phase = initial.copy()
    best_link_power = -1
    for _ in range(8):
        channel = direct + np.exp(1j * phase) @ paths
        power = np.vdot(channel, channel).real
        if power > best_link_power:
            best_link_power, best_phase = power, phase.copy()
        beam = channel.conj()
        reference = np.angle(direct @ beam)
        phase = quantize(reference - np.angle(paths @ beam), bits)
    return best_phase, best_link_power


def link_budget(trials, seed):
    config = Config()
    env = ISACEnvironment(config, seed)
    rng = np.random.default_rng(seed + 1)
    samples = []
    for _ in range(trials):
        env.reset()
        u = env.ris_users[0].conj()[:, None] * env.bs_ris
        r = env.ris_targets[0].conj()[:, None] * env.bs_ris
        d = env.direct[0].conj()
        zero = np.zeros_like(d)
        initial = rng.uniform(-np.pi, np.pi, env.elements)
        initial = quantize(initial, config.bits)
        _, user = align_phases(u, zero, config.bits, initial)
        _, reflected = align_phases(r, zero, config.bits, initial)
        _, combined = align_phases(r, d, config.bits, initial)
        combined = max(combined, np.linalg.norm(d + r.sum(axis=0)) ** 2)
        bound_rate, bound_target = env.link_upper_bounds()
        samples.append([
            user * config.power_w, np.log2(1 + user * config.power_w / config.noise_w),
            np.linalg.norm(d) ** 2 * config.power_w, reflected * config.power_w,
            combined * config.power_w, bound_rate[0], bound_target[0],
            np.any(bound_rate < config.min_rate), np.any(bound_target < config.min_sensing_w),
        ])
    data = np.asarray(samples)
    thresholds = np.array([
        config.noise_w * np.expm1(config.min_rate * np.log(2)), config.min_rate,
        config.min_sensing_w, config.min_sensing_w, config.min_sensing_w,
        config.min_rate, config.min_sensing_w,
    ])
    labels = ["User received power (W)", "User rate (bit/s/Hz)", "Target direct MRT (W)",
              "Target reflected MRT (W)", "Target combined MRT (W)", "User upper rate bound",
              "Target upper power bound (W)"]
    for name, median, threshold, passed in zip(
            labels, np.median(data[:, :7], axis=0), thresholds,
            (data[:, :7] >= thresholds).mean(axis=0)):
        print(f"{name:30} median={median:.4g}  threshold={threshold:.4g}  pass={passed:.1%}")
    print(f"Draws with at least one CU below its upper bound threshold: {data[:, 7].mean():.1%}")
    print(f"Draws with at least one ST below its upper bound threshold: {data[:, 8].mean():.1%}")


def fixed_grid(config):
    grid = np.empty((config.rows, config.cols), dtype=int)
    row_mid, col_mid = config.rows // 2, config.cols // 2
    grid[:row_mid, :col_mid] = 0
    grid[:row_mid, col_mid:] = 1
    grid[row_mid:, :col_mid] = 2
    grid[row_mid:, col_mid:] = 3
    return grid


def heuristic(env, shared):
    """Phase alignment to representative links followed by regularized ZF."""
    c = env.config
    grid = fixed_grid(c)
    labels = grid.ravel()
    phase = np.zeros(env.elements)
    zero = np.zeros(c.antennas, dtype=complex)
    strongest_target = np.linalg.norm(env.direct, axis=1).argmax()
    if shared:
        # Optimize the shared surface for the group center with the strongest
        # reflected channel norm before phase alignment.
        center = max(env.centers, key=lambda k: np.linalg.norm(env.ris_users[k]))
        phase, _ = align_phases(env.ris_users[center].conj()[:, None] * env.bs_ris,
                                zero, c.bits, phase)
        user_mask = target_mask = np.ones(env.elements, dtype=bool)
    else:
        for group, k in enumerate(env.centers):
            mask = labels == group
            paths = env.ris_users[k, mask].conj()[:, None] * env.bs_ris[mask]
            phase[mask], _ = align_phases(paths, zero, c.bits, phase[mask])
        target_mask = labels == c.groups
        paths = env.ris_targets[strongest_target, target_mask].conj()[:, None] * env.bs_ris[target_mask]
        phase[target_mask], _ = align_phases(
            paths, env.direct[strongest_target].conj(), c.bits, phase[target_mask])
        user_mask = labels == env.user_group[:, None]
    reflected = np.exp(1j * phase)[:, None] * env.bs_ris
    users = (env.ris_users.conj() * user_mask) @ reflected
    targets = env.direct.conj() + (env.ris_targets.conj() * target_mask) @ reflected
    norms = np.linalg.norm(users, axis=1, keepdims=True)
    normalized_users = users / np.maximum(norms, np.finfo(float).tiny)
    gram = normalized_users @ normalized_users.conj().T
    comm_beam = np.linalg.solve(gram + 0.1 * np.eye(c.users), normalized_users).conj().T
    beam = np.column_stack((comm_beam, targets.conj().T))
    beam *= np.sqrt(c.power_w / env.streams) / np.maximum(
        np.linalg.norm(beam, axis=0), np.finfo(float).tiny)
    return env.evaluate(None if shared else grid, beam, phase)


def sweep(seed, trials, steps, plot_path):
    sizes = (3, 4, 5, 6, 7, 8, 10)
    names = ("Fixed partition heuristic", "Shared RIS heuristic", "DDPG pilot")
    results = []
    for side in sizes:
        config = replace(Config(), rows=side, cols=side)
        agent, _ = train(config, seed, steps, lr=1e-6, tau=1e-8)
        env = ISACEnvironment(config, seed + 100_000)
        state = env.reset()
        draws = []
        for _ in range(trials):
            fixed = heuristic(env, shared=False)
            shared = heuristic(env, shared=True)
            state, _, learned = env.step(agent.act(state))
            draws.append([(info["sum_rate"],
                           np.all(info["rates"] >= config.min_rate)
                           and np.all(info["sensing_w"] >= config.min_sensing_w))
                          for info in (fixed, shared, learned)])
        result = np.mean(draws, axis=0)
        results.append(result)
        print(f"M={side**2:3}: " + " | ".join(
            f"{name}: {rate:.6f} bit/s/Hz, {joint:.1%} joint pass"
            for name, (rate, joint) in zip(names, result)), flush=True)

    results = np.asarray(results)
    elements = np.square(sizes)
    fig, (rate_ax, joint_ax) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                           layout="constrained")
    for index, name in enumerate(names):
        rate_ax.plot(elements, results[:, index, 0], marker=("o", "s", "^")[index], label=name)
    rate_ax.legend()
    rate_ax.set_ylabel("Mean sum rate (bit/s/Hz)")
    rate_ax.set_title("RIS size sweep, 16 BS antennas")
    if np.all(results[:, :, 1] == 0):
        joint_ax.plot(elements, np.zeros_like(elements), "o-", color="0.25")
        joint_ax.text(0.5, 0.5, "0% joint satisfaction for every method",
                      ha="center", va="center", transform=joint_ax.transAxes)
    else:
        for index, name in enumerate(names):
            joint_ax.plot(elements, 100 * results[:, index, 1], marker=("o", "s", "^")[index])
    joint_ax.set_ylim(-4, 104)
    joint_ax.set_yticks([0, 50, 100])
    joint_ax.set_xticks(elements)
    joint_ax.set_ylabel("Joint QoS pass (%)")
    joint_ax.set_xlabel("Number of RIS elements")
    for ax in fig.axes:
        ax.grid(alpha=0.25)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=Path, default=Path("ris_size_pilot.png"))
    args = parser.parse_args()
    if args.trials < 1 or args.steps < 129:
        parser.error("Use at least one trial and 129 steps to include policy exploration.")
    torch.set_num_threads(1)
    link_budget(args.trials, args.seed + 100_000)
    sweep(args.seed, args.trials, args.steps, args.plot)


if __name__ == "__main__":
    main()

