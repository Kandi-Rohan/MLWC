"""Compare channel-angle assumptions and measure the gain needed for QoS."""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isac_environment import Config, ISACEnvironment
from reproduction_pilot import heuristic


class GeometryAngleEnvironment(ISACEnvironment):
    """Use the endpoint direction for the strongest path on each link."""

    def _path_angles(self, source, destination, kind, gains):
        angles = super()._path_angles(source, destination, kind, gains)
        direction = destination - source
        direction /= np.linalg.norm(direction)
        # ULA along x, UPA along x (rows) and z (columns).
        # BS-to-RIS arrival uses the direction from the RIS back to the BS.
        ris_direction = -direction if kind == "bs_ris" else direction
        x, y, z = ris_direction
        azimuth = np.arctan2(x, y)
        zenith = np.arccos(np.clip(z, -1, 1))
        departure = np.arcsin(np.clip(direction[0], -1, 1))
        angles[np.argmax(np.abs(gains))] = azimuth, zenith, departure
        return angles


def required_gain_db(numerator, margin):
    """Return the worst link's required power multiplier in dB."""
    if np.any(margin <= 0):
        return np.inf
    return 10 * np.log10(np.max(numerator / margin))


def gain_margins(env):
    """Measure weakest-user and weakest-target margins for each evaluation."""
    c = env.config
    threshold = np.expm1(c.min_rate * np.log(2))
    rate_bound, sensing_bound = env.link_upper_bounds()
    signal_bound = c.noise_w * np.expm1(rate_bound * np.log(2))
    fixed = heuristic(env, shared=False)
    # CU signal and interference both scale with receiver-side gain G.
    # G*S / (G*I + noise) >= threshold requires G >= threshold*noise/(S-threshold*I).
    residual_signal = fixed["desired_w"] - threshold * fixed["interference_w"]
    return [
        [required_gain_db(threshold * c.noise_w, signal_bound),
         required_gain_db(c.min_sensing_w, sensing_bound)],
        [required_gain_db(threshold * c.noise_w, residual_signal),
         required_gain_db(c.min_sensing_w, fixed["sensing_w"])],
    ]


def summarize(label, margins):
    print(f"\n{label}")
    print("| Evaluation | CU pass | Target pass | Joint pass | Median CU gain needed | "
          "Median target gain needed | No finite CU gain |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, gains in zip(("Upper bounds", "Fixed partition"), margins.transpose(1, 0, 2)):
        users, targets = gains.T
        print(f"| {name} | {np.mean(users <= 0):.0%} | {np.mean(targets <= 0):.0%} | "
              f"{np.mean((users <= 0) & (targets <= 0)):.0%} | "
              f"{np.median(users):.1f} dB | {np.median(targets):.1f} dB | "
              f"{np.mean(np.isinf(users)):.0%} |")
    print("\n| Added effective power gain | Joint upper bounds pass | Joint fixed partition pass |")
    print("| ---: | ---: | ---: |")
    joint_gain = np.max(margins, axis=2)
    for gain in (0, 10, 20, 30, 40, 50, 60):
        bounds, fixed = np.mean(joint_gain <= gain, axis=0)
        print(f"| +{gain} dB | {bounds:.0%} | {fixed:.0%} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100_000)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    config = Config()
    random_angles = ISACEnvironment(config, args.seed)
    position_angles = GeometryAngleEnvironment(config, args.seed)
    samples = [[], []]
    for _ in range(args.draws):
        for env, sample in zip((random_angles, position_angles), samples):
            env.reset()
            sample.append(gain_margins(env))
        # Both variants consume exactly the same random samples.
        if random_angles.rng.bit_generator.state != position_angles.rng.bit_generator.state:
            raise RuntimeError("Channel draw sequences diverged")
    print(f"{args.draws} paired draws with {config.antennas} BS antennas, "
          f"{config.rows * config.cols} RIS elements, seed {args.seed}.")
    for label, sample in zip(("Random path angles", "Geometry-based strongest path"), samples):
        summarize(label, np.asarray(sample))


if __name__ == "__main__":
    main()
