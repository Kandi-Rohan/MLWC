"""Run an equation-to-code channel audit with a deterministic small example."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

# Support execution as python channel_audit/check_link_budget.py from the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isac_environment import Config, ISACEnvironment
from RIS_Phase_rate import a_ula, a_upa, quantize


class BroadsideSamples:
    """Supply zero shadowing, unit-real normalized gains and broadside angles."""

    def __init__(self):
        self.draw_real = True

    def normal(self, loc=0, scale=1, size=None):
        if size is None:
            return 0.0
        value = np.sqrt(2.0) if self.draw_real else 0.0
        self.draw_real = not self.draw_real
        return np.full(size, value)

    def uniform(self, low, high, size):
        return np.array([0.0, np.pi / 2, 0.0])


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=0)


def check_arrays():
    # Verify phase progression and UPA flattening away from broadside.
    az, el = np.pi / 3, np.pi / 4
    expected_upa = [
        np.exp(-1j * np.pi * (row * np.sin(az) * np.sin(el) + col * np.cos(el)))
        for row in range(2) for col in range(3)
    ]
    close(a_upa(2, 3, az, el), expected_upa)
    close(a_ula(4, np.pi / 6), [1, -1j, -1, 1j])
    close(quantize(np.array([0.7, 1.5, -0.7]), 3), [np.pi / 4, np.pi / 2, -np.pi / 4])


def check_group_interference():
    config = Config(antennas=1, rows=1, cols=3, users=3, targets=1, groups=2)
    env = ISACEnvironment(config)
    env.user_group[:] = [0, 0, 1]
    env.bs_ris[:] = 1
    env.ris_users[:] = 1
    env.ris_targets[:] = 1j
    env.direct[:] = 2
    powers = np.array([0.4, 0.1, 0.2, 0.3])
    beam = np.sqrt(powers)[None, :]
    # Each user sees its assigned cell. The other two user streams and
    # the sensing stream all interfere. At the target, j * conj(j) = 1.
    result = env.evaluate(np.array([[0, 1, 2]]), beam, np.array([0, 0, np.pi / 2]))
    close(result["rates"], np.log2(1 + powers[:3] / (1 - powers[:3] + config.noise_w)))
    close(result["sensing_w"], [9])


def main():
    check_arrays()
    check_group_interference()
    config = Config(users=1, targets=1, groups=1, paths=1)
    env = ISACEnvironment(config)
    env.rng = BroadsideSamples()
    bs, ris, user, target = ([0, 0, 10], [10, 30, 30], [8, 35, 1.5], [8, 25, 1.5])
    default_noise = 10 ** ((-90 - 30) / 10)
    default_power = 10 ** ((30 - 30) / 10)
    close(config.noise_w, default_noise)
    close(config.power_w, default_power)

    # Use squared distances directly to compute independent reference gains.
    gain_at_one_meter = 10 ** (-61.4 / 10)
    expected_gains = gain_at_one_meter / np.array([1400, 841.25, 841.25, 761.25])
    links = [
        env._link(np.array(bs), np.array(ris), "bs_ris"),
        env._link(np.array(ris), np.array(user), "ris"),
        env._link(np.array(ris), np.array(target), "ris"),
        env._link(np.array(bs), np.array(target), "direct"),
    ]
    names = ["BS to RIS", "RIS to user", "RIS to target", "BS to target"]
    for name, gain, channel in zip(names, expected_gains, links):
        close(channel, np.full(channel.shape, np.sqrt(gain)))
        print(f"{name:26} path power gain: {gain:.6e} ({10 * np.log10(gain):.2f} dB)")

    multi = ISACEnvironment(replace(config, paths=3))
    multi.rng = BroadsideSamples()
    three_path = multi._link(np.array(bs), np.array(ris), "bs_ris")
    # Three equal coherent amplitudes divided by sqrt(3).
    close(three_path, np.sqrt(3) * links[0])

    env.bs_ris, env.ris_users[0], env.ris_targets[0], env.direct[0] = links
    m, n = env.elements, config.antennas
    beam = np.zeros((n, 2), dtype=complex)
    beam[:, 0] = np.sqrt(default_power / n)
    phi = np.zeros(m)
    split = np.zeros((config.rows, config.cols), dtype=int)
    split[:, config.cols // 2:] = 1
    half = m // 2

    bs_ris_gain, user_gain, target_gain, direct_gain = expected_gains
    user_amplitude = np.sqrt(bs_ris_gain * user_gain)
    target_amplitude = np.sqrt(bs_ris_gain * target_gain)
    direct_amplitude = np.sqrt(direct_gain)
    user_rx = default_power * n * (m * user_amplitude) ** 2
    user_rate = np.log1p(user_rx / default_noise) / np.log(2)
    direct_power = default_power * n * direct_gain
    reflected_power = default_power * n * (m * target_amplitude) ** 2
    full_target_power = default_power * n * (direct_amplitude + m * target_amplitude) ** 2

    full = env.evaluate(None, beam, phi)
    close(full["rates"], [user_rate])
    close(full["sensing_w"], [full_target_power])
    close(full["power_w"], default_power)
    half_out = env.evaluate(split, beam, phi)
    close(half_out["rates"], [np.log2(1 + user_rx / (4 * default_noise))])
    close(half_out["sensing_w"], [
        default_power * n * (direct_amplitude + half * target_amplitude) ** 2
    ])

    # With a quadrature phase on the reflected path, the coherent cross term vanishes.
    quadrature = env.evaluate(None, beam, np.full(m, np.pi / 2))
    close(quadrature["sensing_w"], [direct_power + reflected_power])
    direct_only = env.evaluate(np.zeros_like(split), beam, phi)
    close(direct_only["sensing_w"], [direct_power])
    env.direct[:] = 0
    reflected_only = env.evaluate(None, beam, phi)
    close(reflected_only["sensing_w"], [reflected_power])
    env.direct[0] = links[3]

    # Equal power on the second (sensing) stream adds user interference and
    # preserves total target illumination with the same beam direction.
    two_stream_beam = np.repeat(beam[:, :1] / np.sqrt(2), 2, axis=1)
    two_streams = env.evaluate(None, two_stream_beam, phi)
    close(two_streams["rates"], [np.log2(1 + (user_rx / 2) / (default_noise + user_rx / 2))])
    close(two_streams["sensing_w"], [full_target_power])

    print(f"Full-RIS user received power: {user_rx:.6e} W")
    print(f"Full-RIS user SNR:            {10*np.log10(user_rx/default_noise):.2f} dB")
    print(f"Full-RIS user rate:           {user_rate:.6g} bit/s/Hz")
    print(f"Direct target power:          {direct_power:.6e} W")
    print(f"Reflected target power:       {reflected_power:.6e} W")
    print(f"Combined target power:        {full_target_power:.6e} W")
    print("PASS: steering, phase grid, path loss, path normalization, partition masks,")
    print("      unit conversion, communication interference and coherent sensing.")


if __name__ == "__main__":
    main()

