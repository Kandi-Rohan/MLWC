"""
RIS phase optimization + achievable rate (simplified version of the paper's system model)
- 1 BS (N antennas, ULA) -> RIS (M = Mx*My elements, UPA) -> 1 user
- Channels: Saleh-Valenzuela (eq. 1, 2 of the paper), 28 GHz, half-wavelength spacing
- Path loss: PL(d) = 61.4 + 20*log10(d) + shadowing (std 5.8 dB)
- Phase optimization: alternating (BS beam <-> RIS phases), b-bit quantized phases
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ---------- parameters (from the paper) ----------
N = 16                      # BS antennas
L = 3                       # paths per channel
Pmax = 1.0                  # 30 dBm = 1 W
noise = 10 ** ((-90 - 30) / 10)   # -90 dBm in Watts
BS, RIS, USER = np.array([0, 0, 10.]), np.array([10, 30, 30.]), np.array([8, 35, 1.5])


def gain_var(d):
    """Average power gain of one path: 10^(-0.1*PL(d)), with random shadowing."""
    PL = 61.4 + 20 * np.log10(d) + rng.normal(0, 5.8)
    return 10 ** (-0.1 * PL)


def a_ula(n, ang):
    return np.exp(-1j * np.pi * np.arange(n) * np.sin(ang))          # d = lambda/2


def a_upa(mx, my, az, el):
    ax = np.exp(-1j * np.pi * np.arange(mx) * np.sin(az) * np.sin(el))
    ay = np.exp(-1j * np.pi * np.arange(my) * np.cos(el))
    return np.kron(ax, ay)


def make_channels(mx, my):
    M = mx * my
    # BS -> RIS  (M x N)
    H = np.zeros((M, N), complex)
    var = gain_var(np.linalg.norm(RIS - BS))
    for _ in range(L):
        alpha = np.sqrt(var / 2) * (rng.normal() + 1j * rng.normal())
        az, el, dep = rng.uniform(-np.pi / 2, np.pi / 2, 3)
        H += alpha * np.outer(a_upa(mx, my, az, el), a_ula(N, dep).conj())
    H /= np.sqrt(L)
    # RIS -> user (M,)
    h = np.zeros(M, complex)
    var = gain_var(np.linalg.norm(USER - RIS))
    for _ in range(L):
        alpha = np.sqrt(var / 2) * (rng.normal() + 1j * rng.normal())
        az, el = rng.uniform(-np.pi / 2, np.pi / 2, 2)
        h += alpha * a_upa(mx, my, az, el)
    h /= np.sqrt(L)
    return H, h


def quantize(theta, bits):
    """Snap each phase to the nearest of 2^bits equally spaced values."""
    if bits is None:
        return theta
    if not isinstance(bits, (int, np.integer)) or bits < 0:
        raise ValueError("bits must be a non-negative integer or None")
    step = 2 * np.pi / 2 ** bits
    return np.round(theta / step) * step


def rate_from(H, h, phi):
    g = (h.conj() * phi) @ H                  # 1 x N effective channel  h^H Theta H
    channel_norm = np.linalg.norm(g)
    if channel_norm == 0:
        return 0.0
    w = g.conj() / channel_norm               # MRT beam (best beam for this g)
    snr = Pmax * np.abs(g @ w) ** 2 / noise
    return np.log2(1 + snr)


def optimize(H, h, bits, iters=5):
    """Alternate: (1) beam w for current phases, (2) best phase per element for current w."""
    M = len(h)
    phi = np.ones(M, complex)
    for _ in range(iters):
        g = (h.conj() * phi) @ H
        channel_norm = np.linalg.norm(g)
        if channel_norm == 0:
            return 0.0
        w = g.conj() / channel_norm
        Hw = H @ w                                         # what each element 'sees' from the BS
        theta = -np.angle(h.conj() * Hw)                   # phase that makes every term add in-phase
        phi = np.exp(1j * quantize(theta, bits))
    return rate_from(H, h, phi)


def random_phase_rate(H, h, bits):
    phi = np.exp(1j * quantize(rng.uniform(0, 2 * np.pi, len(h)), bits))
    return rate_from(H, h, phi)


TRIALS = 300


def run_experiments():
    # ---------- Experiment 1: rate vs number of RIS elements ----------
    sizes = [4, 6, 8, 10]                  # M = 16, 36, 64, 100
    Ms = [s * s for s in sizes]
    res1 = {
        "Optimized, continuous": [],
        "Optimized, 3-bit": [],
        "Optimized, 1-bit": [],
        "Random phase (no optimization)": [],
    }

    for s in sizes:
        acc = {key: [] for key in res1}
        for _ in range(TRIALS):
            H, h = make_channels(s, s)
            acc["Optimized, continuous"].append(optimize(H, h, None))
            acc["Optimized, 3-bit"].append(optimize(H, h, 3))
            acc["Optimized, 1-bit"].append(optimize(H, h, 1))
            acc["Random phase (no optimization)"].append(
                random_phase_rate(H, h, 3)
            )
        for key, values in acc.items():
            res1[key].append(np.mean(values))

    plt.figure(figsize=(6.5, 4.5))
    for key, style in zip(res1, ["o-", "s-", "^-", "x--"]):
        plt.plot(Ms, res1[key], style, label=key)
    plt.xlabel("Number of RIS elements M")
    plt.ylabel("Average achievable rate (bit/s/Hz)")
    plt.title(f"Rate vs RIS elements (N={N}, 1 user, {TRIALS} channel draws)")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig1_rate_vs_RIS_elements.png", dpi=200)
    plt.close()

    # ---------- Experiment 2: rate vs phase bits (M = 64) ----------
    bits_list = [1, 2, 3, 4, 5]
    channels = [make_channels(8, 8) for _ in range(TRIALS)]
    r_bits = [
        np.mean([optimize(H, h, bits) for H, h in channels])
        for bits in bits_list
    ]
    r_cont = np.mean([optimize(H, h, None) for H, h in channels])
    r_rand = np.mean([random_phase_rate(H, h, 3) for H, h in channels])

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(bits_list, r_bits, "o-", label="Optimized, b-bit phases")
    plt.axhline(r_cont, color="g", ls="--", label="Optimized, continuous phases")
    plt.axhline(r_rand, color="r", ls=":", label="Random phases")
    plt.xlabel("Phase quantization bits b")
    plt.ylabel("Average achievable rate (bit/s/Hz)")
    plt.title(f"Rate vs phase bits (M=64, N={N})")
    plt.xticks(bits_list)
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig2_rate_vs_phase_bits.png", dpi=200)
    plt.close()

    print("M:", Ms)
    for key, values in res1.items():
        print(f"{key:35s}", np.round(values, 2))
    print(
        "bits:", bits_list,
        "->", np.round(r_bits, 2),
        "| continuous:", round(r_cont, 2),
        "| random:", round(r_rand, 2),
    )


if __name__ == "__main__":
    run_experiments()