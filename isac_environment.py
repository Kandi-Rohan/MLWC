"""Multi-user RIS-assisted ISAC environment for Wei et al., equations (1)-(8)."""

from dataclasses import dataclass

import numpy as np

from kmeans_clustering import k_means_plus_plus_cu_grouping
from RIS_Phase_rate import BS, RIS, USER, a_ula, a_upa, quantize


@dataclass(frozen=True)
class Config:
    antennas: int = 16
    rows: int = 6
    cols: int = 6
    users: int = 6
    targets: int = 3
    groups: int = 3
    paths: int = 3
    bits: int = 3
    noise_w: float = 1e-12
    power_w: float = 1.0
    min_rate: float = 0.5
    min_sensing_w: float = 3e-7


def partition(rows, cols, geometry):
    """Project center, size and priority proposals onto a rectangular tiling.

    Use recursive axis-aligned cuts. Each cut keeps enough cells for every
    group assigned to each side. A zero-sized sensing proposal can opt out.
    """
    grid = np.empty((rows, cols), dtype=int)
    centers = geometry[:, :2] * [rows, cols]
    desired_size = np.maximum(geometry[:, 2:4] * [rows, cols], 1)
    weights = 1 + geometry[:, 4]
    groups = list(range(len(geometry)))
    if np.any(geometry[-1, 2:4] == 0):
        groups.pop()
    if len(groups) > grid.size:
        raise ValueError("RIS must have at least one element per active group")

    def tile(lo, hi, ids):
        if len(ids) == 1:
            grid[lo[0]:hi[0], lo[1]:hi[1]] = ids[0]
            return

        best = None
        best_score = float("inf")
        for axis in (0, 1):
            order = sorted(ids, key=lambda i: (centers[i, axis], -weights[i]))
            width = hi[1 - axis] - lo[1 - axis]
            areas = np.prod(desired_size[order], axis=1) * weights[order]
            fractions = np.cumsum(areas) / areas.sum()
            for cut in range(lo[axis] + 1, hi[axis]):
                left_cells = (cut - lo[axis]) * width
                right_cells = (hi[axis] - cut) * width
                for count in range(max(1, len(ids) - right_cells),
                                   min(len(ids) - 1, left_cells) + 1):
                    left, right = order[:count], order[count:]
                    distance = (
                        np.sum(weights[left] * np.maximum(centers[left, axis] - cut, 0) ** 2)
                        + np.sum(weights[right] * np.maximum(cut - centers[right, axis], 0) ** 2)
                    )

                    fraction = fractions[count - 1]
                    score = distance / (hi[axis] - lo[axis]) ** 2
                    score += (left_cells / (left_cells + right_cells) - fraction) ** 2
                    if score < best_score:
                        best_score = score
                        best = axis, cut, left, right
        axis, cut, left, right = best
        left_hi, right_lo = hi.copy(), lo.copy()
        left_hi[axis] = right_lo[axis] = cut
        tile(lo, left_hi, left)
        tile(right_lo, hi, right)

    tile([0, 0], [rows, cols], groups)
    return grid


class ISACEnvironment:
    def __init__(self, config=Config(), seed=0):
        self.config = config
        c = config
        if not (1 <= c.groups <= c.users and c.targets >= 1 and c.groups + 1 <= c.rows * c.cols):
            raise ValueError("Need at least one target, user per group, and RIS cell per group")
        self.rng = np.random.default_rng(seed)
        self.elements = c.rows * c.cols
        self.streams = c.users + c.targets
        self.action_size = 5 * (c.groups + 1) + 2 * c.antennas * self.streams + self.elements
        self.reset()

    def _positions(self, center, count):
        radius = 2 * np.sqrt(self.rng.random(count))
        angle = self.rng.uniform(0, 2 * np.pi, count)
        return center + np.column_stack((radius * np.cos(angle), radius * np.sin(angle),
                                         np.zeros(count)))

    def _link(self, source, destination, kind):
        c = self.config
        distance = np.linalg.norm(destination - source)
        path_loss = 61.4 + 20 * np.log10(distance) + self.rng.normal(0, 5.8)
        gains = (self.rng.normal(size=c.paths) + 1j * self.rng.normal(size=c.paths))
        gains *= np.sqrt(10 ** (-path_loss / 10) / (2 * c.paths))
        paths = []
        angles = self._path_angles(source, destination, kind, gains)
        for gain, (azimuth, elevation, departure) in zip(gains, angles):
            if kind == "direct":
                response = a_ula(c.antennas, departure)
            else:
                response = a_upa(c.rows, c.cols, azimuth, elevation)
                if kind == "bs_ris":
                    response = np.outer(response, a_ula(c.antennas, departure).conj())
            paths.append(gain * response)
        return np.sum(paths, axis=0)

    def _path_angles(self, source, destination, kind, gains):
        return np.array([self.rng.uniform(-np.pi / 2, np.pi / 2, 3) for _ in gains])

    def reset(self):
        """Draw fresh CSI and fix the NCC groups for that realization."""
        c = self.config
        users = self._positions(USER, c.users)
        targets = self._positions(np.array([8, 25, 1.5]), c.targets)
        self.bs_ris = self._link(BS, RIS, "bs_ris")
        self.ris_users = np.array([self._link(RIS, user, "ris") for user in users])
        self.ris_targets = np.array([self._link(RIS, target, "ris") for target in targets])
        self.direct = np.array([self._link(BS, target, "direct") for target in targets])
        self.groups, self.centers = k_means_plus_plus_cu_grouping(self.ris_users, c.groups)
        self.user_group = np.empty(c.users, dtype=int)
        for label, members in enumerate(self.groups):
            self.user_group[members] = label
        # Fixed scaling keeps absolute channel strengths in the CSI.
        channels = np.concatenate([array.ravel() for array in
                                   (self.bs_ris, self.ris_users, self.ris_targets, self.direct)])
        return (1e4 * np.concatenate([channels.real, channels.imag])).astype(np.float32)

    def link_upper_bounds(self):
        """Bound each link independently with all power, all RIS cells and no interference."""
        c = self.config
        row_norms = np.linalg.norm(self.bs_ris, axis=1)
        user_norms = np.abs(self.ris_users) @ row_norms
        target_norms = np.linalg.norm(self.direct, axis=1) + np.abs(self.ris_targets) @ row_norms
        return (np.log2(1 + c.power_w * user_norms**2 / c.noise_w),
                c.power_w * target_norms**2)

    def reflected_links(self, grid):
        """Return conjugated RIS-to-user and RIS-to-target channels after masking."""
        if grid is None:
            return self.ris_users.conj(), self.ris_targets.conj()
        labels = grid.ravel()
        return (self.ris_users.conj() * (self.user_group[:, None] == labels),
                self.ris_targets.conj() * (labels == self.config.groups))

    def stream_metrics(self, user_streams, target_streams):
        """Compute rates, sensing power and the QoS part of Eq. (8), with optional batches."""
        c = self.config
        received = np.abs(user_streams) ** 2
        users = np.arange(c.users)
        desired = received[..., users, users].copy()
        received[..., users, users] = 0
        interference = received.sum(axis=-1)
        rates = np.log2(1 + desired / (interference + c.noise_w))
        sensing = (np.abs(target_streams) ** 2).sum(axis=-1)
        rate_deficit = np.maximum(c.min_rate - rates, 0).sum(axis=-1)
        sensing_deficit = np.maximum(c.min_sensing_w - sensing, 0).sum(axis=-1)
        score = rates.sum(axis=-1) - 10 * (rate_deficit + sensing_deficit)
        return rates, sensing, desired, interference, score

    def evaluate(self, grid, beam, phases):
        """Evaluate beamforming on the current channels and partition."""
        reflected = np.exp(1j * phases)[:, None] * self.bs_ris
        users, targets = self.reflected_links(grid)
        effective_users = users @ reflected
        effective_targets = self.direct.conj() + targets @ reflected
        rates, sensing, desired, interference, score = self.stream_metrics(
            effective_users @ beam, effective_targets @ beam)
        power = float(np.sum(np.abs(beam) ** 2))
        reward = score - 10 * max(power - self.config.power_w, 0)
        return dict(rates=rates, sensing_w=sensing, power_w=power, reward=float(reward),
                    sum_rate=float(rates.sum()), desired_w=desired, interference_w=interference)

    def step(self, action):
        """Decode a continuous action in [-1, 1] and draw the next CSI."""
        c = self.config
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_size,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite 1-D action of length action_size")
        action = np.clip(action, -1, 1)
        geometry_end = 5 * (c.groups + 1)
        geometry = (action[:geometry_end].reshape(-1, 5) + 1) / 2
        grid = partition(c.rows, c.cols, geometry)
        beam_parts = action[geometry_end:-self.elements].reshape(2, c.antennas, self.streams)
        beam = beam_parts[0] + 1j * beam_parts[1]
        beam /= max(1, np.linalg.norm(beam) / np.sqrt(c.power_w))
        phases = quantize(np.pi * (action[-self.elements:] + 1), c.bits)
        info = self.evaluate(grid, beam, phases)
        info["partition"] = grid
        return self.reset(), info["reward"], info


