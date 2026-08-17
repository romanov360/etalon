"""Ring-to-ring thermal crosstalk for microring WDM banks.

:mod:`siphon.wdm` prices the heater power a ring needs to lock its OWN
resonance (:func:`siphon.wdm.tuning_power_mw`,
:func:`siphon.wdm.optimize_ring_assignment`) but treats every ring as
thermally isolated. On a real chip a ring's heater warms the shared
substrate/cladding, and that heat reaches neighboring rings too — a
resistive heater a few microns away detunes its neighbor by tens of pm,
which is a sizeable fraction of a dense-WDM channel spacing. This module
adds that coupling: a spatial temperature kernel from ring pitch, and the
self-consistent fixed point where every ring's own heater power both
locks its resonance AND compensates the heat arriving from every other
ring's heater.

Physics
-------
Steady-state heat conduction from a heater on a chip with a heat sink
(the package/PCB) a roughly fixed distance beneath the device layer is
well approximated, at the lateral (in-plane) scales relevant to ring
pitch (a few to a few hundred um), by an exponential spatial kernel:

    T(r) = T(0) * exp(-r / decay_um)

with ``decay_um`` a thermal healing length set by the vertical path to
the heat sink (oxide/substrate stack thickness and conductivity). This
is the standard reduced-order form used for on-chip thermal crosstalk
between microring heaters (e.g. Padmaraju & Bergman, IEEE Photonics
Journal 2014; Milanizadeh et al., thermal crosstalk compensation for
programmable PICs). It is a TAIL-MATCHED approximation, not a derived
near-field solution: the sink-loaded 2-D lateral spreading problem this
models is more precisely a modified-Bessel K0(r/decay_um) kernel, which
diverges logarithmically as r -> 0 and only *approaches* the pure
exponential's falloff at r >> decay_um; exp(-r/decay_um) understates the
near-field rise a touch relative to K0, which matters most exactly where
this module is used (adjacent-ring pitch, i.e. small r/decay_um) — one
more reason ``decay_um`` should be treated as a fit parameter against
measurement or a real thermal simulation, not derived from first
principles. Also NOT modeled: transient response (only the steady-state
map matters for a locked ring bank), heat-sink boundary geometry, and
undercut-trench detail — all absorbed into the single ``decay_um``
number. Typical measured values on bulk (non-undercut) SOI are of order
10-40 um; thermal isolation trenches (undercut rings) push the *self*
tuning efficiency up but do not, by themselves, remove substrate
crosstalk to a non-undercut neighbor.

Scope: :class:`RingLayout` is 1-D (rings on a straight bus), the common
case for a WDM ring bank sharing one waveguide. The kernel is evaluated
on the 1-D coordinate difference only — this is a simplification for
that layout, not a fundamental limit of the exponential-kernel idea, but
reusing :class:`RingLayout` for a 2-D ring array would need a proper 2-D
distance (and the near-field caveat above would matter more, not less).

Self-consistency
-----------------
Ring i must sit at its target detune from fabrication PLUS whatever
heat neighbors dump on it. Its own heater then contributes to every
OTHER ring's crosstalk in turn. With linear heat conduction (rise
proportional to input power) and the linear resonance-vs-temperature
model already used throughout :mod:`siphon.wdm`
(:func:`siphon.wdm.resonance_shift_nm_per_k`), the coupled system is
linear in the heater powers and :func:`solve_coupled_powers` solves it
exactly (a single ``numpy.linalg.solve``), not by iterative relaxation.

Conventions: positions in um, powers in mW, resonance shifts in nm,
temperatures in K — consistent with :mod:`siphon.wdm`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import wdm


@dataclass(frozen=True)
class RingLayout:
    """1-D positions of N rings sharing a bus, in um, in ring order.

    Order matches the ring order used elsewhere for the same bank (e.g.
    :func:`siphon.wdm.optimize_ring_assignment`'s ``offsets_nm``). Only
    relative spacing matters; positions need not start at zero.
    """

    positions_um: tuple[float, ...]

    def __post_init__(self):
        if len(self.positions_um) < 1:
            raise ValueError("positions_um must contain at least one ring")
        if not all(np.isfinite(p) for p in self.positions_um):
            raise ValueError("positions_um must be finite")

    @property
    def n(self) -> int:
        return len(self.positions_um)

    def pitch_matrix_um(self) -> np.ndarray:
        """Symmetric (n, n) matrix of |x_i - x_j|, zero diagonal."""
        x = np.asarray(self.positions_um, dtype=float)
        return np.abs(x[:, None] - x[None, :])

    @classmethod
    def uniform(cls, n: int, pitch_um: float) -> "RingLayout":
        """N rings on a straight bus at constant ``pitch_um`` spacing."""
        if n < 1:
            raise ValueError("n must be >= 1")
        if pitch_um <= 0:
            raise ValueError("pitch_um must be positive")
        return cls(tuple(float(i * pitch_um) for i in range(n)))


def crosstalk_kernel(dx_um: np.ndarray, decay_um: float) -> np.ndarray:
    """Fraction of a heater's temperature rise seen at distance ``dx_um``.

    Exponential healing-length kernel ``exp(-dx / decay_um)``, elementwise
    (see module docstring for the physical justification and its limits).
    At ``dx_um = 0`` returns 1.0 exactly (a ring's own site sees its own
    full rise — used only to build the coupling matrix's off-diagonal;
    :func:`coupling_matrix` zeroes the diagonal itself so self-heating is
    not double counted against the ring's own tuning efficiency).
    """
    if decay_um <= 0:
        raise ValueError("decay_um must be positive")
    dx = np.asarray(dx_um, dtype=float)
    if np.any(dx < 0):
        raise ValueError("dx_um must be non-negative (pass a distance, not a signed offset)")
    return np.exp(-dx / decay_um)


def coupling_matrix(layout: RingLayout, decay_um: float) -> np.ndarray:
    """(n, n) matrix K where K[i, j] = fraction of ring j's heat seen at ring i.

    Off-diagonal only: ``K[i, j] = crosstalk_kernel(|x_i - x_j|, decay_um)``
    for ``i != j``; the diagonal is exactly zero (a ring's own heater
    power already sets its own temperature via the tuning efficiency in
    :mod:`siphon.wdm` — the diagonal would double count it).
    """
    k = crosstalk_kernel(layout.pitch_matrix_um(), decay_um)
    np.fill_diagonal(k, 0.0)
    return k


@dataclass(frozen=True)
class CoupledTuningResult:
    """Result of :func:`solve_coupled_powers`.

    Attributes
    ----------
    heater_mw:
        Self-consistent heater power per ring (mW) that locks every ring
        at its target detune once neighbor heating is accounted for.
    naive_mw:
        Heater power per ring ignoring crosstalk entirely (what
        :func:`siphon.wdm.tuning_power_mw` would report per ring).
    neighbor_shift_nm:
        Resonance shift each ring receives from ITS NEIGHBORS' heaters
        alone (excluding its own), at the coupled solution, in nm
        (red-positive, same sign convention as heating).
    """

    heater_mw: tuple[float, ...]
    naive_mw: tuple[float, ...]
    neighbor_shift_nm: tuple[float, ...]

    @property
    def total_mw(self) -> float:
        return float(sum(self.heater_mw))

    @property
    def naive_total_mw(self) -> float:
        return float(sum(self.naive_mw))

    def report(self) -> str:
        """Compact plain-text per-ring table."""
        lines = [
            f"thermal crosstalk-coupled tuning ({len(self.heater_mw)} rings):",
            f"  {'ring':>4}  {'naive mW':>9}  {'coupled mW':>10}  {'neighbor shift (nm)':>20}",
        ]
        for i, (n, c, s) in enumerate(
            zip(self.naive_mw, self.heater_mw, self.neighbor_shift_nm)
        ):
            lines.append(f"  {i:>4}  {n:>9.4f}  {c:>10.4f}  {s:>+20.5f}")
        lines.append(
            f"  total: naive {self.naive_total_mw:.4f} mW, "
            f"coupled {self.total_mw:.4f} mW "
            f"({100.0 * (self.total_mw / self.naive_total_mw - 1.0):+.1f}%)"
            if self.naive_total_mw > 0
            else f"  total: naive {self.naive_total_mw:.4f} mW, coupled {self.total_mw:.4f} mW"
        )
        return "\n".join(lines)


def solve_coupled_powers(
    target_detune_nm,
    layout: RingLayout,
    decay_um: float,
    tuning_efficiency_nm_per_mw: float = wdm.TUNING_EFFICIENCY_NM_PER_MW,
) -> CoupledTuningResult:
    """Self-consistent heater powers under ring-to-ring thermal crosstalk.

    Each ring must reach ``target_detune_nm[i]`` (red-shift from its
    as-fabricated resonance, same sign convention as
    :func:`siphon.wdm.tuning_power_mw` — pass only non-negative values;
    heaters cannot cool a ring, see that function's docstring for how to
    handle a physical blue-shift request). Ring i's total shift is its own
    heater's contribution plus the coupled fraction of every other ring's
    heater power:

        target[i] = P[i] * eff + sum_{j != i} K[i, j] * P[j] * eff

    with ``eff = tuning_efficiency_nm_per_mw`` and K from
    :func:`coupling_matrix`. This is linear in P: with ``M = I + K`` the
    exact solution is ``P = solve(M, target) / eff`` — no iterative
    relaxation, no convergence tolerance to tune. A
    negative solved power means the target is unreachable without
    crosstalk assistance alone overshooting it (neighbors already push
    this ring past its target); ValueError is raised naming the ring,
    since a resistive heater cannot draw power away from the shared
    substrate to compensate.

    Parameters
    ----------
    target_detune_nm:
        Array-like, length ``layout.n``: required red-shift per ring in
        nm (>= 0).
    layout:
        Ring positions (see :class:`RingLayout`).
    decay_um:
        Thermal healing length in um (> 0; see module docstring).
    tuning_efficiency_nm_per_mw:
        Same convention as :mod:`siphon.wdm` (> 0).
    """
    if tuning_efficiency_nm_per_mw <= 0:
        raise ValueError("tuning_efficiency_nm_per_mw must be positive")
    target = np.atleast_1d(np.asarray(target_detune_nm, dtype=float))
    if target.ndim != 1 or target.size != layout.n:
        raise ValueError(
            f"target_detune_nm must have length {layout.n} (one per ring in layout)"
        )
    if not np.all(np.isfinite(target)):
        raise ValueError("target_detune_nm must be finite")
    if np.any(target < 0):
        raise ValueError(
            "target_detune_nm must be >= 0 (heaters only red-shift; wrap a physical "
            "blue-shift request to the long way around the FSR before calling, as "
            "siphon.wdm.optimize_ring_assignment does)"
        )

    k = coupling_matrix(layout, decay_um)
    eff = tuning_efficiency_nm_per_mw
    m = np.eye(layout.n) + k
    try:
        power = np.linalg.solve(m, target) / eff
    except np.linalg.LinAlgError as exc:
        raise ValueError("coupled tuning system is singular") from exc

    if np.any(power < 0):
        bad = np.flatnonzero(power < 0)
        raise ValueError(
            f"target detune unreachable at ring(s) {bad.tolist()}: neighbor crosstalk "
            "alone overshoots the target (solved heater power would be negative). "
            "Reduce decay_um, increase ring pitch, or revisit the target assignment."
        )

    naive = target / eff
    neighbor_shift = k @ (power * eff)  # nm: coupling * (power[mW] * eff[nm/mW])
    return CoupledTuningResult(
        heater_mw=tuple(float(p) for p in power),
        naive_mw=tuple(float(p) for p in naive),
        neighbor_shift_nm=tuple(float(s) for s in neighbor_shift),
    )


def worst_case_neighbor_shift_nm(
    heater_mw,
    layout: RingLayout,
    decay_um: float,
    tuning_efficiency_nm_per_mw: float = wdm.TUNING_EFFICIENCY_NM_PER_MW,
) -> np.ndarray:
    """Uncoupled screening bound: neighbor-induced shift given FIXED heater powers.

    Unlike :func:`solve_coupled_powers` (which solves for the powers that
    hit a target despite crosstalk), this is the cheap one-shot direction:
    given heater powers already chosen (e.g. from
    :func:`siphon.wdm.optimize_ring_assignment`, which knows nothing about
    crosstalk), how much does that choice detune each ring's neighbors?
    ``shift[i] = eff * sum_j K[i, j] * heater_mw[j]``. Use this to check
    whether a crosstalk-blind assignment is good enough before reaching
    for the coupled solver.
    """
    if tuning_efficiency_nm_per_mw <= 0:
        raise ValueError("tuning_efficiency_nm_per_mw must be positive")
    power = np.atleast_1d(np.asarray(heater_mw, dtype=float))
    if power.ndim != 1 or power.size != layout.n:
        raise ValueError(f"heater_mw must have length {layout.n} (one per ring in layout)")
    if not np.all(np.isfinite(power)):
        raise ValueError("heater_mw must be finite")
    k = coupling_matrix(layout, decay_um)
    return k @ power * tuning_efficiency_nm_per_mw
