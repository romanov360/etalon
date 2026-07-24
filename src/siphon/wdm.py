"""WDM system helpers for ring-based and coarse-WDM (CWDM) links.

Channel plans, ring-bank capacity limits, thermal tuning power estimates,
and worst-case crosstalk aggregation. All wavelengths in um unless a name
says otherwise (``*_nm``); frequencies in GHz/THz as named; power in mW;
isolation/crosstalk in dB.

Frequency <-> wavelength conversion uses f[GHz] = C_UM_GHZ / wl[um] with
C_UM_GHZ = 2.99792458e5 um*GHz (vacuum speed of light).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import materials

# Speed of light expressed so that f[GHz] * wl[um] = C_UM_GHZ.
# c = 2.99792458e14 um/s = 2.99792458e14 um*Hz = 2.99792458e5 um*GHz.
C_UM_GHZ = 2.99792458e5

# Typical thermal tuning efficiency of a silicon microring with an
# integrated (doped-Si or metal) heater: ~0.25 nm of resonance shift per mW
# of heater power. Measured values span roughly 0.1-0.4 nm/mW depending on
# thermal isolation (undercut rings reach far higher); see e.g.
# Padmaraju & Bergman, Nanophotonics 3, 269 (2014).
TUNING_EFFICIENCY_NM_PER_MW = 0.25

# Typical fraction of the silicon material thermo-optic coefficient seen by
# the effective index of a 450x220 nm strip TE mode (core confinement; the
# oxide cladding's much smaller dn/dT is neglected).
STRIP_TE_CONFINEMENT = 0.85


@dataclass(frozen=True)
class ChannelPlan:
    """A WDM channel plan: tuple of channel center wavelengths in um.

    Channels are stored in ascending wavelength order regardless of the
    order given at construction.
    """

    centers_um: tuple[float, ...]

    def __post_init__(self):
        if len(self.centers_um) == 0:
            raise ValueError("channel plan must contain at least one channel")
        if any(wl <= 0 for wl in self.centers_um):
            raise ValueError("channel wavelengths must be positive (in um)")
        object.__setattr__(self, "centers_um", tuple(sorted(self.centers_um)))

    # --- standard plans ---------------------------------------------------

    @classmethod
    def cwdm4(cls) -> "ChannelPlan":
        """CWDM4 plan (100G-CWDM4 MSA): 1271/1291/1311/1331 nm, 20 nm grid."""
        return cls((1.271, 1.291, 1.311, 1.331))

    @classmethod
    def lr4(cls) -> "ChannelPlan":
        """LAN-WDM plan of 100GBASE-LR4 (IEEE 802.3ba): 800 GHz grid.

        Center wavelengths 1295.56 / 1300.05 / 1304.58 / 1309.14 nm
        (231.4 / 230.6 / 229.8 / 229.0 THz).
        """
        return cls((1.29556, 1.30005, 1.30458, 1.30914))

    @classmethod
    def dwdm(cls, center_um: float, spacing_ghz: float, n_channels: int) -> "ChannelPlan":
        """DWDM plan: ``n_channels`` on a uniform *frequency* grid.

        The grid is centered on the optical frequency of ``center_um`` and
        spaced by ``spacing_ghz``; wavelengths follow from wl = c/f, so the
        wavelength spacing is slightly non-uniform (~ spacing * wl^2 / c).
        """
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if spacing_ghz <= 0:
            raise ValueError("spacing_ghz must be positive")
        f_center_ghz = C_UM_GHZ / center_um
        offsets = (np.arange(n_channels) - (n_channels - 1) / 2.0) * spacing_ghz
        f_ghz = f_center_ghz + offsets
        if np.any(f_ghz <= 0):
            raise ValueError("frequency grid extends to non-positive frequencies")
        return cls(tuple(float(wl) for wl in C_UM_GHZ / f_ghz))

    # --- derived quantities -----------------------------------------------

    def spacing_nm(self) -> np.ndarray:
        """Adjacent channel spacings in nm (length n_channels - 1)."""
        wl_nm = np.asarray(self.centers_um) * 1e3
        return np.diff(wl_nm)

    def as_frequencies_thz(self) -> np.ndarray:
        """Channel center frequencies in THz, same order as ``centers_um``."""
        return C_UM_GHZ / np.asarray(self.centers_um) * 1e-3

    def span_nm(self) -> float:
        """Wavelength span from first to last channel, in nm."""
        return float((self.centers_um[-1] - self.centers_um[0]) * 1e3)


def ring_bank_channel_count_limit(
    fsr_nm: float, channel_spacing_nm: float, guard_channels: int = 1
) -> int:
    """Maximum number of WDM channels a ring bank can serve within one FSR.

    A cascade of rings sharing a bus can address at most one FSR of unique
    resonances; ``guard_channels`` slots are reserved as dead band so the
    edge channel does not alias onto the first ring's adjacent resonance
    order. Returns floor(fsr/spacing) - guard_channels, clipped at 0.
    """
    if fsr_nm <= 0 or channel_spacing_nm <= 0:
        raise ValueError("fsr_nm and channel_spacing_nm must be positive")
    if guard_channels < 0:
        raise ValueError("guard_channels must be >= 0")
    return max(int(np.floor(fsr_nm / channel_spacing_nm)) - guard_channels, 0)


def tuning_power_mw(
    detune_nm: float, efficiency_nm_per_mw: float = TUNING_EFFICIENCY_NM_PER_MW
) -> float:
    """Heater power (mW) to shift a ring resonance by ``detune_nm``.

    Linear heater model P = |detune| / efficiency, charging the MAGNITUDE
    of the detuning. Heaters only red-shift (heat) a silicon ring, so a
    physical blue-shift request actually costs the long way around to the
    next resonance order, (FSR - |detune|) / efficiency — pass that shift
    explicitly if you mean it (as :func:`optimize_ring_assignment` does);
    this function does not know the FSR.
    """
    if efficiency_nm_per_mw <= 0:
        raise ValueError("efficiency_nm_per_mw must be positive")
    return abs(detune_nm) / efficiency_nm_per_mw


def expected_tuning_power_mw(
    fsr_nm: float,
    efficiency_nm_per_mw: float = TUNING_EFFICIENCY_NM_PER_MW,
    bidirectional: bool = False,
) -> float:
    """Mean per-ring heater power (mW) to lock a ring under fabrication offset.

    Fabrication variation places the as-built resonance uniformly over one
    FSR relative to the target channel.

    * ``bidirectional=False`` (default, physical for resistive heaters):
      heaters can only red-shift the resonance, so the ring must always be
      heated forward to the next resonance order at or red of the target.
      The required shift is uniform on [0, FSR) with mean **FSR/2**.
      Expected power = (FSR/2) / efficiency.
    * ``bidirectional=True`` (idealized: thermal bias point pre-budgeted, or
      a tuner that shifts both ways): lock to the *nearest* order, shift
      uniform on [0, FSR/2] with mean **FSR/4**.

    Real systems often operate between the two by pre-biasing every ring
    near mid-range; use the default for a conservative power budget.
    """
    if fsr_nm <= 0:
        raise ValueError("fsr_nm must be positive")
    detune = fsr_nm / 4.0 if bidirectional else fsr_nm / 2.0
    return tuning_power_mw(detune, efficiency_nm_per_mw)


@dataclass(frozen=True)
class RingAssignment:
    """Result of a barrel-shift channel-assignment optimization.

    Attributes
    ----------
    rotation:
        Optimal barrel shift ``r``: ring ``i`` serves channel
        ``(i + r) mod N``. Shared by the whole bank.
    per_ring_mw:
        Heater power per ring (mW) at the optimal rotation, in ring order.
    total_mw:
        Sum of ``per_ring_mw`` (mW).
    mean_mw_per_ring:
        ``total_mw / N`` (mW).
    naive_total_mw:
        Total heater power (mW) at rotation 0 (ring ``i`` serves channel
        ``i``), for comparison against the optimized assignment.
    """

    rotation: int
    per_ring_mw: tuple[float, ...]
    total_mw: float
    mean_mw_per_ring: float
    naive_total_mw: float


def optimize_ring_assignment(
    offsets_nm,
    fsr_nm: float,
    efficiency_nm_per_mw: float = TUNING_EFFICIENCY_NM_PER_MW,
    bidirectional: bool = False,
) -> RingAssignment:
    """Barrel-shift channel assignment minimizing ring-bank tuning power.

    N rings serve N channels equally spaced by ``delta = fsr_nm / N``
    within one FSR. Ring ``i`` nominally resonates at channel ``i``; its
    as-built resonance sits ``offsets_nm[i]`` nm red (+) or blue (-) of
    that nominal. Because channel labels are a WDM-plan convention, the
    bank may cyclically rotate its assignment: under barrel shift ``r``
    ring ``i`` serves channel ``(i + r) mod N``, and all N rotations are
    valid plans. The optimal rotation absorbs the common-mode fabrication
    offset nearly for free, leaving only the differential spread plus an
    FSR/(2N) quantization residual.

    Required red-shift per ring at rotation ``r`` (linear heater model
    P = shift / efficiency, the same model as :func:`tuning_power_mw` —
    but unlike that function, a blue-shift request here is charged its
    full physical wrap cost ``FSR - |d|``, not ``|d|``):

    * ``bidirectional=False`` (default, physical for resistive heaters —
      heaters only red-shift, so a blue-shift request costs the long way
      around to the next resonance order):
      ``h_i(r) = (r * delta - offsets_nm[i]) mod fsr_nm``
    * ``bidirectional=True`` (idealized two-way tuner): lock to the
      nearest order, ``h_i(r) = min(x, fsr_nm - x)`` with ``x`` the same
      modular detune.

    All N rotations are enumerated and the one with the lowest total
    power is returned (lowest rotation index on ties). Note that in the
    red-shift-only model every per-ring shift satisfies
    ``h_i(r) = r*delta - offsets_nm[i] + m*fsr_nm`` for integer ``m``, so
    rotation totals differ only by integer multiples of the FSR and exact
    ties between rotations are common, not exceptional.

    Compare :func:`expected_tuning_power_mw`, the pessimistic per-ring
    bound when the assignment is *not* optimized (offset uniform over one
    FSR, mean FSR/2 or FSR/4 of shift). Assumes identical rings on one
    bus (same FSR and tuning efficiency for all); ``r`` is shared by the
    whole bank — it relabels the WDM plan, it is not a per-ring degree of
    freedom. Architecture-level estimate, not signoff.

    Parameters
    ----------
    offsets_nm:
        Array-like of as-built resonance offsets in nm, one per ring,
        red-positive. Length N >= 1; all values must be finite.
    fsr_nm:
        Ring free spectral range in nm (> 0).
    efficiency_nm_per_mw:
        Heater tuning efficiency in nm/mW (> 0).
    bidirectional:
        Tuner sign convention, see above.
    """
    if fsr_nm <= 0:
        raise ValueError("fsr_nm must be positive")
    if efficiency_nm_per_mw <= 0:
        raise ValueError("efficiency_nm_per_mw must be positive")
    offsets = np.atleast_1d(np.asarray(offsets_nm, dtype=float))
    if offsets.ndim != 1:
        raise ValueError("offsets_nm must be a 1-D array-like")
    n = offsets.size
    if n < 1:
        raise ValueError("offsets_nm must contain at least one ring")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("offsets_nm must be finite")

    delta = fsr_nm / n
    rotations = np.arange(n)
    # x[r, i] = red-shift-only detune of ring i under barrel shift r.
    x = (rotations[:, None] * delta - offsets[None, :]) % fsr_nm
    shifts = np.minimum(x, fsr_nm - x) if bidirectional else x
    powers = shifts / efficiency_nm_per_mw
    totals = powers.sum(axis=1)
    best = int(np.argmin(totals))
    return RingAssignment(
        rotation=best,
        per_ring_mw=tuple(float(p) for p in powers[best]),
        total_mw=float(totals[best]),
        mean_mw_per_ring=float(totals[best] / n),
        naive_total_mw=float(totals[0]),
    )


def resonance_shift_nm_per_k(
    wl_um: float, ng: float, dneff_dT: float = materials.DN_DT_SI * STRIP_TE_CONFINEMENT
) -> float:
    """Thermal shift of a ring resonance, d(lambda)/dT in nm/K.

    d(lambda)/dT = lambda * (dn_eff/dT) / n_g (from the resonance condition
    n_eff L = m lambda, including first-order dispersion via n_g).

    The default dn_eff/dT scales the silicon material thermo-optic
    coefficient :data:`siphon.materials.DN_DT_SI` by the typical strip-TE
    confinement factor :data:`STRIP_TE_CONFINEMENT`. For 1.55 um and
    n_g = 4.2 this gives ~0.058 nm/K (~58 pm/K), consistent with measured
    Si microrings (~50-80 pm/K).
    """
    if wl_um <= 0 or ng <= 0:
        raise ValueError("wl_um and ng must be positive")
    return wl_um * 1e3 * dneff_dT / ng


def aggregate_crosstalk_db(per_channel_isolation_db: float, n_aggressors: int) -> float:
    """Worst-case aggregate crosstalk (dB, negative) from ``n_aggressors``.

    Incoherent power sum of ``n_aggressors`` equal-power aggressors each
    suppressed by ``per_channel_isolation_db`` (positive dB of isolation):
    XT = 10 log10(n * 10^(-iso/10)). Worst case in the sense that every
    aggressor is assumed at full power at the victim's worst filter
    rejection; coherent beating penalties are not included.
    """
    if per_channel_isolation_db < 0:
        raise ValueError("per_channel_isolation_db must be non-negative (dB of isolation)")
    if n_aggressors < 1:
        raise ValueError("n_aggressors must be >= 1")
    return 10.0 * np.log10(n_aggressors * 10.0 ** (-per_channel_isolation_db / 10.0))


def laser_grid_check(plan: ChannelPlan, ring_fsr_nm: float) -> None:
    """Check that a channel plan fits within one ring FSR.

    Raises ValueError if the plan's wavelength span exceeds ``ring_fsr_nm``
    (edge channels would alias onto adjacent resonance orders); emits a
    UserWarning if the span exceeds 90% of the FSR, leaving little guard
    band for thermal drift and fabrication offset.
    """
    if ring_fsr_nm <= 0:
        raise ValueError("ring_fsr_nm must be positive")
    span = plan.span_nm()
    if span > ring_fsr_nm:
        raise ValueError(
            f"channel span {span:.2f} nm exceeds ring FSR {ring_fsr_nm:.2f} nm; "
            "edge channels alias onto adjacent resonance orders"
        )
    if span > 0.9 * ring_fsr_nm:
        warnings.warn(
            f"channel span {span:.2f} nm uses >90% of ring FSR {ring_fsr_nm:.2f} nm; "
            "little guard band remains for drift",
            UserWarning,
            stacklevel=2,
        )
