"""Waveguide mode solvers.

Two levels of model, both fast and dependency-free:

1. ``slab_neffs`` — exact transcendental solution of the three-layer slab
   waveguide (TE and TM), the workhorse of quick photonic reasoning.
2. ``Waveguide`` — strip/rib waveguide solved with the effective index
   method (EIM): vertical slab first, then the horizontal problem with the
   swapped polarization. Accurate to roughly the percent level for n_eff of
   well-guided modes; use a full vectorial solver for final signoff.

Near modal cutoff the EIM is biased *conservative* (it declares modes guided
earlier than a full-vectorial solver: e.g. TE1 first appears at ~316 nm strip
width under EIM versus ~450-500 nm full-vectorial). When a solved mode sits
within ``NEAR_CUTOFF_NEFF_MARGIN`` of the lateral cladding index the solver
emits an :class:`EimAccuracyWarning` rather than silently returning a number
that a vectorial solver might not reproduce. Architecture-level, not signoff.

Conventions: wavelengths and dimensions in um; quasi-TE means dominant
E-field parallel to the wafer plane.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from . import materials


class EimAccuracyWarning(UserWarning):
    """Warns that an EIM result is in the regime where the method is least accurate.

    Emitted when the horizontally-solved effective index sits within
    ``NEAR_CUTOFF_NEFF_MARGIN`` of the side/cladding effective index, i.e. the
    mode is marginally guided. Near cutoff the EIM is biased early/conservative
    (it guides modes a full-vectorial solver would find cut off), so the
    returned n_eff — and the mode's very existence — should be confirmed with a
    full-vectorial solver. The numerical result is returned unchanged.
    """


#: Margin (dimensionless, in effective-index units) below which a mode counts
#: as marginally guided: when the horizontally-solved n_eff sits within this of
#: the side/cladding effective index, the mode is barely confined laterally —
#: exactly where the EIM is least trustworthy and biased early/conservative
#: (validated example: TE1 first guided at ~316 nm strip width under EIM versus
#: ~450-500 nm full-vectorial). Crossing it triggers :class:`EimAccuracyWarning`.
NEAR_CUTOFF_NEFF_MARGIN = 5e-3


def _slab_dispersion(neff, k0, n_core, n_top, n_bot, thickness, m, polarization):
    """Zero of this function <=> guided slab mode m at n_eff = neff."""
    kappa = k0 * np.sqrt(n_core**2 - neff**2)
    gamma = k0 * np.sqrt(max(neff**2 - n_top**2, 0.0))
    delta = k0 * np.sqrt(max(neff**2 - n_bot**2, 0.0))
    if polarization == "TE":
        phi_top = np.arctan2(gamma, kappa)
        phi_bot = np.arctan2(delta, kappa)
    elif polarization == "TM":
        phi_top = np.arctan2((n_core / n_top) ** 2 * gamma, kappa)
        phi_bot = np.arctan2((n_core / n_bot) ** 2 * delta, kappa)
    else:
        raise ValueError("polarization must be 'TE' or 'TM'")
    return kappa * thickness - phi_top - phi_bot - m * np.pi


def slab_neffs(
    n_core: float,
    n_top: float,
    n_bot: float,
    thickness_um: float,
    wavelength_um: float,
    polarization: str = "TE",
    max_modes: int = 16,
) -> list[float]:
    """Effective indices of all guided modes of a three-layer slab.

    Returns a list sorted from fundamental (highest n_eff) down; empty list
    if the slab guides no mode at this wavelength/polarization. Index
    contrasts below the ~2e-9 root-bracketing epsilon are numerically
    degenerate and report no modes (real platforms sit >= 1e-3 above it).
    """
    if thickness_um <= 0:
        return []
    if n_core <= max(n_top, n_bot):
        return []
    k0 = 2.0 * np.pi / wavelength_um
    n_min = max(n_top, n_bot)
    eps = 1e-9
    lo, hi = n_min + eps, n_core - eps
    if lo >= hi:  # contrast below bracketing epsilon: degenerate, no solve
        return []
    out: list[float] = []
    for m in range(max_modes):
        f_lo = _slab_dispersion(lo, k0, n_core, n_top, n_bot, thickness_um, m, polarization)
        f_hi = _slab_dispersion(hi, k0, n_core, n_top, n_bot, thickness_um, m, polarization)
        if f_lo <= 0.0:
            break  # this and all higher-order modes are cut off
        if f_hi >= 0.0:  # numerically degenerate bracket; shouldn't happen
            break
        neff = brentq(
            _slab_dispersion,
            lo,
            hi,
            args=(k0, n_core, n_top, n_bot, thickness_um, m, polarization),
            xtol=1e-12,
        )
        out.append(float(neff))
    return out


@dataclass(frozen=True)
class Waveguide:
    """Strip or rib waveguide on SOI (or SiN) solved by the effective index method.

    Parameters
    ----------
    width_um, height_um : core cross-section.
    slab_um : residual slab thickness for a rib; 0 for a fully etched strip.
    core, cladding, box : material names understood by :mod:`siphon.materials`.
        ``cladding`` is the top/side cladding, ``box`` the buried oxide.
    """

    width_um: float
    height_um: float
    slab_um: float = 0.0
    core: str = "si"
    cladding: str = "sio2"
    box: str = "sio2"

    def __post_init__(self):
        if self.width_um <= 0 or self.height_um <= 0:
            raise ValueError("width_um and height_um must be positive")
        if not 0 <= self.slab_um < self.height_um:
            raise ValueError("slab_um must satisfy 0 <= slab_um < height_um")

    # --- effective index -------------------------------------------------

    def neff(self, wavelength_um: float, mode: str = "TE") -> float:
        """Effective index of the fundamental quasi-TE or quasi-TM mode.

        Raises ValueError if the structure guides no such mode.

        If the mode is marginally guided (n_eff within
        ``NEAR_CUTOFF_NEFF_MARGIN`` of the lateral cladding/slab effective
        index), an :class:`EimAccuracyWarning` is issued — the EIM is biased
        conservative near cutoff and the result should be checked with a
        full-vectorial solver. Because results are memoized (``lru_cache``),
        the warning fires only on the *first* computation for a given
        (geometry, wavelength, mode); repeated calls return the cached value
        silently.
        """
        return self._neff_cached(round(float(wavelength_um), 9), mode)

    @lru_cache(maxsize=4096)
    def _neff_cached(self, wavelength_um: float, mode: str) -> float:
        if mode.upper() not in ("TE", "TM"):
            raise ValueError(f"mode must be 'TE' or 'TM', got {mode!r}")
        n_core = materials.index(self.core, wavelength_um)
        n_clad = materials.index(self.cladding, wavelength_um)
        n_box = materials.index(self.box, wavelength_um)

        # EIM polarization bookkeeping: quasi-TE of the channel guide uses the
        # TE slab solution vertically, then TM horizontally (and vice versa).
        vert_pol = "TE" if mode.upper() == "TE" else "TM"
        horiz_pol = "TM" if mode.upper() == "TE" else "TE"

        core_slab = slab_neffs(n_core, n_clad, n_box, self.height_um, wavelength_um, vert_pol)
        if not core_slab:
            raise ValueError(
                f"no vertical slab mode for {self!r} at {wavelength_um} um ({mode})"
            )
        n_center = core_slab[0]

        if self.slab_um > 0:
            side_slab = slab_neffs(
                n_core, n_clad, n_box, self.slab_um, wavelength_um, vert_pol
            )
            n_side = side_slab[0] if side_slab else max(n_clad, n_box)
        else:
            n_side = n_clad

        if n_center <= n_side:
            raise ValueError(f"no lateral confinement for {self!r} ({mode})")
        horiz = slab_neffs(n_center, n_side, n_side, self.width_um, wavelength_um, horiz_pol)
        if not horiz:
            raise ValueError(f"no guided mode for {self!r} at {wavelength_um} um ({mode})")
        margin = horiz[0] - n_side
        if margin < NEAR_CUTOFF_NEFF_MARGIN:
            warnings.warn(
                f"{self!r}: quasi-{mode.upper()} mode at {wavelength_um} um is "
                f"marginally guided (n_eff - n_side = {margin:.2e} < "
                f"{NEAR_CUTOFF_NEFF_MARGIN:.0e}). The EIM is biased "
                "early/conservative near cutoff; confirm this mode with a "
                "full-vectorial solver before relying on it.",
                EimAccuracyWarning,
                stacklevel=3,
            )
        return horiz[0]

    # --- dispersion ------------------------------------------------------

    def group_index(self, wavelength_um: float, mode: str = "TE", dwl_um: float = 1e-4) -> float:
        """Group index n_g = n_eff - lambda d(n_eff)/d(lambda).

        Includes both waveguide and material dispersion (materials are
        re-evaluated at each finite-difference point).
        """
        n0 = self.neff(wavelength_um, mode)
        n_p = self.neff(wavelength_um + dwl_um, mode)
        n_m = self.neff(wavelength_um - dwl_um, mode)
        dn_dwl = (n_p - n_m) / (2.0 * dwl_um)
        return float(n0 - wavelength_um * dn_dwl)

    def dispersion_ps_nm_km(
        self, wavelength_um: float, mode: str = "TE", dwl_um: float = 5e-4
    ) -> float:
        """Chromatic dispersion parameter D = -(lambda/c) d^2 n_eff / d lambda^2.

        Returned in the conventional units of ps/(nm*km).
        """
        n_p = self.neff(wavelength_um + dwl_um, mode)
        n_0 = self.neff(wavelength_um, mode)
        n_m = self.neff(wavelength_um - dwl_um, mode)
        d2n = (n_p - 2.0 * n_0 + n_m) / dwl_um**2  # 1/um^2
        c_um_per_ps = 2.99792458e2  # speed of light in um/ps
        d_ps_per_um2 = -(wavelength_um / c_um_per_ps) * d2n  # ps/um^2
        # convert ps per (um of bandwidth) per (um of length) -> ps/(nm*km)
        return float(d_ps_per_um2 * 1e-3 * 1e9)


def bend_loss_db_per_90deg(radius_um: float) -> float:
    """HEURISTIC pure-bend loss per 90-degree turn for a 220 nm strip Si waveguide.

    Exponential fit anchored to published measurements (Vlasov & McNab 2004:
    ~0.086 dB at R=1 um, ~0.013 dB at R=2 um). Order-of-magnitude guidance
    only; real bend loss depends strongly on geometry, wavelength, and
    sidewall roughness.
    """
    if radius_um <= 0:
        raise ValueError("radius must be positive")
    a, b = 0.57, 1.89  # loss = a * exp(-b * R[um]) dB per 90 degrees
    return a * np.exp(-b * radius_um)
