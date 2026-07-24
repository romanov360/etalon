"""Refractive index models for common silicon-photonics materials.

All models take vacuum wavelength in micrometers (um) and return the real
refractive index. Valid ranges are noted per model; outside them a ValueError
is raised rather than silently extrapolating.

Thermo-optic coefficients (dn/dT, 1/K, near 1.55 um and room temperature) are
provided as module constants.
"""

from __future__ import annotations

import numpy as np

# Thermo-optic coefficients near 1550 nm, 300 K (1/K)
DN_DT_SI = 1.86e-4
DN_DT_SIO2 = 0.95e-5
DN_DT_SIN = 2.45e-5


def _as_array(wavelength_um) -> np.ndarray:
    wl = np.asarray(wavelength_um, dtype=float)
    if np.any(wl <= 0):
        raise ValueError("wavelength must be positive (in um)")
    return wl


def _check_range(wl: np.ndarray, lo: float, hi: float, name: str) -> None:
    if np.any(wl < lo) or np.any(wl > hi):
        raise ValueError(
            f"{name} model valid for {lo}-{hi} um; got wavelength outside range"
        )


def n_si(wavelength_um):
    """Refractive index of crystalline silicon (Li 1980 fit, 1.2-14 um).

    n(1.55 um) ~= 3.476.
    """
    wl = _as_array(wavelength_um)
    _check_range(wl, 1.2, 14.0, "Si")
    wl2 = wl**2
    l1sq = 1.1071**2
    n2 = 11.6858 + 0.939816 / wl2 + 8.10461e-3 * l1sq / (wl2 - l1sq)
    return np.sqrt(n2) if np.ndim(wavelength_um) else float(np.sqrt(n2))


def n_sio2(wavelength_um):
    """Refractive index of fused silica (Malitson 1965, 0.21-6.7 um).

    n(1.55 um) ~= 1.444.
    """
    wl = _as_array(wavelength_um)
    _check_range(wl, 0.21, 6.7, "SiO2")
    wl2 = wl**2
    n2 = (
        1.0
        + 0.6961663 * wl2 / (wl2 - 0.0684043**2)
        + 0.4079426 * wl2 / (wl2 - 0.1162414**2)
        + 0.8974794 * wl2 / (wl2 - 9.896161**2)
    )
    return np.sqrt(n2) if np.ndim(wavelength_um) else float(np.sqrt(n2))


def n_sin(wavelength_um):
    """Refractive index of LPCVD stoichiometric Si3N4 (Luke 2015, 0.31-5.5 um).

    n(1.55 um) ~= 1.996.
    """
    wl = _as_array(wavelength_um)
    _check_range(wl, 0.31, 5.5, "Si3N4")
    wl2 = wl**2
    n2 = (
        1.0
        + 3.0249 * wl2 / (wl2 - 0.1353406**2)
        + 40314.0 * wl2 / (wl2 - 1239.842**2)
    )
    return np.sqrt(n2) if np.ndim(wavelength_um) else float(np.sqrt(n2))


MATERIALS = {
    "si": n_si,
    "sio2": n_sio2,
    "sin": n_sin,
}


def index(material: str, wavelength_um):
    """Look up refractive index by material name ('si', 'sio2', 'sin')."""
    key = material.lower()
    if key not in MATERIALS:
        raise KeyError(f"unknown material {material!r}; known: {sorted(MATERIALS)}")
    return MATERIALS[key](wavelength_um)


def group_index_material(material: str, wavelength_um: float, dwl_um: float = 1e-4) -> float:
    """Material group index n_g = n - lambda * dn/dlambda (central difference)."""
    n = index(material, wavelength_um)
    dn = (index(material, wavelength_um + dwl_um) - index(material, wavelength_um - dwl_um)) / (
        2.0 * dwl_um
    )
    return float(n - wavelength_um * dn)
