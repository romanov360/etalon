"""Tests for the near-cutoff EIM accuracy diagnostic in etalon.waveguide.

The EIM is biased conservative near modal cutoff (it guides modes earlier
than full-vectorial solvers: TE1 appears at ~316 nm strip width under EIM
versus ~450-500 nm full-vectorial). The solver flags marginally guided modes
— n_eff within NEAR_CUTOFF_NEFF_MARGIN of the lateral cladding effective
index — with EimAccuracyWarning, without changing any returned value.

Trigger geometries were located by independently scanning the margin
n_eff - n_side: a 30 nm-wide strip sits ~3.1e-3 above SiO2 (below the 5e-3
margin) while the standard 500x220 nm strip sits ~1.05 above it. The pinned
n_eff value below was computed with the pre-diagnostic code.
"""

import warnings

import pytest

from etalon import materials
from etalon.waveguide import (
    NEAR_CUTOFF_NEFF_MARGIN,
    EimAccuracyWarning,
    Waveguide,
)

WL = 1.55

# neff(1.55, "TE") of the 500x220 nm strip computed BEFORE the diagnostic was
# added — the warning must not perturb the numerics by even one ulp.
NEFF_STRIP_500x220_TE_PRE_CHANGE = 2.491857436485278


def test_well_guided_strip_no_warning():
    # Standard 500x220 nm SOI strip: n_eff - n_clad ~ 1.05, nowhere near the
    # 5e-3 margin, so computing it must not emit EimAccuracyWarning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", EimAccuracyWarning)
        Waveguide(width_um=0.5, height_um=0.22).neff(WL, "TE")


def test_marginally_guided_narrow_strip_warns():
    # A 30 nm-wide strip's quasi-TE mode sits ~3.1e-3 above the SiO2 cladding
    # index (checked independently against materials.index), inside the margin.
    wg = Waveguide(width_um=0.030, height_um=0.22)
    with pytest.warns(EimAccuracyWarning):
        neff = wg.neff(WL, "TE")
    # Confirm this geometry really is in the marginal regime the warning claims.
    n_clad = materials.index("sio2", WL)
    assert 0.0 < neff - n_clad < NEAR_CUTOFF_NEFF_MARGIN


def test_marginally_guided_shallow_rib_warns():
    # A shallow-etch rib (10 nm etch: 220 nm core over a 210 nm slab) barely
    # confines laterally: n_eff sits ~2.2e-3 above the side-slab index.
    wg = Waveguide(width_um=0.3, height_um=0.22, slab_um=0.21)
    with pytest.warns(EimAccuracyWarning):
        wg.neff(WL, "TE")


def test_neff_bit_identical_to_pre_change_value():
    # The diagnostic must be warning-only: the returned value equals the
    # value computed by the pre-change code (pinned above) to ~1e-9.
    neff = Waveguide(width_um=0.5, height_um=0.22).neff(WL, "TE")
    assert neff == pytest.approx(NEFF_STRIP_500x220_TE_PRE_CHANGE, abs=1e-9)


def test_warning_message_mentions_vectorial():
    # Distinct geometry from the other trigger tests: equal (frozen, hashable)
    # Waveguide instances share an lru_cache entry, and the warning fires only
    # on the first computation for a given (geometry, wavelength, mode).
    wg = Waveguide(width_um=0.032, height_um=0.22)
    with pytest.warns(EimAccuracyWarning, match="vectorial"):
        wg.neff(WL, "TE")


def test_warning_fires_on_first_computation_only():
    # Documented lru_cache behavior: a repeated call returns the memoized
    # value without re-warning.
    wg = Waveguide(width_um=0.034, height_um=0.22)
    with pytest.warns(EimAccuracyWarning):
        first = wg.neff(WL, "TE")
    with warnings.catch_warnings():
        warnings.simplefilter("error", EimAccuracyWarning)
        assert wg.neff(WL, "TE") == first


def test_margin_constant_value():
    # The margin is part of the documented contract (5e-3 in effective-index
    # units); a silent change would move every trigger threshold.
    assert NEAR_CUTOFF_NEFF_MARGIN == 5e-3
