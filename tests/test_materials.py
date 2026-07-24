"""Tests for siphon.materials: Sellmeier fits, dispatch, and group index.

Reference values at 1.55 um: Si n=3.4776 (Li 1980), SiO2 n=1.4440
(Malitson 1965), Si3N4 n=1.9963 (Luke 2015). Wavelengths in um.
"""

import numpy as np
import pytest

from siphon import materials


# --- refractive index values at 1.55 um -----------------------------------


def test_n_si_1550():
    assert materials.n_si(1.55) == pytest.approx(3.4776, abs=1e-3)


def test_n_sio2_1550():
    assert materials.n_sio2(1.55) == pytest.approx(1.4440, abs=1e-3)


def test_n_sin_1550():
    assert materials.n_sin(1.55) == pytest.approx(1.9963, abs=1e-3)


def test_index_ordering():
    # Si > Si3N4 > SiO2 across the C-band.
    for wl in (1.53, 1.55, 1.565):
        assert materials.n_si(wl) > materials.n_sin(wl) > materials.n_sio2(wl)


def test_normal_dispersion():
    # All three materials have dn/dlambda < 0 in the telecom window.
    for fn in (materials.n_si, materials.n_sio2, materials.n_sin):
        assert fn(1.50) > fn(1.60)


# --- scalar/vector behaviour -----------------------------------------------


def test_scalar_input_returns_float():
    assert isinstance(materials.n_si(1.55), float)
    assert isinstance(materials.n_sio2(1.55), float)
    assert isinstance(materials.n_sin(1.55), float)


def test_vector_input_returns_array():
    wl = np.array([1.50, 1.55, 1.60])
    for fn in (materials.n_si, materials.n_sio2, materials.n_sin):
        out = fn(wl)
        assert isinstance(out, np.ndarray)
        assert out.shape == wl.shape
        # elementwise agreement with scalar evaluation
        np.testing.assert_allclose(out, [fn(float(w)) for w in wl], rtol=1e-12)


# --- range and input validation --------------------------------------------


@pytest.mark.parametrize(
    "fn,bad_wl",
    [
        (materials.n_si, 1.0),   # Si valid 1.2-14
        (materials.n_si, 15.0),
        (materials.n_sio2, 0.1),  # SiO2 valid 0.21-6.7
        (materials.n_sio2, 7.0),
        (materials.n_sin, 0.2),   # Si3N4 valid 0.31-5.5
        (materials.n_sin, 6.0),
    ],
)
def test_out_of_range_raises(fn, bad_wl):
    with pytest.raises(ValueError):
        fn(bad_wl)


def test_out_of_range_in_array_raises():
    with pytest.raises(ValueError):
        materials.n_si(np.array([1.55, 20.0]))


def test_nonpositive_wavelength_raises():
    with pytest.raises(ValueError):
        materials.n_si(0.0)
    with pytest.raises(ValueError):
        materials.n_sio2(-1.55)


# --- name dispatch ----------------------------------------------------------


def test_index_dispatch():
    assert materials.index("si", 1.55) == materials.n_si(1.55)
    assert materials.index("SiO2", 1.55) == materials.n_sio2(1.55)  # case-insensitive
    assert materials.index("sin", 1.55) == materials.n_sin(1.55)


def test_unknown_material_raises_keyerror():
    with pytest.raises(KeyError):
        materials.index("inp", 1.55)


# --- group index -------------------------------------------------------------


def test_group_index_si():
    assert materials.group_index_material("si", 1.55) == pytest.approx(3.60, abs=0.05)


def test_group_index_exceeds_phase_index():
    # n_g = n - lambda dn/dlambda > n when dispersion is normal.
    for name in ("si", "sio2", "sin"):
        assert materials.group_index_material(name, 1.55) > materials.index(name, 1.55)


# --- thermo-optic constants ---------------------------------------------------


def test_thermo_optic_constants_positive():
    assert materials.DN_DT_SI > 0
    assert materials.DN_DT_SIO2 > 0
    assert materials.DN_DT_SIN > 0
    # Silicon's TO coefficient dominates by roughly an order of magnitude.
    assert materials.DN_DT_SI > materials.DN_DT_SIN > materials.DN_DT_SIO2
