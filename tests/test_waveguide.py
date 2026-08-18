"""Tests for etalon.waveguide: slab solver, EIM strip/rib guide, bend loss.

Reference points: 220 nm Si/SiO2 slab TE0 n_eff = 2.849 at 1.55 um; a
500x220 nm strip solved by EIM gives quasi-TE n_eff ~= 2.49 (full-vector
solvers give ~2.44 — the difference is the known EIM bias). Symmetric-slab
TE mode count follows M = 1 + floor(2 t sqrt(n_c^2 - n_s^2)/lambda).
Wavelengths and geometry in um; loss in dB.
"""

import numpy as np
import pytest

from etalon import materials
from etalon.waveguide import Waveguide, bend_loss_db_per_90deg, slab_neffs

WL = 1.55
N_SI = materials.n_si(WL)
N_OX = materials.n_sio2(WL)


# --- slab solver --------------------------------------------------------------


def test_slab_220nm_te_single_mode():
    neffs = slab_neffs(N_SI, N_OX, N_OX, 0.220, WL, "TE")
    assert len(neffs) == 1
    assert neffs[0] == pytest.approx(2.848, abs=2e-3)


def test_slab_tm0_below_te0():
    te = slab_neffs(N_SI, N_OX, N_OX, 0.220, WL, "TE")[0]
    tm = slab_neffs(N_SI, N_OX, N_OX, 0.220, WL, "TM")[0]
    assert tm < te


def test_thick_slab_multimode_ordered_and_bounded():
    neffs = slab_neffs(N_SI, N_OX, N_OX, 1.5, WL, "TE")
    assert len(neffs) > 1
    assert all(a > b for a, b in zip(neffs, neffs[1:]))  # strictly decreasing
    assert all(N_OX < n < N_SI for n in neffs)


def test_thick_slab_mode_count_matches_analytic():
    # Symmetric slab TE mode count: M = 1 + floor(2 t sqrt(n_c^2-n_s^2)/lambda).
    t = 1.5
    expected = 1 + int(np.floor(2 * t * np.sqrt(N_SI**2 - N_OX**2) / WL))
    assert len(slab_neffs(N_SI, N_OX, N_OX, t, WL, "TE")) == expected


def test_symmetric_slab_fundamental_has_no_cutoff():
    # The symmetric slab guides TE0/TM0 at any thickness; a 10 nm film still
    # returns one mode barely above the cladding index.
    for pol in ("TE", "TM"):
        neffs = slab_neffs(N_SI, N_OX, N_OX, 0.010, WL, pol)
        assert len(neffs) == 1
        assert N_OX < neffs[0] < N_OX + 0.02


def test_asymmetric_thin_slab_cut_off():
    # With air on top the fundamental acquires a cutoff: 10 nm guides nothing.
    assert slab_neffs(N_SI, 1.0, N_OX, 0.010, WL, "TM") == []
    assert slab_neffs(N_SI, 1.0, N_OX, 0.010, WL, "TE") == []


def test_core_index_below_cladding_no_modes():
    assert slab_neffs(1.3, N_OX, N_OX, 0.220, WL, "TE") == []


def test_zero_thickness_no_modes():
    assert slab_neffs(N_SI, N_OX, N_OX, 0.0, WL, "TE") == []


def test_asymmetric_slab_fewer_and_shifted_modes():
    sym = slab_neffs(N_SI, N_OX, N_OX, 0.50, WL, "TE")
    asym = slab_neffs(N_SI, 1.0, N_OX, 0.50, WL, "TE")  # air top
    assert len(asym) < len(sym)
    # Removing high-index cladding pulls every surviving mode down.
    assert all(a < s for a, s in zip(asym, sym))


def test_bad_polarization_raises():
    with pytest.raises(ValueError):
        slab_neffs(N_SI, N_OX, N_OX, 0.220, WL, "TEM")


# --- Waveguide (effective index method) ----------------------------------------


def test_strip_te_neff():
    wg = Waveguide(width_um=0.5, height_um=0.22)
    assert wg.neff(WL, "TE") == pytest.approx(2.49, abs=0.02)


def test_strip_tm_neff():
    wg = Waveguide(width_um=0.5, height_um=0.22)
    assert wg.neff(WL, "TM") == pytest.approx(1.85, abs=0.03)


def test_strip_tm_below_te():
    wg = Waveguide(width_um=0.5, height_um=0.22)
    assert wg.neff(WL, "TM") < wg.neff(WL, "TE")


def test_strip_group_index():
    wg = Waveguide(width_um=0.5, height_um=0.22)
    assert wg.group_index(WL, "TE") == pytest.approx(4.0, abs=0.15)


def test_group_index_exceeds_neff():
    wg = Waveguide(width_um=0.5, height_um=0.22)
    assert wg.group_index(WL, "TE") > wg.neff(WL, "TE")


def test_neff_decreases_with_narrower_width():
    n_500 = Waveguide(0.5, 0.22).neff(WL, "TE")
    n_400 = Waveguide(0.4, 0.22).neff(WL, "TE")
    n_350 = Waveguide(0.35, 0.22).neff(WL, "TE")
    assert n_350 < n_400 < n_500


def test_wide_guide_approaches_slab_limit():
    # As width -> inf the EIM n_eff converges to the vertical slab TE0 index.
    slab = slab_neffs(N_SI, N_OX, N_OX, 0.22, WL, "TE")[0]
    wide = Waveguide(10.0, 0.22).neff(WL, "TE")
    assert wide < slab
    assert slab - wide < 5e-3


def test_rib_neff_above_strip():
    strip = Waveguide(0.5, 0.22).neff(WL, "TE")
    rib = Waveguide(0.5, 0.22, slab_um=0.09).neff(WL, "TE")
    assert rib > strip


def test_neff_monotonic_in_wavelength():
    wg = Waveguide(0.5, 0.22)
    wls = np.linspace(1.5, 1.6, 11)
    neffs = [wg.neff(float(w), "TE") for w in wls]
    assert all(a > b for a, b in zip(neffs, neffs[1:]))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(width_um=0.0, height_um=0.22),
        dict(width_um=-0.5, height_um=0.22),
        dict(width_um=0.5, height_um=0.0),
        dict(width_um=0.5, height_um=0.22, slab_um=0.22),  # slab >= height
        dict(width_um=0.5, height_um=0.22, slab_um=-0.05),
    ],
)
def test_invalid_geometry_raises(kwargs):
    with pytest.raises(ValueError):
        Waveguide(**kwargs)


def test_no_mode_raises_valueerror():
    # A 30 nm Si film with SiN top cladding (asymmetric) has no vertical slab
    # mode at 1.55 um, so the EIM must refuse rather than fabricate a mode.
    wg = Waveguide(width_um=0.5, height_um=0.03, cladding="sin")
    with pytest.raises(ValueError):
        wg.neff(WL, "TE")
    with pytest.raises(ValueError):
        wg.neff(WL, "TM")


def test_dispersion_finite():
    d = Waveguide(0.5, 0.22).dispersion_ps_nm_km(WL, "TE")
    assert np.isfinite(d)
    assert abs(d) < 5000.0  # plausible magnitude for a sub-um strip guide


# --- bend loss ------------------------------------------------------------------


def test_bend_loss_anchor_point():
    # Anchored to Vlasov & McNab 2004: ~0.086 dB per 90 deg at R = 1 um.
    assert bend_loss_db_per_90deg(1.0) == pytest.approx(0.086, rel=0.05)


def test_bend_loss_positive_and_decreasing():
    radii = [0.5, 1.0, 2.0, 5.0, 10.0]
    losses = [bend_loss_db_per_90deg(r) for r in radii]
    assert all(loss > 0 for loss in losses)
    assert all(a > b for a, b in zip(losses, losses[1:]))


def test_bend_loss_nonpositive_radius_raises():
    with pytest.raises(ValueError):
        bend_loss_db_per_90deg(0.0)
    with pytest.raises(ValueError):
        bend_loss_db_per_90deg(-1.0)
