"""Tests for etalon.fdmode: semi-vectorial finite-difference mode solver.

Physics anchors, all independent of the solver's own output:

- Wide-strip limit: a 6.0 x 0.22 um strip approaches the analytic
  three-layer slab TE0 index (2.848 at 1.55 um) from
  :func:`etalon.waveguide.slab_neffs` (exact transcendental solve).
- Grid convergence: n_eff is O(h^2); halving the pitch must move it by
  well under the accuracy claim.
- Bounds and ordering: every guided n_eff lies strictly between the
  cladding and core material indices, and mode order is by decreasing
  n_eff.
- EIM cross-check ("the punchline"): the FD solver exists so Etalon can
  bound its own EIM bias in-library; for the standard 500 x 220 nm strip
  the two must agree to well under a percent but must NOT agree exactly.
- Symmetry: the structure is mirror-symmetric in x, so TE0 is even and
  TE1 odd about x = 0 (exact on the mirror-symmetric grid).

Wavelengths and geometry in um. All solves are deterministic (no rng).
Coarse grids (dx 0.03-0.06) keep the whole file fast; the convergence
test is the only one that touches a fine grid.
"""

import functools

import numpy as np
import pytest

from etalon import materials
from etalon.fdmode import FdMode, solve_modes
from etalon.waveguide import Waveguide, slab_neffs

WL = 1.55
N_SI = materials.n_si(WL)
N_OX = materials.n_sio2(WL)


@functools.cache
def _solve(width_um, height_um, wl_um=WL, **kwargs):
    """Memoized solve so shared geometries are computed once per session."""
    return solve_modes(width_um, height_um, wl_um, **kwargs)


# --- gate 1: wide-strip slab limit ---------------------------------------------


def test_wide_strip_matches_analytic_slab():
    """A 6 um wide strip must reproduce the analytic slab TE0 within 2e-3.

    Honest accounting: at width 6.0 the *physical* lateral confinement
    already lowers n_eff by ~2.9e-3 relative to the infinite slab (EIM
    estimate), while the FD y-discretization error at this pitch is
    ~+2.2e-3, so the net agreement (~ -0.7e-3 here) reflects both the
    solver being close to the true 2D value and the strip being close to
    (not identical to) the slab.
    """
    slab_te0 = slab_neffs(N_SI, N_OX, N_OX, 0.22, WL, "TE")[0]
    fd = _solve(6.0, 0.22, dx_um=0.06, n_modes=1)[0].neff
    assert abs(fd - slab_te0) < 2e-3


# --- gate 2: grid convergence ---------------------------------------------------


def test_convergence_halving_pitch():
    """n_eff of the 500 x 220 nm strip TE0 moves < 2e-3 from dx 0.03 to 0.015.

    The scheme is O(h^2); this bounds the discretization error at the
    coarse end of the useful range and backs the ~1e-3 accuracy claim at
    the default dx = 0.02.
    """
    coarse = _solve(0.5, 0.22, dx_um=0.03, n_modes=1)[0].neff
    fine = _solve(0.5, 0.22, dx_um=0.015, n_modes=1)[0].neff
    assert abs(coarse - fine) < 2e-3


# --- gate 3: physical bounds and ordering ---------------------------------------


def test_te0_between_cladding_and_core():
    te0 = _solve(0.5, 0.22, dx_um=0.03, n_modes=1)[0].neff
    assert N_OX < te0 < N_SI


def test_te0_above_te1_when_both_guided():
    modes = _solve(0.8, 0.22, dx_um=0.03, n_modes=2)
    assert len(modes) == 2
    te0, te1 = modes
    assert N_OX < te1.neff < te0.neff < N_SI


def test_wider_guide_holds_more_modes_sorted():
    modes = _solve(1.2, 0.22, dx_um=0.03, n_modes=3)
    assert len(modes) == 3
    neffs = [m.neff for m in modes]
    assert all(a > b for a, b in zip(neffs, neffs[1:]))  # strictly decreasing
    assert all(N_OX < n < N_SI for n in neffs)


def test_narrow_strip_single_te_mode():
    # A 300 nm strip guides TE0 only; asking for 3 modes returns just 1.
    modes = _solve(0.3, 0.22, dx_um=0.03, n_modes=3)
    assert len(modes) == 1


# --- gate 4: EIM cross-validation (the punchline) --------------------------------


def test_fd_vs_eim_close_but_not_identical():
    """FD and EIM must agree loosely and disagree measurably for 500x220 TE0.

    This is the reason fdmode exists: Etalon quantifies its own EIM
    approximation instead of pointing users off-library. The EIM is known
    to overestimate n_eff for this geometry (it ignores the true 2D field
    at the corners and sidewalls); verified numerically here: FD gives
    ~2.489 at the default grid versus EIM ~2.492, i.e. FD sits *below*
    EIM by ~2.7e-3. Bounds are loose on purpose — this asserts the sign
    and the order of magnitude of the EIM bias, not a signoff number.
    """
    fd = _solve(0.5, 0.22)[0].neff  # default dx_um=0.02 grid
    eim = Waveguide(0.5, 0.22).neff(WL, "TE")
    assert abs(fd - eim) < 0.15  # same physics, same ballpark
    assert abs(fd - eim) > 1e-4  # genuinely different methods
    assert fd < eim  # EIM overestimates (verified sign for this geometry)


def test_tm_fd_vs_eim_loose_agreement():
    fd = _solve(0.5, 0.22, polarization="TM", dx_um=0.03, n_modes=1)[0].neff
    eim = Waveguide(0.5, 0.22).neff(WL, "TM")
    assert abs(fd - eim) < 0.05


# --- gate 5: mode symmetry -------------------------------------------------------


def test_te0_field_even_in_x():
    m = _solve(0.8, 0.22, dx_um=0.03, n_modes=2)[0]
    corr = np.corrcoef(m.field.ravel(), m.field[:, ::-1].ravel())[0, 1]
    assert corr > 0.99


def test_te1_field_odd_in_x():
    m = _solve(0.8, 0.22, dx_um=0.03, n_modes=2)[1]
    corr = np.corrcoef(m.field.ravel(), m.field[:, ::-1].ravel())[0, 1]
    assert corr < -0.99


# --- polarization and geometry physics -------------------------------------------


def test_tm0_below_te0():
    te = _solve(0.5, 0.22, dx_um=0.03, n_modes=1)[0].neff
    tm = _solve(0.5, 0.22, polarization="TM", dx_um=0.03, n_modes=1)[0].neff
    assert tm < te


def test_rib_neff_above_strip():
    # Adding a residual slab adds core material -> n_eff must rise.
    strip = _solve(0.5, 0.22, dx_um=0.03, n_modes=1)[0].neff
    rib = _solve(0.5, 0.22, dx_um=0.03, n_modes=1, slab_um=0.09)[0].neff
    assert rib > strip


def test_neff_decreases_with_wavelength():
    n_lo = _solve(0.5, 0.22, 1.50, dx_um=0.03, n_modes=1)[0].neff
    n_hi = _solve(0.5, 0.22, 1.60, dx_um=0.03, n_modes=1)[0].neff
    assert n_hi < n_lo


def test_sin_platform_guided():
    # A 1000 x 400 nm nitride strip guides TE0 between oxide and SiN indices.
    m = _solve(1.0, 0.4, core="sin", dx_um=0.04, n_modes=1)
    assert len(m) == 1
    assert N_OX < m[0].neff < materials.n_sin(WL)


# --- FdMode contract --------------------------------------------------------------


def test_field_shape_normalization_and_boundaries():
    m = _solve(0.5, 0.22, dx_um=0.03, n_modes=1)[0]
    assert isinstance(m, FdMode)
    assert m.polarization == "TE"
    assert m.field.shape == (len(m.y_um), len(m.x_um))
    assert np.max(np.abs(m.field)) == pytest.approx(1.0, abs=1e-12)
    # Dirichlet window: the guided field must have decayed to nothing at the
    # padded edge (pad 1.5 um >> the ~0.12 um evanescent decay length).
    edges = np.concatenate([m.field[0], m.field[-1], m.field[:, 0], m.field[:, -1]])
    assert np.max(np.abs(edges)) < 1e-3


def test_field_peak_inside_core():
    m = _solve(0.5, 0.22, dx_um=0.03, n_modes=1)[0]
    j, i = np.unravel_index(np.argmax(np.abs(m.field)), m.field.shape)
    assert abs(m.x_um[i]) < 0.25  # inside |x| < width/2
    assert 0.0 < m.y_um[j] < 0.22  # inside the core layer


def test_x_grid_mirror_symmetric():
    m = _solve(0.5, 0.22, dx_um=0.03, n_modes=1)[0]
    assert np.allclose(m.x_um, -m.x_um[::-1])


# --- input validation --------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(width_um=0.0, height_um=0.22),
        dict(width_um=-0.5, height_um=0.22),
        dict(width_um=0.5, height_um=0.0),
        dict(width_um=0.5, height_um=0.22, slab_um=0.22),  # slab >= height
        dict(width_um=0.5, height_um=0.22, slab_um=-0.05),
        dict(width_um=0.5, height_um=0.22, polarization="TEM"),
        dict(width_um=0.5, height_um=0.22, core="sio2"),  # core not above clad
        dict(width_um=0.5, height_um=0.22, n_modes=0),
        dict(width_um=0.5, height_um=0.22, dx_um=0.0),
        dict(width_um=0.5, height_um=0.22, pad_um=-1.0),
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        solve_modes(kwargs.pop("width_um"), kwargs.pop("height_um"), WL, **kwargs)
