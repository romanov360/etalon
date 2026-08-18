"""Regression tests for the 2026-07-24 external validation pass.

Findings and derivations: docs/research/validation-notes-2026-07-24.md.
F1: RIN noise must be evaluated at P_top = OMA * ER/(ER-1), not at OMA.
F2: Circuit.s_params solves all wavelengths as one stacked system.
F3: wdm's default dn_eff/dT derives from materials.DN_DT_SI.
F4: slab_neffs handles degenerate index contrast without warnings.
"""

import inspect
import math
import warnings

import numpy as np
import pytest

from etalon import components, link, materials, wdm
from etalon.circuit import Circuit
from etalon.waveguide import slab_neffs


# --- F1: RIN penalty evaluated at the true top level -------------------------


def test_rin_penalty_matches_implicit_top_eye_equation_at_finite_er():
    # Independent derivation: solve OMA/(M-1) = 2q sqrt(sig^2 + (P_top r)^2)
    # with P_top = OMA * ER/(ER-1) and compare against the closed form.
    q = link.q_from_ber(2.4e-4, 4)
    er_db, rate, rin, m = 5.0, 53.125, -140.0, 4
    er = 10.0 ** (er_db / 10.0)
    r0 = math.sqrt(10.0 ** (rin / 10.0) * 0.75 * rate * 1e9)
    sig = 1.0  # normalized thermal rms; penalty is independent of it
    oma0 = 2.0 * q * (m - 1) * sig

    def gap(oma):
        p_top = oma * er / (er - 1.0)
        return oma / (m - 1) - 2.0 * q * math.hypot(sig, p_top * r0)

    lo, hi = oma0, 10.0 * oma0
    for _ in range(200):  # bisection, no scipy needed here
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    implicit = 10.0 * math.log10(lo / oma0)
    closed = link.rin_penalty_db(rin, rate, q, m, extinction_ratio_db=er_db)
    assert closed == pytest.approx(implicit, abs=1e-6)


def test_rin_penalty_finite_er_exceeds_infinite_er():
    q = link.q_from_ber(2.4e-4, 4)
    at_oma = link.rin_penalty_db(-140.0, 53.125, q, 4)  # default: infinite ER
    at_top = link.rin_penalty_db(-140.0, 53.125, q, 4, extinction_ratio_db=5.0)
    assert at_top > at_oma  # P_top > OMA whenever ER is finite
    # ER 5 dB gives P_top/OMA = 1.46: penalty roughly doubles at this preset
    assert at_top == pytest.approx(0.961, abs=0.005)
    assert at_oma == pytest.approx(0.397, abs=0.005)


def test_link_budget_charges_rin_at_modulator_extinction_ratio():
    lb = link.preset_pluggable_dr4()
    expected = link.rin_penalty_db(
        lb.laser.rin_db_hz,
        lb.signaling.rate_gbd,
        lb.signaling.q_factor,
        lb.signaling.levels,
        extinction_ratio_db=lb.modulator.extinction_ratio_db,
    )
    assert lb.rin_penalty_db == pytest.approx(expected, abs=1e-12)
    assert lb.rin_penalty_db > link.rin_penalty_db(
        lb.laser.rin_db_hz,
        lb.signaling.rate_gbd,
        lb.signaling.q_factor,
        lb.signaling.levels,
    )


def test_rin_penalty_rejects_nonpositive_er():
    q = link.q_from_ber(2.4e-4, 2)
    with pytest.raises(ValueError):
        link.rin_penalty_db(-145.0, 32.0, q, 2, extinction_ratio_db=0.0)


# --- F2: batched circuit solve keeps the per-wavelength contract -------------


def _composed_add_drop(loss_db_per_cm=2.0):
    c = Circuit()
    c.add("dc1", components.DirectionalCoupler(coupling=0.2))
    c.add("dc2", components.DirectionalCoupler(coupling=0.2))
    half = dict(length_um=30.0, neff0=2.4, ng=4.2, loss_db_per_cm=loss_db_per_cm)
    c.add("top", components.Straight(**half))
    c.add("bot", components.Straight(**half))
    c.connect(("dc1", "out1"), ("top", "in"))
    c.connect(("top", "out"), ("dc2", "in1"))
    c.connect(("dc2", "out1"), ("bot", "in"))
    c.connect(("bot", "out"), ("dc1", "in1"))
    c.expose("in", ("dc1", "in0"))
    c.expose("through", ("dc1", "out0"))
    c.expose("add", ("dc2", "in0"))
    c.expose("drop", ("dc2", "out0"))
    return c


def test_batched_solve_matches_single_wavelength_slices():
    # Solving a sweep in one call must equal solving each wavelength alone.
    c = _composed_add_drop()
    wl = np.linspace(1.549, 1.551, 101)
    batched = c.s_params(wl)
    for k in (0, 37, 100):
        np.testing.assert_allclose(
            batched[k], c.s_params(np.array([wl[k]]))[0], atol=1e-13
        )


class _GainLoop:
    """Stub 2-port: t(wl) = 2 exp(1j 10pi (wl - 1.55)); resonant only at 1.55."""

    ports = ("in", "out")

    def s_params(self, wl):
        wl = np.asarray(wl, dtype=float)
        s = np.zeros((len(wl), 2, 2), dtype=complex)
        s[:, 0, 1] = s[:, 1, 0] = 2.0 * np.exp(1j * 10.0 * np.pi * (wl - 1.55))
        return s


def test_batched_solve_reports_first_singular_wavelength():
    # r = 0.5 coupler with a gain-2 loop: 1 - r*t = 0 exactly at wl = 1.55,
    # the middle of the sweep — the raise must name that index, not index 0.
    c = Circuit()
    c.add("dc", components.DirectionalCoupler(coupling=0.75))
    c.add("loop", _GainLoop())
    c.connect(("dc", "out1"), ("loop", "in"))
    c.connect(("loop", "out"), ("dc", "in1"))
    c.expose("in", ("dc", "in0"))
    c.expose("out", ("dc", "out0"))
    with pytest.raises(np.linalg.LinAlgError, match="wavelength index 1"):
        c.s_params(np.array([1.50, 1.55, 1.60]))


# --- F3: one source of truth for silicon dn/dT --------------------------------


def test_wdm_default_dneff_dt_derives_from_materials_constant():
    default = inspect.signature(wdm.resonance_shift_nm_per_k).parameters[
        "dneff_dT"
    ].default
    assert default == materials.DN_DT_SI * wdm.STRIP_TE_CONFINEMENT
    assert 0.0 < wdm.STRIP_TE_CONFINEMENT <= 1.0


# --- F4: degenerate index contrast is quiet and deliberate --------------------


def test_slab_neffs_degenerate_contrast_returns_empty_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        for dn in (0.0, 5e-10, 1e-9, 2e-9):
            assert slab_neffs(1.5 + dn, 1.5, 1.5, 10.0, 1.55) == []
    # just above the epsilon regime a symmetric slab guides its fundamental
    assert len(slab_neffs(1.5 + 1e-3, 1.5, 1.5, 10.0, 1.55)) >= 1
