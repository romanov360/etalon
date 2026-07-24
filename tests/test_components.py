"""Tests for siphon.components against analytic/known values."""

import numpy as np
import pytest

from siphon.components import (
    MZI,
    DirectionalCoupler,
    GratingCoupler,
    PhaseShifter,
    RingAddDrop,
    RingAllPass,
    Straight,
    ThermalPhaseShifter,
    YBranch,
    finesse,
    loaded_q,
    ring_fsr_um,
)
from siphon.waveguide import Waveguide

WL = np.linspace(1.50, 1.60, 401)
NEFF0, NG, WL0 = 2.4, 4.2, 1.55


def assert_reciprocal(s):
    assert np.allclose(s, np.swapaxes(s, 1, 2))


def assert_unitary(s, atol=1e-12):
    n = s.shape[-1]
    prod = s @ np.conj(np.swapaxes(s, 1, 2))
    assert np.allclose(prod, np.eye(n), atol=atol)


def resonance_wl(m, circumference_um, neff0=NEFF0, ng=NG, wl0=WL0):
    """Closed-form resonance of the linearized-dispersion ring.

    phi = 2*pi*L*(ng/wl + s) with s = (neff0 - ng)/wl0, so phi = 2*pi*m at
    wl = ng*L / (m - s*L).
    """
    s = (neff0 - ng) / wl0
    return ng * circumference_um / (m - s * circumference_um)


# --- Straight ---------------------------------------------------------------


def test_straight_phase_hand_computed():
    st = Straight(length_um=100.0, neff0=NEFF0, ng=NG, wl0_um=WL0)
    s = st.s_params(np.array([1.55, 1.60]))
    # at wl0 the linearized neff is exactly neff0
    expected0 = np.exp(-1j * 2 * np.pi * 2.4 * 100.0 / 1.55)
    assert s[0, 1, 0] == pytest.approx(expected0, abs=1e-12)
    # at 1.60 um: neff = 2.4 + 0.05*(2.4-4.2)/1.55
    neff = 2.4 + 0.05 * (2.4 - 4.2) / 1.55
    expected1 = np.exp(-1j * 2 * np.pi * neff * 100.0 / 1.60)
    assert s[1, 1, 0] == pytest.approx(expected1, abs=1e-12)
    assert_reciprocal(s)
    assert np.allclose(np.abs(s[:, 0, 0]), 0.0)  # no reflection


def test_straight_loss_and_unitarity():
    # 1 cm at 2 dB/cm -> |S21|^2 = 10^-0.2
    st = Straight(10000.0, NEFF0, NG, loss_db_per_cm=2.0)
    s = st.s_params(np.array([1.55]))
    assert np.abs(s[0, 1, 0]) ** 2 == pytest.approx(10 ** (-0.2), rel=1e-12)
    assert_unitary(Straight(123.4, NEFF0, NG).s_params(WL))


def test_straight_from_waveguide():
    wg = Waveguide(width_um=0.5, height_um=0.22)
    st = Straight.from_waveguide(wg, length_um=50.0, loss_db_per_cm=2.0, wl0_um=1.55)
    assert st.neff0 == pytest.approx(wg.neff(1.55, "TE"), rel=1e-12)
    assert st.ng == pytest.approx(wg.group_index(1.55, "TE"), rel=1e-9)
    assert st.loss_db_per_cm == 2.0 and st.length_um == 50.0


# --- DirectionalCoupler -----------------------------------------------------


def test_dc_values_and_unitarity():
    dc = DirectionalCoupler(coupling=0.3)
    s = dc.s_params(np.array([1.55]))
    assert s[0, 2, 0] == pytest.approx(np.sqrt(0.7))  # in0 -> out0 through
    assert s[0, 3, 0] == pytest.approx(1j * np.sqrt(0.3))  # in0 -> out1 cross
    assert np.allclose(s[0, :2, :2], 0.0)  # no reflection / in-in coupling
    assert_reciprocal(s)
    assert_unitary(s)
    assert_unitary(DirectionalCoupler(0.5).s_params(WL))


def test_dc_loss_scales_power():
    s = DirectionalCoupler(coupling=0.5, loss_db=1.0).s_params(np.array([1.55]))
    total = np.sum(np.abs(s[0, :, 0]) ** 2)
    assert total == pytest.approx(10 ** (-0.1), rel=1e-12)


# --- YBranch ----------------------------------------------------------------


def test_ybranch_split_and_power_conservation():
    s = YBranch().s_params(np.array([1.55]))
    assert np.abs(s[0, 1, 0]) ** 2 == pytest.approx(0.5)
    assert np.abs(s[0, 2, 0]) ** 2 == pytest.approx(0.5)
    assert_reciprocal(s)
    # each excitation conserves power (<= 1); combining direction radiates
    col_power = np.sum(np.abs(s[0]) ** 2, axis=0)
    assert np.all(col_power <= 1.0 + 1e-12)
    assert col_power[1] == pytest.approx(0.5)  # single-arm input loses half


# --- Phase shifters ---------------------------------------------------------


def test_phase_shifter_and_thermal_pi():
    wl = np.array([1.55])
    s_pi = PhaseShifter(phase_rad=np.pi).s_params(wl)
    assert s_pi[0, 1, 0] == pytest.approx(-1.0, abs=1e-12)
    th = ThermalPhaseShifter(power_mw=20.0, p_pi_mw=20.0)
    assert th.phase_rad == pytest.approx(np.pi)
    assert np.allclose(th.s_params(wl), s_pi)
    half = ThermalPhaseShifter(power_mw=10.0, p_pi_mw=20.0).s_params(wl)
    assert half[0, 1, 0] == pytest.approx(np.exp(-1j * np.pi / 2), abs=1e-12)
    assert_unitary(s_pi)


# --- GratingCoupler ---------------------------------------------------------


def test_grating_coupler_peak_and_1db_bandwidth():
    gc = GratingCoupler(peak_il_db=4.0, bw_1db_nm=35.0, center_um=1.55)
    half_bw = 0.5 * 35.0e-3
    wl = np.array([1.55, 1.55 - half_bw, 1.55 + half_bw])
    s = gc.s_params(wl)
    il = -10.0 * np.log10(np.abs(s[:, 1, 0]) ** 2)
    assert il[0] == pytest.approx(4.0, abs=1e-12)
    assert il[1] == pytest.approx(5.0, abs=1e-10)  # exactly 1 dB down
    assert il[2] == pytest.approx(5.0, abs=1e-10)
    assert_reciprocal(s)


# --- Rings ------------------------------------------------------------------


def test_allpass_resonance_and_critical_coupling():
    L = 100.0
    kappa = 0.2
    t = np.sqrt(1.0 - kappa)
    # choose loss so round-trip amplitude a equals t (critical coupling)
    loss_db_rt = -20.0 * np.log10(t)
    ring = RingAllPass(L, NEFF0, NG, kappa_power=kappa,
                       loss_db_per_cm=loss_db_rt / (L * 1e-4))
    m = round(L * NEFF0 / WL0)  # resonance order near wl0
    wl_res = resonance_wl(m, L)
    s = ring.s_params(np.array([wl_res]))
    assert np.abs(s[0, 1, 0]) < 1e-12  # full extinction at critical coupling
    # off resonance (half FSR away) transmission is high
    s_off = ring.s_params(np.array([resonance_wl(m + 0.5, L)]))
    assert np.abs(s_off[0, 1, 0]) ** 2 > 0.9


def test_allpass_lossless_unitary():
    ring = RingAllPass(60.0, NEFF0, NG, kappa_power=0.3, loss_db_per_cm=0.0)
    s = ring.s_params(WL)
    assert_unitary(s)
    assert_reciprocal(s)


def test_adddrop_resonance_position_and_drop_peak():
    L = 120.0
    ring = RingAddDrop(L, NEFF0, NG, kappa1_power=0.1, kappa2_power=0.1,
                       loss_db_per_cm=1.0)
    m = round(L * NEFF0 / WL0)
    wl_res = resonance_wl(m, L)
    wl = np.linspace(wl_res - 2e-3, wl_res + 2e-3, 4001)
    drop = np.abs(ring.s_params(wl)[:, 3, 0]) ** 2
    # drop maximum sits at the closed-form resonance within the grid step
    assert abs(wl[np.argmax(drop)] - wl_res) <= wl[1] - wl[0]
    # at resonance: drop = k1 k2 sqrt(a) / (1 - t1 t2 a)^2, through minimal
    a = 10 ** (-1.0 * L * 1e-4 / 20)
    t = np.sqrt(0.9)
    expected = (0.1 * 0.1 * a) / (1 - t * t * a) ** 2
    s_res = ring.s_params(np.array([wl_res]))
    assert np.abs(s_res[0, 3, 0]) ** 2 == pytest.approx(expected, rel=1e-9)
    thru = np.abs(ring.s_params(wl)[:, 1, 0]) ** 2
    assert abs(wl[np.argmin(thru)] - wl_res) <= wl[1] - wl[0]


def test_adddrop_lossless_energy_conservation():
    ring = RingAddDrop(80.0, NEFF0, NG, kappa1_power=0.15, kappa2_power=0.25,
                       loss_db_per_cm=0.0)
    s = ring.s_params(WL)
    assert_unitary(s, atol=1e-10)
    assert_reciprocal(s)
    # counter-propagating ports carry no direct coupling
    assert np.allclose(s[:, 2, 0], 0.0) and np.allclose(s[:, 3, 1], 0.0)


def test_adddrop_symmetry_add_to_through():
    ring = RingAddDrop(80.0, NEFF0, NG, kappa1_power=0.1, kappa2_power=0.3,
                       loss_db_per_cm=2.0)
    s = ring.s_params(WL)
    assert np.allclose(s[:, 3, 0], s[:, 1, 2])  # in->drop == add->through


def test_fsr_formula_matches_drop_peak_spacing():
    L = 100.0
    ring = RingAddDrop(L, NEFF0, NG, 0.1, 0.1, loss_db_per_cm=2.0)
    wl = np.linspace(1.53, 1.57, 200001)
    drop = np.abs(ring.s_params(wl)[:, 3, 0]) ** 2
    peaks = np.flatnonzero((drop[1:-1] > drop[:-2]) & (drop[1:-1] > drop[2:])) + 1
    wl_peaks = wl[peaks]
    assert len(wl_peaks) >= 3
    spacings = np.diff(wl_peaks)
    mid = wl_peaks[:-1] + spacings / 2
    assert np.allclose(spacings, ring_fsr_um(mid, NG, L), rtol=1e-2)


# --- MZI --------------------------------------------------------------------


def test_mzi_lossless_unitary():
    assert_unitary(MZI(100.0, NEFF0, NG).s_params(WL))
    assert_unitary(MZI(100.0, NEFF0, NG, coupling_in=0.3, coupling_out=0.7).s_params(WL))
    assert_reciprocal(MZI(100.0, NEFF0, NG, coupling_in=0.3).s_params(WL))


def test_mzi_balanced_full_cross():
    s = MZI(0.0, NEFF0, NG).s_params(WL)
    assert np.allclose(np.abs(s[:, 2, 0]), 0.0, atol=1e-12)  # bar dark
    assert np.allclose(np.abs(s[:, 3, 0]), 1.0)  # all power crosses


def test_mzi_extinction_wavelengths_and_fsr():
    dl = 100.0
    mzi = MZI(dl, NEFF0, NG)
    # bar port extinct where the differential phase is 2*pi*m
    m = round(dl * NEFF0 / WL0)
    wl_m = resonance_wl(m, dl)
    s = mzi.s_params(np.array([wl_m]))
    assert np.abs(s[0, 2, 0]) < 1e-12  # in0 -> out0 dark
    assert np.abs(s[0, 3, 0]) == pytest.approx(1.0)  # in0 -> out1 full
    # anti-resonance: bar port full
    wl_half = resonance_wl(m + 0.5, dl)
    s_half = mzi.s_params(np.array([wl_half]))
    assert np.abs(s_half[0, 2, 0]) == pytest.approx(1.0)
    # spacing of adjacent nulls matches wl^2/(ng*dl)
    wl_m1 = resonance_wl(m + 1, dl)
    fsr = wl_m - wl_m1
    assert fsr == pytest.approx(ring_fsr_um(0.5 * (wl_m + wl_m1), NG, dl), rel=1e-2)


# --- helpers ----------------------------------------------------------------


def test_helper_formulas():
    assert ring_fsr_um(1.55, 4.2, 100.0) == pytest.approx(1.55**2 / 420.0)
    assert loaded_q(1.55, 1.55e-4) == pytest.approx(1e4)
    assert finesse(5.7e-3, 1.9e-4) == pytest.approx(30.0)


def test_parameter_validation():
    with pytest.raises(ValueError):
        DirectionalCoupler(coupling=1.5)
    with pytest.raises(ValueError):
        RingAllPass(100.0, NEFF0, NG, kappa_power=-0.1)
    with pytest.raises(ValueError):
        ThermalPhaseShifter(power_mw=1.0, p_pi_mw=0.0)
    with pytest.raises(ValueError):
        Straight(100.0, NEFF0, NG).s_params(np.array([-1.55]))
