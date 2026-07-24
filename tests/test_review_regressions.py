"""Regression tests for defects found in the 2026-07-23 adversarial review.

Each test pins the corrected behavior with an independent formula or a
composition cross-check, so reintroducing the original bug fails loudly.
Findings and refuter transcripts: docs/research/raw/workflow-adversarial-review/.
"""

import math

import numpy as np
import pytest

from siphon import components, link, materials, montecarlo as mc, waveguide, wdm
from siphon.circuit import Circuit


# --- finding: ring phase conjugation broke composition ---------------------


def _composed_all_pass(circumference_um, neff0, ng, kappa, loss_db_per_cm, wl):
    """The same physical ring built from a coupler + waveguide loop."""
    c = Circuit()
    c.add("dc", components.DirectionalCoupler(coupling=kappa))
    c.add(
        "loop",
        components.Straight(
            length_um=circumference_um,
            neff0=neff0,
            ng=ng,
            loss_db_per_cm=loss_db_per_cm,
        ),
    )
    c.connect(("dc", "out1"), ("loop", "in"))
    c.connect(("loop", "out"), ("dc", "in1"))
    c.expose("in", ("dc", "in0"))
    c.expose("out", ("dc", "out0"))
    return c.transmission(wl, "in", "out")


def test_ring_all_pass_matches_composed_ring_in_full_complex_value():
    wl = np.linspace(1.549, 1.551, 401)
    args = dict(circumference_um=60.0, neff0=2.4, ng=4.2)
    ring = components.RingAllPass(**args, kappa_power=0.2, loss_db_per_cm=0.1)
    analytic = ring.s_params(wl)[:, 1, 0]
    composed = _composed_all_pass(**args, kappa=0.2, loss_db_per_cm=0.1, wl=wl)
    # phase too, not just magnitude — a conjugated ring passes |.| checks
    np.testing.assert_allclose(analytic, composed, atol=1e-10)


def test_ring_group_delay_is_positive_at_resonance():
    # a passive all-pass ring delays light; group delay -d(arg S21)/d(omega)
    # must be positive (and resonance-enhanced), never an acausal advance
    wl = np.linspace(1.5495, 1.5505, 20001)
    ring = components.RingAllPass(
        circumference_um=60.0, neff0=2.4, ng=4.2, kappa_power=0.2, loss_db_per_cm=0.1
    )
    s21 = ring.s_params(wl)[:, 1, 0]
    omega = 2.0 * np.pi * 2.99792458e14 / wl  # rad/s, wl in um
    tau = -np.gradient(np.unwrap(np.angle(s21)), omega)  # seconds
    assert tau.min() > 0.0
    bare_round_trip_s = 4.2 * 60.0e-6 / 2.99792458e8
    assert tau.max() > 2.0 * bare_round_trip_s  # resonance enhancement


def test_add_drop_drop_path_matches_composed_half_ring_phase():
    wl = np.linspace(1.5495, 1.5505, 501)
    ring = components.RingAddDrop(
        circumference_um=60.0, neff0=2.4, ng=4.2, kappa1_power=0.2, kappa2_power=0.2
    )
    c = Circuit()
    c.add("dc1", components.DirectionalCoupler(coupling=0.2))
    c.add("dc2", components.DirectionalCoupler(coupling=0.2))
    half = dict(length_um=30.0, neff0=2.4, ng=4.2)
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
    np.testing.assert_allclose(
        ring.s_params(wl)[:, 3, 0], c.transmission(wl, "in", "drop"), atol=1e-10
    )
    np.testing.assert_allclose(
        ring.s_params(wl)[:, 1, 0], c.transmission(wl, "in", "through"), atol=1e-10
    )


# --- finding: ER double-counted in the OMA budget ---------------------------


def test_launched_oma_closed_form_at_finite_er():
    # OMA = P1 - P0 with P1/P0 = ER and (P1+P0)/2 = P_avg
    # => OMA = 2 P_avg (ER-1)/(ER+1); inverting the ratio must fail this
    lb = link.preset_pluggable_dr4()
    er = 10.0 ** (lb.modulator.extinction_ratio_db / 10.0)
    p_avg_mw = 10.0 ** (lb.average_launch_power_dbm / 10.0)
    oma_mw = 2.0 * p_avg_mw * (er - 1.0) / (er + 1.0)
    assert lb.launched_oma_dbm == pytest.approx(10.0 * math.log10(oma_mw), abs=1e-9)


def test_required_oma_composition_charges_er_exactly_once():
    lb = link.preset_pluggable_dr4()
    thermal = link.receiver_sensitivity_oma_dbm(lb.photodiode, lb.tia, lb.signaling)
    expected = thermal + lb.rin_penalty_db + sum((lb.penalties_db or {}).values())
    assert lb.sensitivity_oma_dbm == pytest.approx(expected, abs=1e-12)
    # thermal-limited OMA requirement is ER-independent; only launch changes
    hi_er = link.LinkBudget(
        laser=lb.laser,
        modulator=link.Modulator(
            insertion_loss_db=lb.modulator.insertion_loss_db,
            extinction_ratio_db=100.0,
            energy_fj_per_bit=lb.modulator.energy_fj_per_bit,
        ),
        path=lb.path,
        photodiode=lb.photodiode,
        tia=lb.tia,
        signaling=lb.signaling,
        penalties_db=lb.penalties_db,
    )
    assert hi_er.sensitivity_oma_dbm == pytest.approx(lb.sensitivity_oma_dbm, abs=1e-12)
    assert hi_er.launched_oma_dbm > lb.launched_oma_dbm


def test_margin_from_first_principles():
    # hand-built link, every term recomputed independently in the test
    lb = link.LinkBudget(
        laser=link.Laser(power_dbm=10.0, wpe=0.2, rin_db_hz=-math.inf),
        modulator=link.Modulator(
            insertion_loss_db=3.0, extinction_ratio_db=6.0, energy_fj_per_bit=100.0
        ),
        path=[link.LossElement("loss", 4.0)],
        photodiode=link.Photodiode(responsivity_a_per_w=1.0),
        tia=link.Tia(input_noise_pa_per_sqrt_hz=10.0, bandwidth_ghz=30.0, energy_fj_per_bit=100.0),
        signaling=link.Signaling(rate_gbd=50.0, levels=2, target_ber=1e-12),
    )
    q = math.sqrt(2.0) * float(__import__("scipy.special", fromlist=["erfcinv"]).erfcinv(2e-12))
    er = 10.0 ** 0.6
    p_avg_mw = 10.0 ** ((10.0 - 3.0) / 10.0) * (er + 1.0) / (2.0 * er)  # CW -> avg
    oma_launch_mw = 2.0 * p_avg_mw * (er - 1.0) / (er + 1.0)
    i_n_a = 10.0e-12 * math.sqrt(0.75 * 50.0e9)
    oma_req_mw = 2.0 * q * 1.0 * i_n_a / 1.0 * 1e3  # (M-1)=1, R=1 A/W
    expected_margin = (
        10.0 * math.log10(oma_launch_mw) - 4.0 - 10.0 * math.log10(oma_req_mw)
    )
    assert lb.margin_db == pytest.approx(expected_margin, abs=1e-9)


# --- finding: laser sharing violated energy conservation ---------------------


def test_laser_energy_scales_with_lanes_not_devices():
    lb = link.preset_cpo_optical_io()
    e1 = link.energy_per_bit_pj(lb, n_lanes=1)
    e8 = link.energy_per_bit_pj(lb, n_lanes=8, laser_shared_by=8)
    assert e8["laser"] == pytest.approx(e1["laser"], rel=1e-12)
    line_mw = 10.0 ** (lb.laser.power_dbm / 10.0)
    expected = (8 * line_mw / lb.laser.wpe) / (8 * lb.signaling.bit_rate_gbps)
    assert e8["laser"] == pytest.approx(expected, rel=1e-12)


# --- finding: montecarlo sensitivity() zeroed out under failed corners -------


def test_sensitivity_survives_failed_corners():
    def metric(p):
        if p["x"] > 1.5:
            raise ValueError("corner fails")
        return 2.0 * p["x"] + 0.01 * p["noise"]

    res = mc.run(
        metric,
        [mc.Normal("x", 0.0, 1.0), mc.Normal("noise", 0.0, 1.0)],
        n=4000,
        seed=4,
    )
    assert res.n_failed > 0
    s = res.sensitivity()
    assert abs(s["x"]) > 0.95  # not silently zero
    assert abs(s["noise"]) < 0.2


# --- finding: chromatic dispersion only smoke-tested --------------------------


def test_dispersion_consistent_with_group_index_derivative():
    # independent formula: D = (1/c) * d(n_g)/d(lambda)
    wg = waveguide.Waveguide(width_um=0.5, height_um=0.22)
    wl, dwl = 1.55, 5e-4
    dng_dwl = (wg.group_index(wl + dwl) - wg.group_index(wl - dwl)) / (2.0 * dwl)  # 1/um
    c_um_per_ps = 2.99792458e2
    expected = dng_dwl / c_um_per_ps * 1e6  # ps/um^2 -> ps/(nm*km)
    got = wg.dispersion_ps_nm_km(wl)
    assert got == pytest.approx(expected, rel=0.02)
    assert got < -300.0  # sign and magnitude pinned for this EIM geometry


# --- smaller confirmed findings ----------------------------------------------


def test_n_si_li1980_resonance_term():
    # B multiplies lambda_1^2, not lambda^2 (Li 1980): n(1.55) = 3.4764
    assert materials.n_si(1.55) == pytest.approx(3.4764, abs=2e-4)
    # pin the exact defining expression at a second wavelength so the
    # resonance term cannot silently be re-transcribed as B*wl^2/(wl^2-l1^2)
    for wl in (1.3, 3.0):
        wl2, l1sq = wl**2, 1.1071**2
        n2 = 11.6858 + 0.939816 / wl2 + 8.10461e-3 * l1sq / (wl2 - l1sq)
        assert materials.n_si(wl) == pytest.approx(math.sqrt(n2), abs=1e-12)
        wrong = math.sqrt(11.6858 + 0.939816 / wl2 + 8.10461e-3 * wl2 / (wl2 - l1sq))
        assert abs(materials.n_si(wl) - wrong) > 5e-4


def test_waveguide_rejects_bogus_mode_string():
    wg = waveguide.Waveguide(width_um=0.5, height_um=0.22)
    with pytest.raises(ValueError, match="mode"):
        wg.neff(1.55, mode="TEM")


def test_expected_tuning_power_heat_only_vs_bidirectional():
    assert wdm.expected_tuning_power_mw(16.0, 0.25) == pytest.approx(32.0)
    assert wdm.expected_tuning_power_mw(16.0, 0.25, bidirectional=True) == pytest.approx(16.0)
