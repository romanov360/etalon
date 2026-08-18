"""Tests for etalon.link: BER/Q math, penalties, budgets, energy, presets."""

import math

import pytest

from etalon.link import (
    LinkBudget,
    Laser,
    LossElement,
    Modulator,
    Photodiode,
    Signaling,
    Tia,
    aggregate_pdl_db,
    ber_from_q,
    crosstalk_penalty_db,
    energy_per_bit_pj,
    er_penalty_db,
    pdl_penalty_db,
    preset_cpo_optical_io,
    preset_pluggable_dr4,
    q_from_ber,
    receiver_sensitivity_oma_dbm,
    rin_penalty_db,
)


def _simple_link(**overrides) -> LinkBudget:
    kwargs = dict(
        laser=Laser(power_dbm=10.0, wpe=0.15, rin_db_hz=-145.0),
        modulator=Modulator(
            insertion_loss_db=4.0, extinction_ratio_db=5.0, energy_fj_per_bit=200.0
        ),
        path=[LossElement("coupler", 1.5), LossElement("fiber", 0.5)],
        photodiode=Photodiode(responsivity_a_per_w=0.9),
        tia=Tia(input_noise_pa_per_sqrt_hz=15.0, bandwidth_ghz=35.0, energy_fj_per_bit=500.0),
        signaling=Signaling(rate_gbd=53.125, levels=4, target_ber=2.4e-4),
        penalties_db={"mpi": 0.5},
    )
    kwargs.update(overrides)
    return LinkBudget(**kwargs)


# --- Q <-> BER --------------------------------------------------------------


def test_q_pam4_kp4_threshold():
    # KP4 pre-FEC threshold 2.4e-4 on PAM4 needs q ~ 3.4-3.6
    q = q_from_ber(2.4e-4, levels=4)
    assert 3.4 < q < 3.6


def test_q_nrz_1e12():
    # Textbook value: BER 1e-12 on NRZ <=> Q ~ 7.03 (Agrawal table 4.x)
    q = q_from_ber(1e-12, levels=2)
    assert q == pytest.approx(7.034, abs=0.01)


@pytest.mark.parametrize("levels", [2, 4, 8])
@pytest.mark.parametrize("ber", [2.4e-4, 1e-6, 1e-12])
def test_q_ber_round_trip(levels, ber):
    q = q_from_ber(ber, levels)
    assert ber_from_q(q, levels) == pytest.approx(ber, rel=1e-9)


def test_q_from_ber_rejects_bad_input():
    with pytest.raises(ValueError):
        q_from_ber(0.6, 2)
    with pytest.raises(ValueError):
        q_from_ber(1e-4, 3)  # not a power of two


# --- penalties ---------------------------------------------------------------


def test_er_penalty_3db():
    # 10*log10((10^0.3 + 1)/(10^0.3 - 1)) = 4.78 dB
    assert er_penalty_db(3.0) == pytest.approx(4.77, abs=0.05)


def test_er_penalty_vanishes_at_infinite_er():
    assert er_penalty_db(math.inf) == 0.0
    assert er_penalty_db(80.0) < 1e-3


def test_modulation_loss():
    # infinite ER: average of full-swing PAM is half the peak -> 3.01 dB
    m = Modulator(insertion_loss_db=0.0, extinction_ratio_db=math.inf, energy_fj_per_bit=0.0)
    assert m.modulation_loss_db == pytest.approx(10 * math.log10(2), abs=1e-12)
    # ER = 3 dB: -10 log10((1 + 1/1.995)/2) = 1.246 dB
    m = Modulator(insertion_loss_db=0.0, extinction_ratio_db=3.0, energy_fj_per_bit=0.0)
    er = 10 ** 0.3
    assert m.modulation_loss_db == pytest.approx(-10 * math.log10((1 + 1 / er) / 2), abs=1e-12)


def test_rin_penalty_small_for_good_laser():
    q = q_from_ber(2.4e-4, 4)
    pen = rin_penalty_db(-145.0, 53.125, q, 4)
    assert 0.0 < pen < 0.5


def test_rin_penalty_unreachable_raises():
    q = q_from_ber(2.4e-4, 4)
    with pytest.raises(ValueError):
        rin_penalty_db(-110.0, 106.25, q, 4)


def test_crosstalk_penalty():
    # -20 dB aggregate crosstalk: -10 log10(1 - 0.01) = 0.0436 dB
    assert crosstalk_penalty_db(-20.0) == pytest.approx(0.0436, abs=1e-3)
    with pytest.raises(ValueError):
        crosstalk_penalty_db(0.0)


# --- polarization-dependent loss (PDL) ------------------------------------------


def test_pdl_penalty_is_identity():
    # By definition PDL_dB = 10log10(Pmax/Pmin): the worst-case penalty
    # IS the PDL value.
    assert pdl_penalty_db(0.0) == 0.0
    assert pdl_penalty_db(0.8) == pytest.approx(0.8)
    assert pdl_penalty_db(2.3) == pytest.approx(2.3)


def test_pdl_penalty_rejects_negative():
    with pytest.raises(ValueError):
        pdl_penalty_db(-0.1)


def test_aggregate_pdl_is_linear_sum():
    assert aggregate_pdl_db([0.5, 0.8, 0.3]) == pytest.approx(1.6)
    assert aggregate_pdl_db([1.2]) == pytest.approx(1.2)
    assert aggregate_pdl_db([0.0, 0.0]) == pytest.approx(0.0)


def test_aggregate_pdl_matches_penalty_composition():
    # Aggregating then penalizing must equal summing the individual
    # penalties -- both are the same linear-dB-sum rule, just composed
    # in a different order.
    components = [0.4, 0.6, 1.1]
    via_aggregate = pdl_penalty_db(aggregate_pdl_db(components))
    via_sum = sum(pdl_penalty_db(v) for v in components)
    assert via_aggregate == pytest.approx(via_sum)


def test_aggregate_pdl_rejects_empty_or_negative():
    with pytest.raises(ValueError):
        aggregate_pdl_db([])
    with pytest.raises(ValueError):
        aggregate_pdl_db([0.5, -0.2])


def test_pdl_booked_in_link_budget_penalties():
    base = preset_cpo_optical_io()
    grating_coupler_pdl_db = [0.4, 0.4]  # two fiber-chip interfaces
    system_pdl = aggregate_pdl_db(grating_coupler_pdl_db)
    with_pdl = LinkBudget(
        laser=base.laser,
        modulator=base.modulator,
        path=base.path,
        photodiode=base.photodiode,
        tia=base.tia,
        signaling=base.signaling,
        penalties_db={**(base.penalties_db or {}), "pdl": pdl_penalty_db(system_pdl)},
    )
    assert with_pdl.margin_db == pytest.approx(base.margin_db - system_pdl, abs=1e-9)


# --- receiver sensitivity ------------------------------------------------------


def test_sensitivity_closed_form():
    # Hand calculation: OMA = 2 q (M-1) i_n / R, i_n = S_i sqrt(0.75 * baud)
    pd = Photodiode(responsivity_a_per_w=0.8)
    tia = Tia(input_noise_pa_per_sqrt_hz=16.0, bandwidth_ghz=40.0, energy_fj_per_bit=0.0)
    sig = Signaling(rate_gbd=53.125, levels=4, target_ber=2.4e-4)
    i_n = 16e-12 * math.sqrt(0.75 * 53.125e9)
    oma_w = 2 * sig.q_factor * 3 * i_n / 0.8
    expected_dbm = 10 * math.log10(oma_w * 1e3)
    assert receiver_sensitivity_oma_dbm(pd, tia, sig) == pytest.approx(expected_dbm, abs=1e-9)


def test_sensitivity_worsens_with_rate():
    # i_n ~ sqrt(rate): 4x the baud costs 5*log10(4) = 3.01 dB of OMA
    pd = Photodiode(responsivity_a_per_w=1.0)
    tia = Tia(input_noise_pa_per_sqrt_hz=10.0, bandwidth_ghz=50.0, energy_fj_per_bit=0.0)
    s1 = receiver_sensitivity_oma_dbm(pd, tia, Signaling(25.0, 2, 1e-12))
    s2 = receiver_sensitivity_oma_dbm(pd, tia, Signaling(100.0, 2, 1e-12))
    assert s2 > s1
    assert s2 - s1 == pytest.approx(5 * math.log10(4), abs=1e-9)


def test_pam4_vs_nrz_penalty():
    # Same baud and same per-eye q: PAM4 needs (M-1) = 3x the OMA of NRZ.
    # That is 20*log10(3) = 9.54 dB electrical, i.e. 10*log10(3) = 4.77 dB
    # optical (OMA in dBm).
    pd = Photodiode(responsivity_a_per_w=1.0)
    tia = Tia(input_noise_pa_per_sqrt_hz=15.0, bandwidth_ghz=40.0, energy_fj_per_bit=0.0)
    sig4 = Signaling(rate_gbd=53.125, levels=4, target_ber=2.4e-4)
    ber_nrz_same_q = ber_from_q(sig4.q_factor, 2)  # NRZ target with identical q
    sig2 = Signaling(rate_gbd=53.125, levels=2, target_ber=ber_nrz_same_q)
    diff_optical_db = receiver_sensitivity_oma_dbm(pd, tia, sig4) - receiver_sensitivity_oma_dbm(
        pd, tia, sig2
    )
    assert 2 * diff_optical_db == pytest.approx(20 * math.log10(3), abs=1e-9)
    # At equal target BER (q shifts slightly) the gap stays close to that.
    sig2_same_ber = Signaling(rate_gbd=53.125, levels=2, target_ber=2.4e-4)
    diff = receiver_sensitivity_oma_dbm(pd, tia, sig4) - receiver_sensitivity_oma_dbm(
        pd, tia, sig2_same_ber
    )
    assert diff == pytest.approx(10 * math.log10(3), abs=0.2)


# --- link budget ------------------------------------------------------------------


def test_launched_oma_infinite_er_equals_peak():
    # At infinite ER, modulation loss (-3.01) and avg->OMA (+3.01) cancel:
    # OMA equals the peak power, laser - IL.
    link = _simple_link(
        modulator=Modulator(
            insertion_loss_db=4.0, extinction_ratio_db=math.inf, energy_fj_per_bit=0.0
        )
    )
    assert link.launched_oma_dbm == pytest.approx(10.0 - 4.0, abs=1e-12)


def test_path_loss_one_db_in_one_db_out():
    link = _simple_link()
    m0 = link.margin_db
    link_more = _simple_link(path=link.path + [LossElement("extra", 1.0)])
    assert link_more.margin_db == pytest.approx(m0 - 1.0, abs=1e-12)
    assert link_more.received_oma_dbm == pytest.approx(link.received_oma_dbm - 1.0, abs=1e-12)


def test_explicit_penalty_one_db_in_one_db_out():
    link = _simple_link()
    link_pen = _simple_link(penalties_db={"mpi": 0.5, "aging": 1.0})
    assert link_pen.margin_db == pytest.approx(link.margin_db - 1.0, abs=1e-12)


# --- energy -----------------------------------------------------------------------


def test_energy_unit_identity():
    # 1 mW laser at wpe=1 on a 1 Gb/s lane is exactly 1 pJ/bit.
    link = _simple_link(
        laser=Laser(power_dbm=0.0, wpe=1.0, rin_db_hz=-150.0),
        signaling=Signaling(rate_gbd=1.0, levels=2, target_ber=1e-12),
    )
    e = energy_per_bit_pj(link, n_lanes=1, serdes_pj_per_bit=0.0)
    assert e["laser"] == pytest.approx(1.0, abs=1e-12)


def test_energy_breakdown_sums_and_sharing():
    link = preset_pluggable_dr4()
    e = energy_per_bit_pj(
        link, n_lanes=4, laser_shared_by=1, thermal_tuning_mw_per_lane=2.0,
        serdes_pj_per_bit=4.0,
    )
    parts = ("laser", "modulator", "tia", "tuning", "serdes")
    assert e["total"] == pytest.approx(sum(e[k] for k in parts), rel=1e-12)
    assert all(e[k] >= 0.0 for k in parts)
    # sharing a source does NOT reduce per-line optical power (energy
    # conservation: N lanes still need N lines of full power); it only
    # amortizes the per-device overhead term
    e_shared = energy_per_bit_pj(link, n_lanes=4, laser_shared_by=4)
    assert e_shared["laser"] == pytest.approx(e["laser"], rel=1e-12)
    e_ovh1 = energy_per_bit_pj(
        link, n_lanes=4, laser_shared_by=1, laser_overhead_mw_per_device=10.0
    )
    e_ovh4 = energy_per_bit_pj(
        link, n_lanes=4, laser_shared_by=4, laser_overhead_mw_per_device=10.0
    )
    agg_gbps = 4 * link.signaling.bit_rate_gbps
    assert e_ovh1["laser"] - e_shared["laser"] == pytest.approx(40.0 / agg_gbps, rel=1e-9)
    assert e_ovh4["laser"] - e_shared["laser"] == pytest.approx(10.0 / agg_gbps, rel=1e-9)
    # closed form for the laser term: mW_electrical / aggregate Gb/s
    laser_mw = 4 * 10 ** (link.laser.power_dbm / 10) / link.laser.wpe
    assert e["laser"] == pytest.approx(
        laser_mw / (4 * link.signaling.bit_rate_gbps), rel=1e-12
    )


# --- presets ------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [preset_pluggable_dr4, preset_cpo_optical_io])
def test_presets_close_with_positive_margin(factory):
    link = factory()
    assert math.isfinite(link.margin_db)
    assert link.margin_db > 0.0
    assert math.isfinite(link.launched_oma_dbm)
    assert link.received_oma_dbm < link.launched_oma_dbm


@pytest.mark.parametrize("factory", [preset_pluggable_dr4, preset_cpo_optical_io])
def test_preset_report_contains_everything(factory):
    link = factory()
    text = link.report()
    for elem in link.path:
        assert elem.name in text
    for name in link.penalties_db:
        assert name in text
    for token in ("launched OMA", "received OMA", "RIN penalty", "MARGIN"):
        assert token in text
    # ER appears once, at the transmitter conversion — never as an added
    # receiver-side "ER penalty" row (that would double-count ER, see
    # LinkBudget.sensitivity_oma_dbm)
    assert "avg -> outer OMA" in text
    assert "ER penalty" not in text


def test_report_margin_matches_property():
    link = _simple_link()
    text = link.report()
    margin_line = next(line for line in text.splitlines() if "MARGIN" in line)
    assert f"{link.margin_db:+8.2f} dB" in margin_line
