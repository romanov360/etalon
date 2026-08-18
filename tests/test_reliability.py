"""Tests for etalon.reliability: Arrhenius, FIT/MTTF algebra, lognormal
wear-out, competing-risks module survival, and WPE thermal derating.

Anchors are analytic or independently re-derived (hand-computed Arrhenius
exponent, normal-CDF identities), never the module's own output.
"""

import math

import pytest

from etalon.reliability import (
    BOLTZMANN_EV_PER_K,
    arrhenius_acceleration,
    fit_from_mttf,
    module_fit,
    module_mttf_hours,
    module_survival,
    mttf_from_fit,
    survival_probability,
    wearout_fraction,
    wpe_derated,
)

# --- Arrhenius ---------------------------------------------------------------


def test_arrhenius_anchor_52p8():
    # Ea = 0.97 eV, 25 C -> 60 C: hand-derived
    # exponent = (0.97 / 8.617333262e-5) * (1/298.15 - 1/333.15) = 3.9665...
    # AF = exp(3.9665) = 52.8 (classic quotable laser-reliability number).
    af = arrhenius_acceleration(25.0, 60.0, 0.97)
    assert af == pytest.approx(52.8, rel=0.01)


def test_arrhenius_independent_rederivation():
    # Recompute from the definition with different intermediate grouping.
    t_use_k, t_stress_k = 40.0 + 273.15, 85.0 + 273.15
    ea = 0.7
    expected = math.exp(ea / BOLTZMANN_EV_PER_K / t_use_k) / math.exp(
        ea / BOLTZMANN_EV_PER_K / t_stress_k
    )
    assert arrhenius_acceleration(40.0, 85.0, ea) == pytest.approx(expected, rel=1e-12)


def test_arrhenius_identity_at_equal_temperature():
    assert arrhenius_acceleration(55.0, 55.0, 0.97) == pytest.approx(1.0)


def test_arrhenius_monotonic_in_ea():
    afs = [arrhenius_acceleration(25.0, 85.0, ea) for ea in (0.3, 0.5, 0.7, 0.97)]
    assert all(a2 > a1 for a1, a2 in zip(afs, afs[1:]))
    assert all(a > 1.0 for a in afs)


def test_arrhenius_monotonic_in_stress_temperature():
    afs = [arrhenius_acceleration(25.0, t, 0.7) for t in (40.0, 60.0, 85.0, 100.0)]
    assert all(a2 > a1 for a1, a2 in zip(afs, afs[1:]))


def test_arrhenius_below_use_temperature_decelerates():
    assert arrhenius_acceleration(60.0, 25.0, 0.97) < 1.0


def test_arrhenius_rejects_bad_input():
    with pytest.raises(ValueError):
        arrhenius_acceleration(25.0, 85.0, 0.0)
    with pytest.raises(ValueError):
        arrhenius_acceleration(-300.0, 85.0, 0.7)
    with pytest.raises(ValueError):
        arrhenius_acceleration(25.0, -280.0, 0.7)


# --- FIT <-> MTTF -------------------------------------------------------------


def test_fit_mttf_definition():
    # 1e9 hours MTTF is exactly 1 FIT, by definition of the unit.
    assert fit_from_mttf(1e9) == pytest.approx(1.0)
    assert mttf_from_fit(1.0) == pytest.approx(1e9)
    # 100 FIT <-> 1e7 hours.
    assert fit_from_mttf(1e7) == pytest.approx(100.0)


@pytest.mark.parametrize("mttf", [1e5, 3.7e6, 1e9])
def test_fit_mttf_round_trip(mttf):
    assert mttf_from_fit(fit_from_mttf(mttf)) == pytest.approx(mttf, rel=1e-12)


def test_fit_mttf_reject_nonpositive():
    with pytest.raises(ValueError):
        fit_from_mttf(0.0)
    with pytest.raises(ValueError):
        mttf_from_fit(-5.0)


def test_module_fit_scales_linearly():
    assert module_fit(50.0, 1) == pytest.approx(50.0)
    assert module_fit(50.0, 64) == pytest.approx(64 * 50.0)
    # Additivity of hazards: n devices at f FIT == 1 device at n*f FIT.
    assert module_fit(50.0, 8) == pytest.approx(module_fit(400.0, 1))


def test_module_mttf_is_device_mttf_over_n():
    fit = 25.0
    device_mttf = mttf_from_fit(fit)
    for n in (1, 4, 64):
        assert module_mttf_hours(fit, n) == pytest.approx(device_mttf / n, rel=1e-12)


def test_module_fit_rejects_bad_input():
    with pytest.raises(ValueError):
        module_fit(50.0, 0)
    with pytest.raises(ValueError):
        module_fit(-1.0, 4)
    with pytest.raises(ValueError):
        module_mttf_hours(0.0, 4)  # zero FIT -> infinite MTTF, rejected


# --- survival (random regime) --------------------------------------------------


def test_survival_probability_analytic():
    # S(t) = exp(-fit * t * 1e-9): 1000 FIT for 1e6 h -> exp(-1).
    assert survival_probability(1e6, 1000.0) == pytest.approx(math.exp(-1.0), rel=1e-12)
    assert survival_probability(0.0, 1000.0) == 1.0
    assert survival_probability(1e6, 0.0) == 1.0


def test_survival_at_mttf_is_1_over_e():
    fit = 200.0
    assert survival_probability(mttf_from_fit(fit), fit) == pytest.approx(
        math.exp(-1.0), rel=1e-12
    )


def test_survival_rejects_negative():
    with pytest.raises(ValueError):
        survival_probability(-1.0, 100.0)
    with pytest.raises(ValueError):
        survival_probability(1.0, -100.0)


# --- lognormal wear-out ---------------------------------------------------------


def test_wearout_half_at_median_exactly():
    assert wearout_fraction(2e5, 2e5, 0.5) == 0.5


def test_wearout_symmetry_in_log_time():
    # Phi(-z) = 1 - Phi(z): F(t50/r) + F(t50*r) = 1 for any r > 0.
    t50, sigma, r = 1e5, 0.7, 3.0
    assert wearout_fraction(t50 / r, t50, sigma) + wearout_fraction(
        t50 * r, t50, sigma
    ) == pytest.approx(1.0, abs=1e-12)


def test_wearout_one_sigma_point():
    # At t = t50 * exp(sigma), z = 1: F = Phi(1) = 0.841345 (standard table).
    t50, sigma = 1e5, 0.6
    f = wearout_fraction(t50 * math.exp(sigma), t50, sigma)
    assert f == pytest.approx(0.8413447, abs=1e-6)


def test_wearout_limits_and_monotonicity():
    assert wearout_fraction(0.0, 1e5, 0.5) == 0.0
    ts = [1e4, 3e4, 1e5, 3e5, 1e6]  # |z| <= 4.6: clear of erf underflow
    fs = [wearout_fraction(t, 1e5, 0.5) for t in ts]
    assert all(f2 > f1 for f1, f2 in zip(fs, fs[1:]))
    assert 0.0 < fs[0] and fs[-1] < 1.0


def test_wearout_rejects_bad_input():
    with pytest.raises(ValueError):
        wearout_fraction(-1.0, 1e5, 0.5)
    with pytest.raises(ValueError):
        wearout_fraction(1e4, 0.0, 0.5)
    with pytest.raises(ValueError):
        wearout_fraction(1e4, 1e5, 0.0)


# --- competing risks / module survival -------------------------------------------


def test_competing_risks_leq_each_factor():
    t, fit, t50, sigma = 8e4, 500.0, 2e5, 0.8
    s = module_survival(t, 1, fit, t50_hours=t50, sigma=sigma)
    s_random = survival_probability(t, fit)
    s_wear = 1.0 - wearout_fraction(t, t50, sigma)
    assert s <= s_random
    assert s <= s_wear
    assert s == pytest.approx(s_random * s_wear, rel=1e-12)


def test_module_survival_power_law_in_n():
    t, fit, t50, sigma = 5e4, 300.0, 3e5, 0.7
    s1 = module_survival(t, 1, fit, t50_hours=t50, sigma=sigma)
    for n in (2, 16, 64):
        assert module_survival(t, n, fit, t50_hours=t50, sigma=sigma) == pytest.approx(
            s1**n, rel=1e-9
        )


def test_module_survival_fit_only_matches_exponential():
    # Without wear-out the module is a series exponential system:
    # n lasers at fit == one pseudo-device at n*fit.
    t, fit, n = 1e5, 100.0, 32
    assert module_survival(t, n, fit) == pytest.approx(
        survival_probability(t, module_fit(fit, n)), rel=1e-12
    )


def test_module_survival_rejects_half_specified_wearout():
    with pytest.raises(ValueError):
        module_survival(1e4, 4, 100.0, t50_hours=1e5)
    with pytest.raises(ValueError):
        module_survival(1e4, 4, 100.0, sigma=0.5)
    with pytest.raises(ValueError):
        module_survival(1e4, 0, 100.0)


# --- WPE derating -----------------------------------------------------------------


def test_wpe_recovers_reference():
    assert wpe_derated(0.15, 50.0, 50.0, 0.01) == pytest.approx(0.15, rel=1e-12)


def test_wpe_analytic_value_and_monotone_decrease():
    # 0.20 * (1 - 0.01 * 20) = 0.16, straight from the formula.
    assert wpe_derated(0.20, 45.0, 65.0, 0.01) == pytest.approx(0.16, rel=1e-12)
    wpes = [wpe_derated(0.20, 45.0, t, 0.01) for t in (45.0, 55.0, 65.0, 85.0)]
    assert all(w2 < w1 for w1, w2 in zip(wpes, wpes[1:]))


def test_wpe_improves_below_reference_and_caps_at_one():
    assert wpe_derated(0.20, 45.0, 25.0, 0.01) == pytest.approx(0.24, rel=1e-12)
    # Absurd slope drives the linear model above 1.0; must cap at 1.0.
    assert wpe_derated(0.9, 45.0, 25.0, 0.05) == 1.0


def test_wpe_raises_when_derated_to_zero_or_below():
    with pytest.raises(ValueError):
        wpe_derated(0.15, 45.0, 145.0, 0.01)  # exactly zero
    with pytest.raises(ValueError):
        wpe_derated(0.15, 45.0, 200.0, 0.01)  # negative


def test_wpe_rejects_bad_input():
    with pytest.raises(ValueError):
        wpe_derated(0.0, 45.0, 55.0, 0.01)
    with pytest.raises(ValueError):
        wpe_derated(1.2, 45.0, 55.0, 0.01)
    with pytest.raises(ValueError):
        wpe_derated(0.15, 45.0, 55.0, -0.01)


def test_wpe_output_feeds_link_laser():
    # The documented handoff: result is a valid Laser wpe.
    from etalon.link import Laser

    wpe = wpe_derated(0.15, 50.0, 75.0, 0.008)
    laser = Laser(power_dbm=10.0, wpe=wpe, rin_db_hz=-145.0)
    assert 0.0 < laser.wpe < 0.15
