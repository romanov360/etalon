"""Tests for shot-noise (and dark-current) penalty in siphon.link.

Physics anchors: an independent bisection solve of the implicit top-eye
requirement (not the module's closed form), limiting behavior (thermal
domination), and monotonicity in dark current, baud, and extinction ratio.
"""

import math

import pytest

from siphon.constants import ELEMENTARY_CHARGE_C
from siphon.link import (
    NOISE_BANDWIDTH_FRACTION,
    Photodiode,
    Signaling,
    Tia,
    preset_cpo_optical_io,
    preset_pluggable_dr4,
    shot_penalty_db,
)

# DR4-like receiver used across the tests
PD = Photodiode(responsivity_a_per_w=0.8, dark_current_na=10.0)
TIA = Tia(input_noise_pa_per_sqrt_hz=16.0, bandwidth_ghz=40.0, energy_fj_per_bit=0.0)
SIG = Signaling(rate_gbd=53.125, levels=4, target_ber=2.4e-4)
ER_DB = 5.0


def _bisect_penalty_db(pd, tia, sig, er_db):
    """Independent re-derivation: bisection on the implicit equation.

    Solves  u R / (2 (M-1)) = q sqrt(i_n^2 + 2 q_e (R k u + I_d) f_n)
    for the required OMA u (W) without using the module's quadratic
    closed form, then returns 10 log10(u / u0).
    """
    q = sig.q_factor
    m = sig.levels
    r = pd.responsivity_a_per_w
    i_d = pd.dark_current_na * 1e-9
    f_n = NOISE_BANDWIDTH_FRACTION * sig.rate_gbd * 1e9
    i_n = tia.input_noise_pa_per_sqrt_hz * 1e-12 * math.sqrt(f_n)
    er = 10.0 ** (er_db / 10.0)
    k = 1.0 if math.isinf(er) else er / (er - 1.0)
    u0 = 2.0 * q * (m - 1) * i_n / r

    def g(u):
        noise = math.sqrt(i_n**2 + 2.0 * ELEMENTARY_CHARGE_C * (r * k * u + i_d) * f_n)
        return u * r / (2.0 * (m - 1)) - q * noise

    lo, hi = u0, 100.0 * u0
    assert g(lo) < 0.0 < g(hi)  # bracket: shot noise always raises the requirement
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 10.0 * math.log10(0.5 * (lo + hi) / u0)


def test_closed_form_matches_independent_bisection():
    # (1) closed-form quadratic root vs bisection on the implicit equation
    for er_db in (ER_DB, 3.0, math.inf):
        expected = _bisect_penalty_db(PD, TIA, SIG, er_db)
        got = shot_penalty_db(PD, TIA, SIG, extinction_ratio_db=er_db)
        assert got == pytest.approx(expected, abs=1e-9)
        assert got > 0.0


def test_penalty_vanishes_when_thermal_noise_dominates():
    # (2) huge TIA noise -> shot contribution negligible -> penalty -> 0
    tia_loud = Tia(
        input_noise_pa_per_sqrt_hz=1e6, bandwidth_ghz=40.0, energy_fj_per_bit=0.0
    )
    pen = shot_penalty_db(PD, tia_loud, SIG, extinction_ratio_db=ER_DB)
    assert 0.0 <= pen < 1e-5  # penalty scales ~1/i_n; 4e-6 dB here
    # and it decreases monotonically as thermal noise grows
    pens = [
        shot_penalty_db(
            PD,
            Tia(input_noise_pa_per_sqrt_hz=s, bandwidth_ghz=40.0, energy_fj_per_bit=0.0),
            SIG,
            extinction_ratio_db=ER_DB,
        )
        for s in (16.0, 160.0, 1600.0)
    ]
    assert pens[0] > pens[1] > pens[2]


def test_monotonic_in_dark_current_and_baud():
    # (3a) more dark current -> strictly larger penalty
    pens_dark = [
        shot_penalty_db(
            Photodiode(responsivity_a_per_w=0.8, dark_current_na=i_d),
            TIA,
            SIG,
            extinction_ratio_db=ER_DB,
        )
        for i_d in (0.0, 10.0, 1e3, 1e5)
    ]
    assert all(b > a for a, b in zip(pens_dark, pens_dark[1:]))
    # (3b) higher baud -> larger penalty (shot term grows ~f_n, thermal ~sqrt(f_n))
    pens_baud = [
        shot_penalty_db(
            PD,
            TIA,
            Signaling(rate_gbd=b, levels=4, target_ber=2.4e-4),
            extinction_ratio_db=ER_DB,
        )
        for b in (26.5625, 53.125, 106.25)
    ]
    assert all(b > a for a, b in zip(pens_baud, pens_baud[1:]))


def test_finite_er_exceeds_infinite_er():
    # (4) k = ER/(ER-1) > 1 raises P_top and hence the shot noise
    pen_fin = shot_penalty_db(PD, TIA, SIG, extinction_ratio_db=5.0)
    pen_inf = shot_penalty_db(PD, TIA, SIG)  # default infinite ER
    assert pen_fin > pen_inf > 0.0


@pytest.mark.parametrize("factory", [preset_pluggable_dr4, preset_cpo_optical_io])
def test_presets_still_positive_margin_and_report_row(factory):
    # (5) presets survive the extra penalty; report shows the shot row
    link = factory()
    assert link.shot_penalty_db > 0.0
    assert link.margin_db > 0.0
    text = link.report()
    assert "shot noise penalty" in text
    lines = text.splitlines()
    i_rin = next(i for i, ln in enumerate(lines) if "RIN penalty" in ln)
    assert "shot noise penalty" in lines[i_rin + 1]  # row right after RIN
    # composition: thermal + RIN + shot + explicit penalties, all in dB
    from siphon.link import receiver_sensitivity_oma_dbm

    expected = (
        receiver_sensitivity_oma_dbm(link.photodiode, link.tia, link.signaling)
        + link.rin_penalty_db
        + link.shot_penalty_db
        + sum(link.penalties_db.values())
    )
    assert link.sensitivity_oma_dbm == pytest.approx(expected, abs=1e-12)


def test_bad_extinction_ratio_raises():
    # (6) ER <= 0 dB is unphysical for this convention
    with pytest.raises(ValueError):
        shot_penalty_db(PD, TIA, SIG, extinction_ratio_db=0.0)
    with pytest.raises(ValueError):
        shot_penalty_db(PD, TIA, SIG, extinction_ratio_db=-3.0)
