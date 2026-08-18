"""Tests for whole-bank (jointly-coupled) yield in etalon.montecarlo.

Anchors are analytic where possible: pure common mode gives every ring the
same draw (bank yield == ring yield exactly), independent rings give bank
yield collapsing with N, and a metric that always fails one specific ring
must show bank_yield_above == 0 regardless of n_rings.
"""

import math

import numpy as np
import pytest

from etalon import montecarlo as mc


def identity(x):
    return np.asarray(x)


def test_pure_common_mode_bank_yield_equals_ring_yield():
    # sigma_diff = 0: every ring in a trial is the same number, so the
    # bank passes iff each ring passes -> yields are equal EXACTLY.
    params = {"x": mc.BankParam("x", mean=1.0, sigma_common=0.7, sigma_diff=0.0)}
    res = mc.run_bank(identity, params, n_rings=6, n_trials=5_000, seed=2)
    assert np.all(res.samples == res.samples[:, :1])
    for spec in (-0.5, 1.0, 2.0):
        assert res.bank_yield_above(spec) == res.ring_yield_above(spec)
    assert res.ring_yield_above(1.0) == pytest.approx(0.5, abs=0.02)


def test_independent_rings_bank_yield_is_ring_yield_power_n():
    # sigma_common = 0: rings are i.i.d., so P(all 4 pass) = p^4 with
    # p = P(N(0,1) > 0) = 0.5.
    params = {"x": mc.BankParam("x", mean=0.0, sigma_common=0.0, sigma_diff=1.0)}
    res = mc.run_bank(identity, params, n_rings=4, n_trials=20_000, seed=1)
    ring = res.ring_yield_above(0.0)
    bank = res.bank_yield_above(0.0)
    assert ring == pytest.approx(0.5, abs=0.01)
    assert bank == pytest.approx(ring**4, abs=0.008)


def test_reproducible_and_seed_sensitive():
    params = {"x": mc.BankParam("x", mean=0.0, sigma_common=1.0, sigma_diff=0.3)}
    r1 = mc.run_bank(identity, params, n_rings=4, n_trials=200, seed=42)
    r2 = mc.run_bank(identity, params, n_rings=4, n_trials=200, seed=42)
    r3 = mc.run_bank(identity, params, n_rings=4, n_trials=200, seed=43)
    np.testing.assert_array_equal(r1.samples, r2.samples)
    assert not np.array_equal(r1.samples, r3.samples)


def test_mixed_bankparam_and_plain_normal():
    params = {
        "a": mc.BankParam("a", mean=1.0, sigma_common=0.8, sigma_diff=0.2),
        "b": mc.Normal("b", mean=2.0, sigma=0.5),
    }
    res = mc.run_bank(lambda a, b: a + b, params, n_rings=2, n_trials=40_000, seed=6)
    assert res.mean == pytest.approx(3.0, abs=0.02)
    assert res.std == pytest.approx(math.sqrt(0.8**2 + 0.2**2 + 0.5**2), rel=0.02)
    # covariance between two rings of one trial comes only from the shared
    # common draw of "a": cov = sigma_common^2 = 0.64
    cov = np.cov(res.samples[:, 0], res.samples[:, 1])[0, 1]
    assert cov == pytest.approx(0.8**2, rel=0.05)


def test_whole_trial_failure_marks_every_ring_nan():
    # A metric that raises for the whole trial when any ring's draw is
    # extreme (mimics thermal.solve_coupled_powers's unreachable-target
    # ValueError) must NaN out every ring in that trial, not just one.
    def metric(x):
        if np.any(np.asarray(x) > 2.5):
            raise ValueError("unreachable corner")
        return np.asarray(x)

    params = {"x": mc.BankParam("x", mean=0.0, sigma_common=0.0, sigma_diff=1.0)}
    res = mc.run_bank(metric, params, n_rings=8, n_trials=5_000, seed=4)
    assert res.n_failed_trials > 0
    failed_rows = np.isnan(res.samples).all(axis=1)
    assert res.n_failed_trials == int(failed_rows.sum())
    # no PARTIAL nan rows: every row is either fully finite or fully NaN
    partial = np.isnan(res.samples).any(axis=1) & ~failed_rows
    assert not partial.any()
    assert "failed trials" in res.report()


def test_bank_yield_zero_when_metric_always_fails_one_ring():
    def metric(x):
        out = np.asarray(x, dtype=float).copy()
        out[0] = -1.0  # ring 0 always fails, regardless of draw
        return out

    params = {"x": mc.BankParam("x", mean=5.0, sigma_common=0.1, sigma_diff=0.1)}
    res = mc.run_bank(metric, params, n_rings=5, n_trials=1_000, seed=5)
    assert res.bank_yield_above(0.0) == 0.0
    assert res.ring_yield_above(0.0) > 0.0  # rings 1-4 mostly pass


def test_ring_yield_never_below_bank_yield():
    params = {
        "x": mc.BankParam("x", mean=0.0, sigma_common=0.5, sigma_diff=0.5),
        "u": mc.Uniform("u", low=-1.0, high=1.0),
    }
    res = mc.run_bank(lambda x, u: x + u, params, n_rings=8, n_trials=3_000, seed=7)
    for spec in np.linspace(-3.0, 3.0, 13):
        assert res.ring_yield_above(spec) >= res.bank_yield_above(spec)


def test_report_shows_ring_and_bank_yield_side_by_side():
    params = {"x": mc.BankParam("x", mean=0.0, sigma_common=0.5, sigma_diff=0.5)}
    res = mc.run_bank(identity, params, n_rings=4, n_trials=2_000, seed=8)
    text = res.report(threshold=0.0)
    assert "ring" in text and "bank" in text and "known-good-die" in text


def test_metric_wrong_shape_raises():
    def metric(x):
        return np.asarray(x)[:-1]  # wrong length

    params = {"x": mc.BankParam("x", mean=0.0, sigma_common=0.1, sigma_diff=0.1)}
    with pytest.raises(ValueError, match="shape"):
        mc.run_bank(metric, params, n_rings=4, n_trials=10)


def test_validation_errors():
    with pytest.raises(ValueError):
        mc.BankParam("x", 0.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        mc.BankParam("x", 0.0, 1.0, -1.0)
    good = {"x": mc.BankParam("x", 0.0, 1.0, 1.0)}
    with pytest.raises(ValueError):
        mc.run_bank(identity, good, n_rings=0, n_trials=10)
    with pytest.raises(ValueError):
        mc.run_bank(identity, good, n_rings=2, n_trials=0)
    with pytest.raises(ValueError):
        mc.run_bank(identity, {"y": mc.Normal("x", 0.0, 1.0)}, n_rings=2, n_trials=10)


def test_bankparam_key_mismatch_raises():
    # BankParam.name must match its dict key too, same as Normal/Uniform.
    params = {"wrong_key": mc.BankParam("actual_name", mean=0.0, sigma_common=1.0, sigma_diff=0.5)}
    with pytest.raises(ValueError, match="does not match"):
        mc.run_bank(identity, params, n_rings=3, n_trials=10)


def test_ring_yield_below_and_bank_yield_below():
    # Pure common mode again: below-spec yields should also match exactly.
    params = {"x": mc.BankParam("x", mean=1.0, sigma_common=0.7, sigma_diff=0.0)}
    res = mc.run_bank(identity, params, n_rings=6, n_trials=5_000, seed=2)
    for spec in (-0.5, 1.0, 2.0):
        assert res.bank_yield_below(spec) == res.ring_yield_below(spec)
    # below + above (strict) should sum to <= 1 (equal-to-spec excluded from both)
    assert res.ring_yield_below(1.0) + res.ring_yield_above(1.0) <= 1.0 + 1e-9


def test_thermal_integration_end_to_end():
    """The motivating use case: correlated fab offsets -> ring-assignment
    optimizer -> thermal-coupled solve, all inside one metric call, with
    an unreachable-lock trial correctly failing the whole bank."""
    from etalon import thermal, wdm

    n_rings = 6
    fsr_nm = 3.2
    layout = thermal.RingLayout.uniform(n_rings, 30.0)
    decay_um = 15.0
    budget_mw = 10.0

    def bank_margin_mw(offset_nm):
        offsets = np.asarray(offset_nm)
        assignment = wdm.optimize_ring_assignment(offsets, fsr_nm)
        target_nm = np.abs(np.array(assignment.per_ring_mw)) * wdm.TUNING_EFFICIENCY_NM_PER_MW
        result = thermal.solve_coupled_powers(target_nm, layout, decay_um)
        return budget_mw - np.array(result.heater_mw)

    params = {"offset_nm": mc.BankParam("offset_nm", mean=0.0, sigma_common=0.3, sigma_diff=0.5)}
    res = mc.run_bank(
        bank_margin_mw, params, n_rings=n_rings, n_trials=300, seed=1,
        metric_name="heater margin (mW)",
    )
    assert res.n_trials == 300
    assert res.n_rings == n_rings
    # some trials should be unlockable at this pitch/decay (a real finding,
    # not a bug -- matches examples/08_thermal_crosstalk.py's headline result)
    assert res.n_failed_trials > 0
    assert res.bank_yield_above(0.0) <= res.ring_yield_above(0.0)
