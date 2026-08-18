"""Tests for etalon.montecarlo against exact distributional facts."""

import math

import numpy as np
import pytest

from etalon import montecarlo as mc


def test_normal_moments():
    res = mc.run(lambda p: p["x"], [mc.Normal("x", mean=2.0, sigma=0.5)], n=40_000, seed=3)
    assert res.mean == pytest.approx(2.0, abs=0.01)
    assert res.std == pytest.approx(0.5, abs=0.01)


def test_uniform_bounds_and_moments():
    res = mc.run(lambda p: p["u"], [mc.Uniform("u", low=1.0, high=3.0)], n=40_000, seed=7)
    assert res.samples.min() >= 1.0 and res.samples.max() <= 3.0
    assert res.mean == pytest.approx(2.0, abs=0.01)
    assert res.std == pytest.approx(2.0 / math.sqrt(12.0), abs=0.01)


def test_truncated_normal_respects_bounds():
    res = mc.run(
        lambda p: p["x"],
        [mc.Normal("x", mean=0.0, sigma=1.0, low=-1.0, high=1.0)],
        n=20_000,
        seed=1,
    )
    assert res.samples.min() >= -1.0 and res.samples.max() <= 1.0
    # exact truncated normal, not clipping: no mass piles at the bounds
    assert np.mean(np.isclose(res.samples, 1.0, atol=1e-6)) < 1e-3


def test_linear_combination_variance():
    params = [mc.Normal("a", 0.0, 1.0), mc.Normal("b", 0.0, 2.0)]
    res = mc.run(lambda p: p["a"] + p["b"], params, n=40_000, seed=5)
    assert res.std == pytest.approx(math.sqrt(1.0 + 4.0), rel=0.02)


def test_yield_matches_gaussian_tail():
    # metric ~ N(1, 1); P(metric > 0) = Phi(1) ~ 0.8413
    res = mc.run(lambda p: p["x"], [mc.Normal("x", 1.0, 1.0)], n=100_000, seed=11)
    assert res.yield_above(0.0) == pytest.approx(0.8413, abs=0.005)


def test_reproducible_and_seed_sensitive():
    params = [mc.Normal("x", 0.0, 1.0)]
    r1 = mc.run(lambda p: p["x"], params, n=100, seed=42)
    r2 = mc.run(lambda p: p["x"], params, n=100, seed=42)
    r3 = mc.run(lambda p: p["x"], params, n=100, seed=43)
    np.testing.assert_array_equal(r1.samples, r2.samples)
    assert not np.array_equal(r1.samples, r3.samples)


def test_failed_corners_are_nan_and_hurt_yield():
    def metric(p):
        if p["x"] > 0:
            raise ValueError("unreachable corner")
        return 1.0

    res = mc.run(metric, [mc.Normal("x", 0.0, 1.0)], n=4_000, seed=2)
    assert 0.4 < res.n_failed / res.n < 0.6
    assert res.mean == pytest.approx(1.0)  # NaN excluded from stats
    assert res.yield_above(0.0) == pytest.approx(1.0 - res.n_failed / res.n)
    assert "failed corners" in res.report()


def test_sensitivity_ranks_dominant_parameter():
    params = [mc.Normal("big", 0.0, 10.0), mc.Normal("small", 0.0, 0.1)]
    res = mc.run(lambda p: p["big"] + p["small"], params, n=5_000, seed=9)
    s = res.sensitivity()
    assert abs(s["big"]) > 0.99
    assert abs(s["small"]) < 0.1


def test_report_contains_stats_and_histogram():
    res = mc.run(lambda p: p["x"], [mc.Normal("x", 0.0, 1.0)], n=2_000, seed=0)
    text = res.report(threshold=0.0)
    assert "yield" in text and "#" in text and "sensitivity" in text


def test_validation_errors():
    with pytest.raises(ValueError):
        mc.Normal("x", 0.0, -1.0)
    with pytest.raises(ValueError):
        mc.Uniform("u", 2.0, 1.0)
    with pytest.raises(ValueError):
        mc.Normal("x", 0.0, 1.0, low=2.0, high=1.0)
    with pytest.raises(ValueError):
        mc.run(lambda p: 0.0, [mc.Normal("x", 0, 1), mc.Uniform("x", 0, 1)], n=10)
    with pytest.raises(ValueError):
        mc.run(lambda p: 0.0, [mc.Normal("x", 0, 1)], n=0)
