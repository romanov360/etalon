"""Tests for module-level (max-of-N) yield in siphon.montecarlo.

Anchors are analytic: independent lanes give module_yield = lane_yield**N,
pure common mode gives module_yield = lane_yield exactly, mixed draws have
lane variance sigma_common^2 + sigma_diff^2 + sigma^2 and between-lane
covariance sigma_common^2.
"""

import math

import numpy as np
import pytest

from siphon import montecarlo as mc


def identity(x):
    return x


def test_independent_lanes_module_yield_is_lane_yield_power_n():
    # sigma_common = 0: lanes are i.i.d., so P(all 4 pass) = p^4 with
    # p = P(N(0,1) > 0) = 0.5. Binomial std of module yield at n = 20_000
    # is sqrt(p^4 (1 - p^4) / n) ~ 0.0017, so abs=0.008 is ~5 sigma.
    params = {"x": mc.CommonDifferential(mean=0.0, sigma_common=0.0, sigma_diff=1.0)}
    res = mc.run_module(identity, params, n_lanes=4, n_modules=20_000, seed=1)
    lane = res.lane_yield_above(0.0)
    module = res.module_yield_above(0.0)
    assert lane == pytest.approx(0.5, abs=0.01)
    assert module == pytest.approx(lane**4, abs=0.008)
    assert module == pytest.approx(0.5**4, abs=0.008)


def test_pure_common_mode_module_yield_equals_lane_yield():
    # sigma_diff = 0: every lane of a module is the same number, so the
    # module passes iff each lane passes -> yields are equal EXACTLY.
    params = {"x": mc.CommonDifferential(mean=1.0, sigma_common=0.7, sigma_diff=0.0)}
    res = mc.run_module(identity, params, n_lanes=8, n_modules=5_000, seed=2)
    assert np.all(res.samples == res.samples[:, :1])
    for spec in (-0.5, 1.0, 2.0):
        assert res.module_yield_above(spec) == res.lane_yield_above(spec)
        assert res.module_yield_below(spec) == res.lane_yield_below(spec)
    # threshold at the mean: symmetric normal -> lane yield ~ 0.5
    assert res.lane_yield_above(1.0) == pytest.approx(0.5, abs=0.02)


def test_truncation_bounds_respected_without_clipping():
    params = {
        "x": mc.CommonDifferential(
            mean=0.0, sigma_common=1.0, sigma_diff=0.5, low=-1.0, high=1.0
        )
    }
    res = mc.run_module(identity, params, n_lanes=4, n_modules=5_000, seed=3)
    assert res.samples.min() >= -1.0 and res.samples.max() <= 1.0
    # resampling, not clipping: no probability mass piles at the bounds
    assert np.mean(np.isclose(res.samples, 1.0, atol=1e-6)) < 1e-3
    assert np.mean(np.isclose(res.samples, -1.0, atol=1e-6)) < 1e-3


def test_unreachable_truncation_raises():
    params = {"x": mc.CommonDifferential(0.0, 0.0, 0.0, low=1.0, high=2.0)}
    with pytest.raises(ValueError):
        mc.run_module(identity, params, n_lanes=2, n_modules=10)


def test_metric_failure_nan_fails_lane_and_module():
    def metric(x):
        if x > 0.0:
            raise ValueError("unreachable corner")
        return 1.0

    params = {"x": mc.Normal("x", mean=0.0, sigma=1.0)}
    res = mc.run_module(metric, params, n_lanes=2, n_modules=4_000, seed=4)
    nan = np.isnan(res.samples)
    assert res.n_failed == int(nan.sum()) and res.n_failed > 0
    # a NaN lane fails, a module with any NaN lane fails
    assert res.lane_yield_above(0.0) == pytest.approx(1.0 - nan.mean())
    assert res.module_yield_above(0.0) == pytest.approx(float(np.mean(~nan.any(axis=1))))
    # independent lanes each fail with p = 0.5 -> module pass ~ 0.25
    assert res.module_yield_above(0.0) == pytest.approx(0.25, abs=0.03)
    assert res.mean == pytest.approx(1.0)  # NaN excluded from stats
    assert "failed lanes" in res.report()


def test_reproducible_and_seed_sensitive():
    params = {"x": mc.CommonDifferential(mean=0.0, sigma_common=1.0, sigma_diff=0.3)}
    r1 = mc.run_module(identity, params, n_lanes=4, n_modules=200, seed=42)
    r2 = mc.run_module(identity, params, n_lanes=4, n_modules=200, seed=42)
    r3 = mc.run_module(identity, params, n_lanes=4, n_modules=200, seed=43)
    np.testing.assert_array_equal(r1.samples, r2.samples)
    assert not np.array_equal(r1.samples, r3.samples)


def test_mixed_common_differential_and_plain_normal():
    params = {
        "a": mc.CommonDifferential(mean=1.0, sigma_common=0.8, sigma_diff=0.2),
        "b": mc.Normal("b", mean=2.0, sigma=0.5),
    }
    res = mc.run_module(lambda a, b: a + b, params, n_lanes=2, n_modules=40_000, seed=6)
    assert res.mean == pytest.approx(3.0, abs=0.02)
    # lane variance adds all three independent parts
    assert res.std == pytest.approx(math.sqrt(0.8**2 + 0.2**2 + 0.5**2), rel=0.02)
    # covariance between two lanes of one module comes only from the shared
    # common draw of "a": cov = sigma_common^2 = 0.64
    cov = np.cov(res.samples[:, 0], res.samples[:, 1])[0, 1]
    assert cov == pytest.approx(0.8**2, rel=0.05)


def test_lane_yield_never_below_module_yield():
    params = {
        "x": mc.CommonDifferential(mean=0.0, sigma_common=0.5, sigma_diff=0.5),
        "u": mc.Uniform("u", low=-1.0, high=1.0),
    }
    res = mc.run_module(lambda x, u: x + u, params, n_lanes=8, n_modules=3_000, seed=7)
    for spec in np.linspace(-3.0, 3.0, 13):
        assert res.lane_yield_above(spec) >= res.module_yield_above(spec)
        assert res.lane_yield_below(spec) >= res.module_yield_below(spec)


def test_report_shows_lane_and_module_yield_side_by_side():
    params = {"x": mc.CommonDifferential(mean=0.0, sigma_common=0.5, sigma_diff=0.5)}
    res = mc.run_module(identity, params, n_lanes=4, n_modules=2_000, seed=8)
    text = res.report(threshold=0.0)
    assert "lane" in text and "module" in text and "known-good-die" in text


def test_validation_errors():
    with pytest.raises(ValueError):
        mc.CommonDifferential(0.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        mc.CommonDifferential(0.0, 1.0, -1.0)
    with pytest.raises(ValueError):
        mc.CommonDifferential(0.0, 1.0, 1.0, low=2.0, high=1.0)
    good = {"x": mc.CommonDifferential(0.0, 1.0, 1.0)}
    with pytest.raises(ValueError):
        mc.run_module(identity, good, n_lanes=0, n_modules=10)
    with pytest.raises(ValueError):
        mc.run_module(identity, good, n_lanes=2, n_modules=0)
    with pytest.raises(ValueError):
        mc.run_module(identity, {"y": mc.Normal("x", 0.0, 1.0)}, n_lanes=2, n_modules=10)
