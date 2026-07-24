"""Monte Carlo corner analysis for link budgets and circuit metrics.

Photonics has no CMOS-grade statistical corner models; the practical
substitute is transparent Monte Carlo over declared parameter variations.
This module runs any scalar metric (link margin, insertion loss, Q, ...)
over sampled process/environment corners and reports distribution statistics
and parametric yield.

Typical use::

    params = [
        Normal("gc_loss_db", mean=1.75, sigma=0.25, low=0.0),
        Uniform("laser_power_dbm", low=11.0, high=13.0),
    ]

    def margin(p):
        link = make_link(gc=p["gc_loss_db"], laser=p["laser_power_dbm"])
        return link.margin_db

    result = run(margin, params, n=10_000, seed=1)
    print(result.report(threshold=0.0))
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Normal:
    """Normally distributed parameter, optionally truncated to [low, high].

    Truncation is by resampling, so the realized distribution is the exact
    truncated normal (no clipping mass at the bounds).
    """

    name: str
    mean: float
    sigma: float
    low: float | None = None
    high: float | None = None

    def __post_init__(self):
        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")
        if self.low is not None and self.high is not None and self.low >= self.high:
            raise ValueError("low must be < high")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        x = rng.normal(self.mean, self.sigma, n)
        lo = -np.inf if self.low is None else self.low
        hi = np.inf if self.high is None else self.high
        for _ in range(1000):
            bad = (x < lo) | (x > hi)
            if not bad.any():
                return x
            x[bad] = rng.normal(self.mean, self.sigma, int(bad.sum()))
        raise ValueError(f"{self.name}: truncation bounds reject nearly all samples")


@dataclass(frozen=True)
class Uniform:
    """Uniformly distributed parameter on [low, high]."""

    name: str
    low: float
    high: float

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError("low must be < high")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, n)


Param = Normal | Uniform


@dataclass
class MonteCarloResult:
    """Samples and statistics of one scalar metric over parameter corners."""

    metric_name: str
    samples: np.ndarray  # shape (n,)
    param_samples: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return self.samples.size

    @property
    def n_failed(self) -> int:
        """Corners where the metric raised (recorded as NaN)."""
        return int(np.isnan(self.samples).sum())

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.samples))

    @property
    def std(self) -> float:
        ok = self.n - self.n_failed
        return float(np.nanstd(self.samples, ddof=1)) if ok > 1 else 0.0

    def percentile(self, p) -> float:
        return float(np.nanpercentile(self.samples, p))

    def yield_above(self, threshold: float) -> float:
        """Fraction of samples with metric > threshold (parametric yield)."""
        return float(np.mean(self.samples > threshold))

    def sensitivity(self) -> dict[str, float]:
        """Pearson correlation of each parameter with the metric.

        A cheap screening tool: |r| near 1 means the parameter dominates the
        metric spread (for near-linear responses). Constant parameters get 0.
        """
        out: dict[str, float] = {}
        m = self.samples
        ms = m - m.mean()
        for name, x in self.param_samples.items():
            xs = x - x.mean()
            denom = np.sqrt((xs**2).sum() * (ms**2).sum())
            out[name] = float((xs * ms).sum() / denom) if denom > 0 else 0.0
        return out

    def report(self, threshold: float | None = None, width: int = 50) -> str:
        """Plain-text distribution report with an ASCII histogram."""
        lines = [
            f"Monte Carlo — {self.metric_name}  (n = {self.n})",
            "-" * 66,
            f"  mean {self.mean:+9.3f}   std {self.std:8.3f}   "
            f"P5 {self.percentile(5):+8.3f}   P50 {self.percentile(50):+8.3f}   "
            f"P95 {self.percentile(95):+8.3f}",
        ]
        if threshold is not None:
            lines.append(
                f"  yield ({self.metric_name} > {threshold:g}): "
                f"{100.0 * self.yield_above(threshold):6.2f} %"
            )
        if self.n_failed:
            lines.append(f"  failed corners (metric raised): {self.n_failed}")
        counts, edges = np.histogram(self.samples[~np.isnan(self.samples)], bins=20)
        peak = counts.max() if counts.max() > 0 else 1
        lines.append("")
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            bar = "#" * max(0, round(width * c / peak))
            lines.append(f"  {lo:+9.3f} .. {hi:+9.3f} | {bar}")
        sens = sorted(self.sensitivity().items(), key=lambda kv: -abs(kv[1]))
        lines.append("")
        lines.append("  sensitivity (Pearson r): " + ", ".join(f"{k} {v:+.2f}" for k, v in sens))
        return "\n".join(lines)


def run(
    metric: Callable[[dict[str, float]], float],
    params: Sequence[Param],
    n: int = 10_000,
    seed: int = 0,
    metric_name: str = "metric",
) -> MonteCarloResult:
    """Evaluate ``metric`` over ``n`` sampled corners of ``params``.

    ``metric`` receives one dict mapping parameter name -> sampled value and
    returns a scalar. Sampling is reproducible for a given seed. Corners where
    ``metric`` raises ValueError are recorded as NaN (e.g. a corner where the
    RIN penalty is unreachable) and excluded from mean/std/percentiles, but a
    failing corner is itself yield-relevant, so NaN counts as a yield loss in
    :meth:`MonteCarloResult.yield_above`.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    names = [p.name for p in params]
    if len(set(names)) != len(names):
        raise ValueError("duplicate parameter names")
    rng = np.random.default_rng(seed)
    draws = {p.name: p.sample(n, rng) for p in params}
    out = np.empty(n)
    for i in range(n):
        corner = {name: float(draws[name][i]) for name in names}
        try:
            out[i] = float(metric(corner))
        except ValueError:
            out[i] = np.nan
    return MonteCarloResult(metric_name=metric_name, samples=out, param_samples=draws)
