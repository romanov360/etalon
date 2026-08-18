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

Module-level (max-of-N) yield
-----------------------------
A co-packaged-optics module carries N parallel lanes and is scrapped if any
single lane misses spec, so module yield is P(ALL lanes pass) — not the
per-lane yield. Within one die the variation is mostly common-mode (all
lanes shift together with lithography, film thickness, temperature) with a
smaller lane-to-lane differential spread; :class:`CommonDifferential`
models that split and :func:`run_module` propagates it to lane and module
yield. The gap between the two IS the known-good-die problem.

Whole-bank (jointly-coupled) yield
-----------------------------------
:func:`run_module` calls its metric once per lane, independently — fine
when each lane's outcome only depends on that lane's own draws. A WDM ring
bank breaks that assumption: :func:`etalon.thermal.solve_coupled_powers`
solves a linear system across ALL rings on the bus at once, so ring i's
achievable margin depends on every other ring's fabrication offset in the
same trial. :func:`run_bank` and :class:`BankParam` give the metric the
whole bank's draws per trial instead of one lane's, and
:class:`BankYieldResult` reports the same ring-vs-bank known-good-die gap
as :class:`ModuleYieldResult` — plus a bank-wide failure mode (crosstalk
makes the assignment physically unlockable) that has no per-lane analogue.

All of this is architecture-level statistics over declared variations, not
foundry-calibrated signoff: use it to size margins and lane counts, not to
predict absolute yield of a specific process.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


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


@dataclass(frozen=True)
class CommonDifferential:
    """Parameter with a shared common-mode part and a per-lane differential part.

    Models within-die variation of an N-lane module: one common draw
    ``common ~ N(0, sigma_common)`` is shared by every lane of a module
    (lithography, film thickness, die temperature move all lanes together),
    and each lane adds an independent ``diff ~ N(0, sigma_diff)``::

        value[trial, lane] = mean + common[trial] + diff[trial, lane]

    so the marginal of each lane is N(mean, sqrt(sigma_common^2 +
    sigma_diff^2)) and the correlation between two lanes of the same module
    is sigma_common^2 / (sigma_common^2 + sigma_diff^2). Units are whatever
    the named parameter uses (dB, um, nm, ...).

    Optional ``low``/``high`` truncate the TOTAL value (no clipping mass at
    the bounds). Semantics: the common draw is NEVER truncated — it is the
    die-level shift and stays exactly Gaussian, so the shared common mode
    and within-module correlation are preserved by construction — and each
    lane's differential is then drawn from the normal conditionally
    truncated to ``[low - mean - common, high - mean - common]`` via
    inverse-CDF sampling, which is valid however far in the tail the
    window sits (no rejection loop, no spurious sampling failure). Every
    total lies in [low, high]. In the ``sigma_diff = 0`` limit the common
    draw itself is the exact truncated normal, matching :class:`Normal`;
    with ``sigma_diff > 0`` a lane's marginal is a Gaussian mixture of
    conditionally truncated normals — close to, but not exactly, the
    truncated N(mean, sqrt(sigma_common^2 + sigma_diff^2)), and biased
    toward the bounds when they cut within ~1 total sigma of the mean
    (declare wider bounds or fold hard limits into the metric if that
    regime matters).
    """

    mean: float
    sigma_common: float
    sigma_diff: float
    low: float | None = None
    high: float | None = None

    def __post_init__(self):
        if self.sigma_common < 0:
            raise ValueError("sigma_common must be >= 0")
        if self.sigma_diff < 0:
            raise ValueError("sigma_diff must be >= 0")
        if self.low is not None and self.high is not None and self.low >= self.high:
            raise ValueError("low must be < high")

    def sample(self, n_modules: int, n_lanes: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a (n_modules, n_lanes) array with one common draw per module."""
        shape = (n_modules, n_lanes)
        if self.low is None and self.high is None:
            common = rng.normal(0.0, self.sigma_common, n_modules)
            return self.mean + common[:, None] + rng.normal(0.0, self.sigma_diff, shape)
        lo = -np.inf if self.low is None else self.low
        hi = np.inf if self.high is None else self.high
        if self.sigma_diff == 0.0:
            if self.sigma_common == 0.0:
                if not lo <= self.mean <= hi:
                    raise ValueError(
                        "CommonDifferential: constant value lies outside [low, high]"
                    )
                return np.full(shape, self.mean)
            a = (lo - self.mean) / self.sigma_common
            b = (hi - self.mean) / self.sigma_common
            common = stats.truncnorm.rvs(
                a, b, loc=self.mean, scale=self.sigma_common,
                size=n_modules, random_state=rng,
            )
            return np.broadcast_to(common[:, None], shape).copy()
        common = rng.normal(0.0, self.sigma_common, n_modules)
        center = np.broadcast_to(self.mean + common[:, None], shape)
        a_arr = (lo - center) / self.sigma_diff
        b_arr = (hi - center) / self.sigma_diff
        return stats.truncnorm.rvs(
            a_arr, b_arr, loc=center, scale=self.sigma_diff, size=shape, random_state=rng
        )


ModuleParam = CommonDifferential | Normal | Uniform


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
        ok = ~np.isnan(self.samples)
        m = self.samples[ok]
        if m.size < 2:
            return {name: 0.0 for name in self.param_samples}
        ms = m - m.mean()
        for name, x in self.param_samples.items():
            xs = x[ok] - x[ok].mean()
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


@dataclass
class ModuleYieldResult:
    """Lane-metric samples and lane vs module yield of an N-lane module.

    ``samples[i, j]`` is the metric of lane j in module trial i; a NaN entry
    is a lane whose metric raised ValueError — that lane fails every spec,
    and with it its module. The difference between :meth:`lane_yield_above`
    and :meth:`module_yield_above` is the known-good-die (KGD) problem: with
    N lanes per module and any lane-level fallout, module yield collapses
    toward lane_yield**N unless the variation is common-mode, which is why
    CPO economics hinge on pre-bond lane test and on how much of the spread
    is shared across a die.
    """

    metric_name: str
    samples: np.ndarray  # shape (n_modules, n_lanes)
    param_samples: dict[str, np.ndarray]  # each shape (n_modules, n_lanes)

    @property
    def n_modules(self) -> int:
        return self.samples.shape[0]

    @property
    def n_lanes(self) -> int:
        return self.samples.shape[1]

    @property
    def n_failed(self) -> int:
        """Lanes (not modules) where the metric raised (recorded as NaN)."""
        return int(np.isnan(self.samples).sum())

    @property
    def mean(self) -> float:
        """NaN-aware mean of the lane metric over all lanes and modules."""
        return float(np.nanmean(self.samples))

    @property
    def std(self) -> float:
        """NaN-aware sample std (ddof=1) of the lane metric."""
        ok = self.samples.size - self.n_failed
        return float(np.nanstd(self.samples, ddof=1)) if ok > 1 else 0.0

    def percentile(self, p) -> float:
        """NaN-aware percentile of the lane metric."""
        return float(np.nanpercentile(self.samples, p))

    def lane_yield_above(self, spec: float) -> float:
        """Fraction of lanes with metric > spec. NaN lanes count as failed."""
        return float(np.mean(self.samples > spec))

    def lane_yield_below(self, spec: float) -> float:
        """Fraction of lanes with metric < spec. NaN lanes count as failed."""
        return float(np.mean(self.samples < spec))

    def module_yield_above(self, spec: float) -> float:
        """Fraction of modules where ALL lanes have metric > spec.

        One bad (or NaN) lane scraps the module — this is the max-of-N /
        known-good-die statistic, always <= :meth:`lane_yield_above`.
        """
        return float(np.mean(np.all(self.samples > spec, axis=1)))

    def module_yield_below(self, spec: float) -> float:
        """Fraction of modules where ALL lanes have metric < spec.

        One bad (or NaN) lane scraps the module — always <=
        :meth:`lane_yield_below`.
        """
        return float(np.mean(np.all(self.samples < spec, axis=1)))

    def report(self, threshold: float | None = None) -> str:
        """Plain-text report showing lane and module yield side by side."""
        lines = [
            f"Module yield — {self.metric_name}  "
            f"(n_modules = {self.n_modules}, n_lanes = {self.n_lanes})",
            "-" * 66,
            f"  lane metric: mean {self.mean:+9.3f}   std {self.std:8.3f}   "
            f"P5 {self.percentile(5):+8.3f}   P50 {self.percentile(50):+8.3f}   "
            f"P95 {self.percentile(95):+8.3f}",
        ]
        if threshold is not None:
            lane = self.lane_yield_above(threshold)
            module = self.module_yield_above(threshold)
            lines.append(
                f"  yield ({self.metric_name} > {threshold:g}):   "
                f"lane {100.0 * lane:6.2f} %   module {100.0 * module:6.2f} %   "
                f"(known-good-die gap {100.0 * (lane - module):.2f} pts)"
            )
        if self.n_failed:
            lines.append(f"  failed lanes (metric raised): {self.n_failed}")
        return "\n".join(lines)


def run_module(
    metric: Callable[..., float],
    params: dict[str, ModuleParam],
    n_lanes: int,
    n_modules: int,
    seed: int = 0,
    metric_name: str = "metric",
) -> ModuleYieldResult:
    """Monte Carlo over ``n_modules`` module trials of ``n_lanes`` lanes each.

    ``params`` maps parameter name -> distribution. A
    :class:`CommonDifferential` shares one common draw across the lanes of a
    module (plus an independent per-lane differential draw); plain
    :class:`Normal` / :class:`Uniform` parameters are drawn fully
    independently per lane (the sigma_common = 0 equivalent), and their
    ``name`` must match the dict key.

    CAUTION: a physically module-shared quantity (one laser feeding all
    lanes, die temperature) must be expressed as
    ``CommonDifferential(mean, sigma_common=s, sigma_diff=0)`` — modeling
    it as a plain Normal/Uniform silently draws it independently per lane,
    decorrelating lane failures and understating module yield. There is
    currently no shared-Uniform equivalent; approximate one with a
    Gaussian of matched spread or fold it into the metric.

    ``metric`` is called per lane with keyword arguments, one scalar per
    parameter: ``metric(**{name: value})`` (note: :func:`run` passes a single
    dict instead). Evaluation is a Python loop of n_modules * n_lanes calls,
    so cost scales linearly with both; keep the metric cheap or the counts
    modest. Lanes where the metric raises ValueError are recorded as NaN:
    the lane fails any spec and scraps its module, and it is excluded from
    mean/std/percentiles. Sampling is reproducible for a given seed.
    """
    if n_lanes < 1:
        raise ValueError("n_lanes must be >= 1")
    if n_modules < 1:
        raise ValueError("n_modules must be >= 1")
    for name, p in params.items():
        if isinstance(p, (Normal, Uniform)) and p.name != name:
            raise ValueError(f"parameter key {name!r} does not match its name {p.name!r}")
    rng = np.random.default_rng(seed)
    draws: dict[str, np.ndarray] = {}
    for name, p in params.items():
        if isinstance(p, CommonDifferential):
            draws[name] = p.sample(n_modules, n_lanes, rng)
        else:
            draws[name] = p.sample(n_modules * n_lanes, rng).reshape(n_modules, n_lanes)
    out = np.empty((n_modules, n_lanes))
    for i in range(n_modules):
        for j in range(n_lanes):
            corner = {name: float(x[i, j]) for name, x in draws.items()}
            try:
                out[i, j] = float(metric(**corner))
            except ValueError:
                out[i, j] = np.nan
    return ModuleYieldResult(metric_name=metric_name, samples=out, param_samples=draws)


# --- whole-bank (jointly-coupled) yield ---------------------------------------


@dataclass(frozen=True)
class BankParam:
    """Parameter correlated across the N rings of ONE WDM bank, per trial.

    Same common+differential split as :class:`CommonDifferential`
    (``value[trial, ring] = mean + common[trial] + diff[trial, ring]``),
    but sampled as a full ring array per trial for :func:`run_bank` rather
    than per-lane for :func:`run_module` — the distinction matters because
    :func:`run_bank`'s metric sees the WHOLE bank at once (needed for
    :func:`etalon.thermal.solve_coupled_powers`, which cannot be evaluated
    ring-by-ring: one ring's required heater power depends on every other
    ring's draw in the same trial). No truncation support (unlike
    :class:`CommonDifferential`) — fold hard bounds into the metric if
    needed, since a bank-wide metric usually wants to see out-of-bound
    draws as part of the physics (e.g. an unreachable thermal lock) rather
    than have them resampled away.
    """

    name: str
    mean: float
    sigma_common: float
    sigma_diff: float

    def __post_init__(self):
        if self.sigma_common < 0:
            raise ValueError("sigma_common must be >= 0")
        if self.sigma_diff < 0:
            raise ValueError("sigma_diff must be >= 0")

    def sample(self, n_trials: int, n_rings: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a (n_trials, n_rings) array with one common draw per trial."""
        common = rng.normal(0.0, self.sigma_common, n_trials)
        diff = rng.normal(0.0, self.sigma_diff, (n_trials, n_rings))
        return self.mean + common[:, None] + diff


BankSample = BankParam | Normal | Uniform


@dataclass
class BankYieldResult:
    """Per-ring metric samples and ring vs bank yield of an N-ring WDM bank.

    ``samples[i, j]`` is the metric of ring j in bank trial i. A whole ROW
    of NaN marks a trial where :func:`run_bank`'s metric raised ValueError
    for the trial as a whole (e.g. :func:`etalon.thermal.solve_coupled_powers`
    finding the crosstalk-coupled lock unreachable) — the failure is
    bank-wide, not attributable to one ring, so every ring in that trial is
    recorded as failed. The gap between :meth:`ring_yield_above` and
    :meth:`bank_yield_above` is the same known-good-die shape as
    :class:`ModuleYieldResult`, but for channels sharing one WDM bank
    instead of lanes sharing one die: correlated fabrication offsets alone
    move some rings together, while ring-to-ring thermal crosstalk (unlike
    plain common-mode variation) can additionally make some assignments
    physically unlockable regardless of power spent — a failure mode with
    no per-lane analogue in :class:`ModuleYieldResult`.
    """

    metric_name: str
    samples: np.ndarray  # shape (n_trials, n_rings)
    param_samples: dict[str, np.ndarray]  # each shape (n_trials, n_rings)

    @property
    def n_trials(self) -> int:
        return self.samples.shape[0]

    @property
    def n_rings(self) -> int:
        return self.samples.shape[1]

    @property
    def n_failed_trials(self) -> int:
        """Trials where the metric raised for the whole bank (all-NaN row)."""
        return int(np.all(np.isnan(self.samples), axis=1).sum())

    @property
    def mean(self) -> float:
        """NaN-aware mean of the per-ring metric over all trials and rings."""
        return float(np.nanmean(self.samples))

    @property
    def std(self) -> float:
        ok = self.samples.size - int(np.isnan(self.samples).sum())
        return float(np.nanstd(self.samples, ddof=1)) if ok > 1 else 0.0

    def percentile(self, p) -> float:
        return float(np.nanpercentile(self.samples, p))

    def ring_yield_above(self, spec: float) -> float:
        """Fraction of (trial, ring) pairs with metric > spec. NaN counts as failed."""
        return float(np.mean(self.samples > spec))

    def ring_yield_below(self, spec: float) -> float:
        """Fraction of (trial, ring) pairs with metric < spec. NaN counts as failed."""
        return float(np.mean(self.samples < spec))

    def bank_yield_above(self, spec: float) -> float:
        """Fraction of trials where EVERY ring has metric > spec.

        The max-of-N statistic for a whole bank, always
        <= :meth:`ring_yield_above`. A failed trial (all-NaN row) counts as
        a bank failure automatically, since ``NaN > spec`` is False.
        """
        return float(np.mean(np.all(self.samples > spec, axis=1)))

    def bank_yield_below(self, spec: float) -> float:
        """Fraction of trials where EVERY ring has metric < spec.

        For specs that fail on being too HIGH (e.g. a max-temperature or
        max-crosstalk limit) rather than too LOW. Always
        <= :meth:`ring_yield_below`; a failed trial (all-NaN row) counts as
        a bank failure automatically, since ``NaN < spec`` is False.
        """
        return float(np.mean(np.all(self.samples < spec, axis=1)))

    def report(self, threshold: float | None = None) -> str:
        """Plain-text report showing ring and bank yield side by side."""
        lines = [
            f"Bank yield — {self.metric_name}  "
            f"(n_trials = {self.n_trials}, n_rings = {self.n_rings})",
            "-" * 66,
            f"  ring metric: mean {self.mean:+9.3f}   std {self.std:8.3f}   "
            f"P5 {self.percentile(5):+8.3f}   P50 {self.percentile(50):+8.3f}   "
            f"P95 {self.percentile(95):+8.3f}",
        ]
        if threshold is not None:
            ring = self.ring_yield_above(threshold)
            bank = self.bank_yield_above(threshold)
            lines.append(
                f"  yield ({self.metric_name} > {threshold:g}):   "
                f"ring {100.0 * ring:6.2f} %   bank {100.0 * bank:6.2f} %   "
                f"(known-good-die gap {100.0 * (ring - bank):.2f} pts)"
            )
        if self.n_failed_trials:
            lines.append(
                f"  failed trials (metric raised for the whole bank): "
                f"{self.n_failed_trials}"
            )
        return "\n".join(lines)


def run_bank(
    metric: Callable[..., np.ndarray],
    params: dict[str, BankSample],
    n_rings: int,
    n_trials: int,
    seed: int = 0,
    metric_name: str = "metric",
) -> BankYieldResult:
    """Monte Carlo over ``n_trials`` whole-bank draws of ``n_rings`` rings each.

    Unlike :func:`run_module` (which calls ``metric`` once per lane with
    scalar arguments), ``metric`` here is called ONCE PER TRIAL with each
    parameter's FULL ``(n_rings,)`` array, and must return an
    ``(n_rings,)`` array of per-ring metric values (or raise ValueError to
    fail the whole trial). This whole-bank shape is required whenever the
    metric involves a joint solve across rings — the motivating case is
    :func:`etalon.thermal.solve_coupled_powers`, where ring i's required
    heater power depends on every other ring's fabrication offset in the
    same trial, so the metric cannot be decomposed into independent
    per-ring calls the way :func:`run_module`'s can.

    ``params`` maps parameter name -> distribution. A :class:`BankParam`
    shares one common (die-level) draw across the rings of a trial, plus
    an independent per-ring differential draw; plain :class:`Normal` /
    :class:`Uniform` parameters are drawn fully independently per ring
    (the ``sigma_common = 0`` equivalent), and their ``name`` must match
    the dict key.

    A trial where ``metric`` raises ValueError is recorded as an all-NaN
    row (every ring in that trial fails — see :class:`BankYieldResult`).
    Sampling is reproducible for a given seed.

    Parameters
    ----------
    metric : ``metric(**{name: array of shape (n_rings,)}) -> array of
        shape (n_rings,)``.
    params : dict of parameter name -> :class:`BankParam` / :class:`Normal`
        / :class:`Uniform`.
    n_rings : rings per bank trial (>= 1).
    n_trials : number of independent bank trials (>= 1).
    """
    if n_rings < 1:
        raise ValueError("n_rings must be >= 1")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    for name, p in params.items():
        if isinstance(p, (Normal, Uniform, BankParam)) and p.name != name:
            raise ValueError(f"parameter key {name!r} does not match its name {p.name!r}")
    rng = np.random.default_rng(seed)
    draws: dict[str, np.ndarray] = {}
    for name, p in params.items():
        if isinstance(p, BankParam):
            draws[name] = p.sample(n_trials, n_rings, rng)
        else:
            draws[name] = p.sample(n_trials * n_rings, rng).reshape(n_trials, n_rings)
    out = np.empty((n_trials, n_rings))
    for i in range(n_trials):
        corner = {name: x[i] for name, x in draws.items()}
        try:
            result = np.asarray(metric(**corner), dtype=float)
        except ValueError:
            out[i, :] = np.nan
            continue
        if result.shape != (n_rings,):
            raise ValueError(
                f"metric must return an array of shape ({n_rings},); got "
                f"{result.shape} on trial {i}"
            )
        out[i, :] = result
    return BankYieldResult(metric_name=metric_name, samples=out, param_samples=draws)
