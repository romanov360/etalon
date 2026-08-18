"""Parameter extraction: fit Etalon models to measured transmission spectra.

Wraps :func:`scipy.optimize.least_squares` around any Etalon S-parameter
model or :class:`etalon.circuit.Circuit`, so measured wafer-probe spectra
can calibrate model parameters (coupling, loss, effective/group index).

How it works
------------
You supply a ``build(**params)`` callable that constructs the model from a
flat dict of floats, an initial guess ``params0``, and measured data on a
wavelength grid. The fitter minimizes the residual between the model's
named ``inport -> outport`` transmission and the data, in one of two
domains:

* ``"power_db"``  — residual in 10*log10(|S|^2) dB (the natural domain for
  detector/OSA power traces; model power is floored at 1e-16, i.e. -160 dB,
  so exact model nulls stay finite).
* ``"field_mag"`` — residual in |S| (linear field magnitude).

Only magnitude information is fit — measured phase is rarely available from
a power sweep. Parameters that enter only through phase offsets common to
all wavelengths are therefore invisible to the fit.

Honesty limits
--------------
This is architecture-level extraction, not metrology signoff:

* Local optimizer. ``least_squares`` converges to the nearest local
  minimum; resonant spectra (rings, MZIs) are violently multimodal in the
  index parameters, so ``params0`` must place the resonance comb within
  roughly half a free spectral range of the data. :func:`fit_ring_add_drop`
  mitigates this with a coarse scan over one resonance order of ``neff0``.
* Identifiability is the user's problem. Classic degeneracies survive any
  optimizer: a ring's ``neff0`` is only determined modulo the resonance
  order (shifts of wl0/L leave the comb almost unchanged); ``ng`` needs
  more than one FSR in the data; symmetric add-drop rings cannot separate
  kappa1 from kappa2 from the through port alone; loss and coupling trade
  off near critical coupling unless through AND drop are fit jointly.
* No measurement-system model. Fiber-coupling ripple, polarization drift,
  and detector nonlinearity must be de-embedded before fitting.

Units: wavelengths um; power ratios in dB where suffixed ``_db``;
``residual_rms_db`` is in the units of the chosen domain (dB for
``power_db``, dimensionless field magnitude for ``field_mag``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy.optimize import least_squares

from .components import RingAddDrop

# Model power floor when converting to dB: exact transmission nulls (e.g. a
# critically coupled ring on resonance) map to -160 dB instead of -inf.
_POWER_FLOOR = 1e-16

_DOMAINS = ("power_db", "field_mag")


# --- result container -------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    """Outcome of a least-squares parameter extraction.

    Attributes
    ----------
    params : fitted parameter values, same keys as ``params0``.
    cost : final least-squares cost, 0.5 * sum(residual**2) (scipy's
        convention), in the square of the fit domain's units.
    residual_rms_db : root-mean-square residual over all fitted points, in
        the fit domain's units (dB for ``power_db``; despite the ``_db``
        suffix it is linear field magnitude for ``field_mag``). For a good
        fit this approaches the measurement noise floor.
    success : scipy convergence flag (a termination criterion was met).
    nfev : number of model evaluations.
    message : scipy termination message.
    """

    params: dict[str, float]
    cost: float
    residual_rms_db: float
    success: bool
    nfev: int
    message: str

    def report(self) -> str:
        """Multi-line human-readable summary of the fit."""
        lines = [
            f"FitResult: success={self.success}, nfev={self.nfev}",
            f"  cost = {self.cost:.6g}, residual rms = "
            f"{self.residual_rms_db:.4g} (fit-domain units)",
            f"  message: {self.message}",
            "  parameters:",
        ]
        width = max(len(k) for k in self.params)
        for key, value in self.params.items():
            lines.append(f"    {key.ljust(width)} = {value:.8g}")
        return "\n".join(lines)


# --- helpers ----------------------------------------------------------------


def _wl_grid(wl_um) -> np.ndarray:
    wl = np.atleast_1d(np.asarray(wl_um, dtype=float))
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError("wl_um must be a 1-D array with at least 2 points")
    if np.any(wl <= 0):
        raise ValueError("wl_um must be positive (in um)")
    return wl


def _model_transmission(model, wl: np.ndarray, inport: str, outport: str) -> np.ndarray:
    """Complex inport->outport transmission of a component model or Circuit.

    A Circuit is recognized by its ``transmission`` method and
    ``external_ports``; anything else must follow the component protocol
    (``ports`` tuple + ``s_params(wl)``). Raises ValueError for unknown
    port names.
    """
    if hasattr(model, "transmission") and hasattr(model, "external_ports"):
        known = model.external_ports
        for name in (inport, outport):
            if name not in known:
                raise ValueError(
                    f"unknown external port {name!r}; circuit exposes {known}"
                )
        return np.asarray(model.transmission(wl, inport, outport))
    ports = getattr(model, "ports", None)
    if not ports or not hasattr(model, "s_params"):
        raise TypeError(
            "build() must return a Circuit or a model exposing "
            "'ports' and 's_params(wl)'"
        )
    for name in (inport, outport):
        if name not in ports:
            raise ValueError(f"unknown port {name!r}; model ports are {ports}")
    s = np.asarray(model.s_params(wl))
    return s[:, ports.index(outport), ports.index(inport)]


def _to_domain(t: np.ndarray, domain: str) -> np.ndarray:
    """Complex transmission -> fit-domain observable (see module docstring)."""
    if domain == "power_db":
        return 10.0 * np.log10(np.maximum(np.abs(t) ** 2, _POWER_FLOOR))
    return np.abs(t)


def _normalize_paths(
    measured, inport, outport, n_wl: int
) -> list[tuple[str, str, np.ndarray]]:
    """Validate measured data into a list of (inport, outport, array)."""
    if isinstance(measured, Mapping):
        if inport is not None or outport is not None:
            raise ValueError(
                "when measured is a dict {(inport, outport): array}, do not "
                "also pass inport/outport"
            )
        if not measured:
            raise ValueError("measured dict is empty")
        paths = []
        for key, data in measured.items():
            if not (isinstance(key, tuple) and len(key) == 2):
                raise ValueError(
                    f"measured dict keys must be (inport, outport) tuples; got {key!r}"
                )
            paths.append((key[0], key[1], np.asarray(data, dtype=float)))
    else:
        if inport is None or outport is None:
            raise ValueError(
                "inport and outport are required when measured is a single array"
            )
        paths = [(inport, outport, np.asarray(measured, dtype=float))]
    for pin, pout, data in paths:
        if data.shape != (n_wl,):
            raise ValueError(
                f"measured data for ({pin!r}, {pout!r}) has shape {data.shape}; "
                f"expected ({n_wl},) matching wl_um"
            )
        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"measured data for ({pin!r}, {pout!r}) contains non-finite values"
            )
    return paths


def _pack_bounds(
    names: list[str],
    params0: dict[str, float],
    bounds: dict[str, tuple[float, float]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds dict -> (lb, ub) vectors in ``names`` order, validated."""
    lb = np.full(len(names), -np.inf)
    ub = np.full(len(names), np.inf)
    if bounds is None:
        return lb, ub
    unknown = set(bounds) - set(names)
    if unknown:
        raise ValueError(
            f"bounds given for parameters not in params0: {sorted(unknown)}"
        )
    for i, name in enumerate(names):
        if name not in bounds:
            continue
        lo, hi = bounds[name]
        if not lo < hi:
            raise ValueError(f"bounds for {name!r} must satisfy lo < hi; got ({lo}, {hi})")
        if not lo <= params0[name] <= hi:
            raise ValueError(
                f"params0[{name!r}] = {params0[name]} lies outside bounds ({lo}, {hi})"
            )
        lb[i], ub[i] = lo, hi
    return lb, ub


# --- generic fitter ----------------------------------------------------------


def fit_transmission(
    build: Callable[..., object],
    params0: dict[str, float],
    wl_um,
    measured,
    *,
    inport: str | None = None,
    outport: str | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    domain: str = "power_db",
    **fixed: float,
) -> FitResult:
    """Fit a model's free parameters to measured transmission spectra.

    Minimizes sum over wavelengths (and paths) of
    ``(model_observable - measured)**2`` with
    :func:`scipy.optimize.least_squares` (trust-region reflective), where
    the observable is ``10*log10(|S_out,in|^2)`` for ``domain="power_db"``
    or ``|S_out,in|`` for ``domain="field_mag"``.

    Parameters
    ----------
    build : callable; ``build(**{**fixed, **params})`` must return either a
        component model (``ports`` + ``s_params(wl)``) or a
        :class:`etalon.circuit.Circuit`.
    params0 : initial guess for the FREE parameters, dict of floats. Key
        order sets the internal parameter vector order.
    wl_um : 1-D wavelength grid in um (>= 2 points).
    measured : either a single array matching ``wl_um`` (then ``inport``
        and ``outport`` are required), or a dict
        ``{(inport, outport): array}`` to fit several paths jointly (then
        ``inport``/``outport`` must be omitted). Values are in the chosen
        domain's units (dB for ``power_db``, |S| for ``field_mag``).
    inport, outport : port names for the single-array form. For a
        component these are entries of ``model.ports``; for a Circuit,
        exposed external port names.
    bounds : optional ``{name: (lo, hi)}``; keys must be a subset of
        ``params0``; missing names are unbounded. ``params0`` must lie
        within its bounds. Domain-constrained parameters (couplings,
        power fractions, losses) SHOULD always be bounded: unbounded, the
        optimizer may step outside the model's validity domain, and the
        fit stops with a ValueError naming the offending trial values.
    domain : ``"power_db"`` (default) or ``"field_mag"``.
    **fixed : extra keyword arguments passed to ``build`` unchanged (fixed,
        not fitted). Keys must not collide with ``params0``.

    Returns
    -------
    FitResult with fitted params (same keys as ``params0``).

    Raises
    ------
    ValueError on empty/non-finite ``params0``, bad domain, shape
    mismatches, unknown ports, bounds/params0/fixed key mismatches.

    Notes
    -----
    Local optimization only — see the module docstring for the honesty
    limits (multimodality of resonant spectra, identifiability).
    """
    if domain not in _DOMAINS:
        raise ValueError(f"domain must be one of {_DOMAINS}; got {domain!r}")
    if not params0:
        raise ValueError("params0 must contain at least one free parameter")
    for key, value in params0.items():
        if not np.isfinite(value):
            raise ValueError(f"params0[{key!r}] = {value} is not finite")
    overlap = set(params0) & set(fixed)
    if overlap:
        raise ValueError(
            f"parameters appear in both params0 and fixed kwargs: {sorted(overlap)}"
        )

    wl = _wl_grid(wl_um)
    paths = _normalize_paths(measured, inport, outport, wl.size)
    names = list(params0)
    x0 = np.array([params0[k] for k in names], dtype=float)
    lb, ub = _pack_bounds(names, params0, bounds)

    # Fail fast (with a clean ValueError) on unknown ports / bad build.
    model0 = build(**{**fixed, **params0})
    for pin, pout, _ in paths:
        _model_transmission(model0, wl, pin, pout)

    def residual(x: np.ndarray) -> np.ndarray:
        params = dict(zip(names, x))
        try:
            model = build(**{**fixed, **params})
        except ValueError as exc:
            raise ValueError(
                f"build() rejected trial parameters {params} during the "
                f"optimization ({exc}). Domain-constrained parameters "
                f"(couplings, power fractions, losses) need explicit "
                f"bounds= so the optimizer cannot step outside the model's "
                f"validity domain."
            ) from exc
        parts = [
            _to_domain(_model_transmission(model, wl, pin, pout), domain) - data
            for pin, pout, data in paths
        ]
        return np.concatenate(parts)

    # x_scale="jac": rescale parameters by Jacobian column norms, so kappas
    # (~1e-2) and indices (~1) converge together instead of stalling on xtol.
    res = least_squares(residual, x0, bounds=(lb, ub), method="trf", x_scale="jac")
    rms = float(np.sqrt(np.mean(res.fun**2)))
    return FitResult(
        params={k: float(v) for k, v in zip(names, res.x)},
        cost=float(res.cost),
        residual_rms_db=rms,
        success=bool(res.success),
        nfev=int(res.nfev),
        message=str(res.message),
    )


# --- ring add-drop convenience ------------------------------------------------


_RING_FREE = ("kappa1_power", "kappa2_power", "loss_db_per_cm", "neff0", "ng")

_RING_X0 = {
    "kappa1_power": 0.1,
    "kappa2_power": 0.1,
    "loss_db_per_cm": 2.0,
    "neff0": 2.4,
    "ng": 4.2,
}

_RING_BOUNDS = {
    "kappa1_power": (1e-4, 0.9),
    "kappa2_power": (1e-4, 0.9),
    "loss_db_per_cm": (0.0, 1e3),
    "neff0": (1.0, 4.5),
    "ng": (1.0, 8.0),
}


def fit_ring_add_drop(
    wl_um,
    through_db,
    drop_db,
    x0: dict | None = None,
    *,
    circumference_um: float,
    wl0_um: float = 1.55,
    bounds: dict[str, tuple[float, float]] | None = None,
    fixed: dict[str, float] | None = None,
) -> FitResult:
    """Fit a :class:`etalon.components.RingAddDrop` to through+drop spectra.

    Jointly fits (kappa1_power, kappa2_power, loss_db_per_cm, neff0, ng)
    to the measured in->through and in->drop power spectra in dB, via
    :func:`fit_transmission`. The joint fit is what breaks the
    loss-vs-coupling degeneracy: through-port extinction and drop-port
    insertion loss constrain different combinations of (kappa1, kappa2, a).

    Before the least-squares run, ``neff0`` is coarsely scanned over one
    resonance-order spacing (wl0/L, 61 samples) about its initial guess and
    the best offset is kept, so the resonance comb starts aligned with the
    data — this is what makes the local optimizer land in the right basin.

    Parameters
    ----------
    wl_um : 1-D wavelength grid in um; SHOULD span several FSRs (see below).
    through_db, drop_db : measured power transmission in dB, same shape as
        ``wl_um``.
    x0 : optional initial-guess overrides for any of the five free
        parameters (unknown keys raise ValueError). Defaults: kappas 0.1,
        loss 2 dB/cm, neff0 2.4, ng 4.2 — supply design values whenever
        you have them; the coarse neff0 scan only searches one resonance
        order around the guess.
    circumference_um : ring round-trip length in um, FIXED from layout.
        It is not fittable: only the products neff0*L and ng*L enter the
        response, so L and the indices are exactly degenerate.
    wl0_um : dispersion expansion wavelength passed to RingAddDrop; pick it
        inside the measured span.
    bounds : optional per-parameter overrides of the defaults
        (kappas in (1e-4, 0.9), loss in (0, 1000) dB/cm, neff0 in
        (1.0, 4.5), ng in (1.0, 8.0)).
    fixed : optional ``{name: value}`` holding any of the five parameters
        at a known value instead of fitting it — the standard remedy for
        the (kappa1, kappa2, loss) degeneracy when loss is known from a
        de-embedding structure. Fixed names must not also appear in ``x0``
        or ``bounds``.

    Returns
    -------
    FitResult over the five free parameters; ``residual_rms_db`` is the rms
    misfit in dB over both spectra.

    Identifiability — read before trusting the numbers
    --------------------------------------------------
    * ``neff0`` is identifiable only modulo the resonance order: adding any
      integer multiple of ~wl0/L to neff0 shifts the comb by whole orders
      and reproduces (almost) the same spectrum. The returned value is the
      one nearest the initial guess; report it as "neff0 assuming order m".
    * ``ng`` is constrained by the FSR, so the data must contain more than
      one FSR; spanning 3+ FSRs is recommended (with a single resonance in
      view, ng returns the initial guess dressed in noise).
    * ``kappa1``/``kappa2``/``loss``: a correlated (kappa1 down, kappa2 up,
      loss up) direction changes the magnitude spectra by only ~0.1 dB per
      ~10% parameter shift near the through-port dip — flat at typical
      probe-noise levels (0.05 dB rms), so several parameter sets fit the
      data equally well. The optimizer returns the valley point nearest
      ``x0``: start from the intended DESIGN asymmetry (kappa1 vs kappa2)
      and treat the split with ~10% uncertainty unless the noise floor is
      well below 0.05 dB or phase/group-delay data is added. Pinning a
      known loss via ``fixed={"loss_db_per_cm": ...}`` (e.g. from a
      de-embedding structure) collapses the valley.
    * Coupler excess loss is not modeled; it is absorbed into
      loss_db_per_cm.
    """
    wl = _wl_grid(wl_um)
    if circumference_um <= 0:
        raise ValueError("circumference_um must be positive")

    held = {k: float(v) for k, v in (fixed or {}).items()}
    unknown = set(held) - set(_RING_FREE)
    if unknown:
        raise ValueError(
            f"fixed keys must be among {_RING_FREE}; unknown: {sorted(unknown)}"
        )
    free = tuple(k for k in _RING_FREE if k not in held)
    if not free:
        raise ValueError("at least one of the five ring parameters must stay free")

    params0 = {k: _RING_X0[k] for k in free}
    if x0:
        unknown = set(x0) - set(_RING_FREE)
        if unknown:
            raise ValueError(
                f"x0 keys must be among {_RING_FREE}; unknown: {sorted(unknown)}"
            )
        overlap = set(x0) & set(held)
        if overlap:
            raise ValueError(f"parameters appear in both x0 and fixed: {sorted(overlap)}")
        params0.update({k: float(v) for k, v in x0.items()})

    eff_bounds = {k: _RING_BOUNDS[k] for k in free}
    if bounds:
        unknown = set(bounds) - set(_RING_FREE)
        if unknown:
            raise ValueError(
                f"bounds keys must be among {_RING_FREE}; unknown: {sorted(unknown)}"
            )
        overlap = set(bounds) & set(held)
        if overlap:
            raise ValueError(
                f"parameters appear in both bounds and fixed: {sorted(overlap)}"
            )
        eff_bounds.update(bounds)

    measured = {
        ("in", "through"): np.asarray(through_db, dtype=float),
        ("in", "drop"): np.asarray(drop_db, dtype=float),
    }

    def build(**p) -> RingAddDrop:
        return RingAddDrop(**p)

    build_kwargs = {"circumference_um": circumference_um, "wl0_um": wl0_um, **held}

    # Coarse neff0 scan over one resonance-order spacing to align the comb
    # (skipped when neff0 is held fixed).
    if "neff0" in free:
        paths = _normalize_paths(measured, None, None, wl.size)
        order_spacing = wl0_um / circumference_um
        lo, hi = eff_bounds["neff0"]
        best_neff0, best_sse = params0["neff0"], np.inf
        for offset in np.linspace(-0.5, 0.5, 61) * order_spacing:
            neff0 = params0["neff0"] + offset
            if not lo <= neff0 <= hi:
                continue
            model = build(**{**build_kwargs, **params0, "neff0": neff0})
            sse = 0.0
            for pin, pout, data in paths:
                obs = _to_domain(_model_transmission(model, wl, pin, pout), "power_db")
                sse += float(np.sum((obs - data) ** 2))
            if sse < best_sse:
                best_neff0, best_sse = neff0, sse
        params0["neff0"] = best_neff0

    # build_kwargs's keys are circumference_um/wl0_um plus `held`, which is
    # validated above (unknown check) to be a subset of _RING_FREE — never
    # "bounds"/"domain"/"inport"/"outport" — so this can't actually collide
    # with fit_transmission's named parameters; mypy can't see across that
    # runtime validation.
    return fit_transmission(
        build,
        params0,
        wl,
        measured,
        bounds=eff_bounds,
        domain="power_db",
        **build_kwargs,  # type: ignore[arg-type]
    )
