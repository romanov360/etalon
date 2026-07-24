"""Laser reliability and thermal derating for CPO-class optical engines.

Lasers are the dominant field-failure mechanism in co-packaged optics, and
their efficiency degrades with junction temperature. This module turns the
quotable reliability facts into reproducible closed-form arithmetic:

* Arrhenius thermal acceleration between a use and a stress temperature
  (the standard model behind laser life-test extrapolation).
* FIT <-> MTTF algebra for the constant-hazard (random-failure) regime,
  JEDEC JESD85-style: FIT = failures per 1e9 device-hours, series systems
  sum FITs.
* Lognormal wear-out, the conventional distribution for laser-diode
  end-of-life (median life t50, shape sigma).
* Competing-risks composition of the two regimes and an all-lasers-alive
  module survival probability.
* First-order linear wall-plug-efficiency derating with temperature, to
  feed :class:`siphon.link.Laser`.

Conventions
-----------
Temperatures in degrees Celsius at the interface, converted to kelvin
internally; times in hours; FIT in failures per 1e9 device-hours;
activation energies in eV; probabilities dimensionless in [0, 1].

Validity: architecture-level actuarial arithmetic, not qualification
signoff. Random and wear-out failures are treated as independent competing
risks with time-independent parameters; there is no infant mortality
(early-life/burn-in) term, no thermal self-heating loop (junction
temperature is an input, not a solved quantity), and activation energies /
t50 / sigma must come from actual life-test data (e.g. Telcordia GR-468
style qualification), not from this module.
"""

from __future__ import annotations

import math

# Boltzmann constant in eV/K, the natural unit for Arrhenius activation
# energies (CODATA: 8.617333262e-5 eV/K).
BOLTZMANN_EV_PER_K = 8.617333262e-5

_ZERO_C_IN_K = 273.15


def _celsius_to_kelvin(t_c: float, name: str) -> float:
    t_k = t_c + _ZERO_C_IN_K
    if t_k <= 0.0:
        raise ValueError(f"{name} must be above absolute zero (-273.15 C)")
    return t_k


# --- Arrhenius acceleration --------------------------------------------------


def arrhenius_acceleration(
    t_use_c: float, t_stress_c: float, activation_energy_ev: float
) -> float:
    """Arrhenius thermal acceleration factor between use and stress temperature.

    The standard model for thermally activated failure mechanisms
    (JEDEC JESD91/JESD85 usage; ubiquitous in laser-diode life testing):

        AF = exp( (Ea / k) * (1/T_use - 1/T_stress) )

    with temperatures in kelvin (converted from the Celsius inputs) and
    k = 8.617333262e-5 eV/K. AF > 1 when t_stress_c > t_use_c: time at the
    stress temperature ages the device AF times faster than at the use
    temperature (equivalently, stress-test hours multiply by AF when
    extrapolated to use conditions). AF < 1 when the "stress" temperature
    is below the use temperature; swapping the two temperatures gives the
    reciprocal factor (AF -> 1/AF — the exponent is antisymmetric in 1/T).

    Example anchor: Ea = 0.97 eV (a common laser-diode value), 25 C -> 60 C
    gives AF ~ 52.8.

    Validity: single dominant thermally activated mechanism with a
    temperature-independent Ea; no humidity/current/optical-power stress
    terms (no Peck or inverse-power-law factors).
    """
    if activation_energy_ev <= 0.0:
        raise ValueError("activation_energy_ev must be positive")
    t_use_k = _celsius_to_kelvin(t_use_c, "t_use_c")
    t_stress_k = _celsius_to_kelvin(t_stress_c, "t_stress_c")
    return math.exp(
        (activation_energy_ev / BOLTZMANN_EV_PER_K) * (1.0 / t_use_k - 1.0 / t_stress_k)
    )


# --- FIT <-> MTTF (constant-hazard / random-failure regime) -------------------


def fit_from_mttf(mttf_hours: float) -> float:
    """FIT rate from MTTF: FIT = 1e9 / MTTF_hours.

    Valid in the exponential (constant-hazard, random-failure) regime,
    where the failure rate is time-independent and MTTF = 1/lambda. One
    FIT is one failure per 1e9 cumulative device-hours (JEDEC JESD85).
    Does NOT describe wear-out; see :func:`wearout_fraction`.
    """
    if mttf_hours <= 0.0:
        raise ValueError("mttf_hours must be positive")
    return 1e9 / mttf_hours


def mttf_from_fit(fit: float) -> float:
    """MTTF in hours from a FIT rate: MTTF = 1e9 / FIT (inverse of
    :func:`fit_from_mttf`; same exponential-regime caveat)."""
    if fit <= 0.0:
        raise ValueError("fit must be positive")
    return 1e9 / fit


def module_fit(fit_per_device: float, n_devices: int) -> float:
    """FIT of a series system of n identical independent devices.

    In the constant-hazard regime failure rates of a series system
    (any-one-fails = module fails) add: FIT_module = n * FIT_device
    (JEDEC JESD85 system FIT algebra). This is the brutal arithmetic of
    CPO: a package with 64 lasers has 64x the laser FIT of a pluggable
    with one.
    """
    if n_devices < 1:
        raise ValueError("n_devices must be >= 1")
    if fit_per_device < 0.0:
        raise ValueError("fit_per_device must be >= 0")
    return fit_per_device * n_devices


def module_mttf_hours(fit_per_device: float, n_devices: int) -> float:
    """MTTF in hours of a series system: 1e9 / (n * FIT_device).

    Equals the single-device MTTF divided by n (constant-hazard regime).
    """
    total_fit = module_fit(fit_per_device, n_devices)
    if total_fit <= 0.0:
        raise ValueError("fit_per_device must be positive for a finite MTTF")
    return 1e9 / total_fit


def survival_probability(t_hours: float, fit: float) -> float:
    """Probability a device survives t hours of random failures.

    Exponential reliability function S(t) = exp(-lambda t) with
    lambda = FIT * 1e-9 per hour. Constant-hazard regime only; compose
    with :func:`wearout_fraction` as competing risks for a full picture.
    """
    if t_hours < 0.0:
        raise ValueError("t_hours must be >= 0")
    if fit < 0.0:
        raise ValueError("fit must be >= 0")
    return math.exp(-fit * t_hours * 1e-9)


# --- lognormal wear-out --------------------------------------------------------


def wearout_fraction(t_hours: float, t50_hours: float, sigma: float) -> float:
    """Cumulative wear-out failure fraction under a lognormal life model.

    F(t) = Phi( ln(t / t50) / sigma ), with Phi the standard normal CDF
    (evaluated via math.erf), t50 the median life in hours and sigma the
    lognormal shape parameter. The lognormal is the conventional wear-out
    distribution for laser diodes (Telcordia GR-468-style qualification;
    Fukuda, *Reliability and Degradation of Semiconductor Lasers and
    LEDs*): degradation proceeds multiplicatively, so log-life is normal.

    Distinct regimes: wear-out (this function, hazard rising with time) is
    NOT the same failure mode as the constant-hazard FIT regime
    (:func:`survival_probability`). They compose as independent competing
    risks:

        S_total(t) = S_random(t) * (1 - wearout_fraction(t)).

    Returns 0.0 at t = 0 (the lognormal limit) and exactly 0.5 at t = t50.
    """
    if t_hours < 0.0:
        raise ValueError("t_hours must be >= 0")
    if t50_hours <= 0.0:
        raise ValueError("t50_hours must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if t_hours == 0.0:
        return 0.0
    z = math.log(t_hours / t50_hours) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def module_survival(
    t_hours: float,
    n_lasers: int,
    fit_per_laser: float,
    t50_hours: float | None = None,
    sigma: float | None = None,
) -> float:
    """Probability that ALL of n independent identical lasers survive t hours.

    Per laser, random (FIT) and lognormal wear-out failures compose as
    independent competing risks:

        S_laser(t) = exp(-FIT * t * 1e-9) * (1 - wearout_fraction(t)),

    and with independent identical lasers the all-alive module probability
    is S_laser(t) ** n. Pass t50_hours and sigma together to include
    wear-out, or leave both None for the FIT-only (constant-hazard) model.

    This is the no-sparing, no-repair number; laser redundancy or field
    repair would need a k-of-n binomial on top, which is out of scope here.
    """
    if n_lasers < 1:
        raise ValueError("n_lasers must be >= 1")
    if (t50_hours is None) != (sigma is None):
        raise ValueError("t50_hours and sigma must be given together (or both None)")
    s_laser = survival_probability(t_hours, fit_per_laser)
    if t50_hours is not None:
        s_laser *= 1.0 - wearout_fraction(t_hours, t50_hours, sigma)
    return s_laser**n_lasers


# --- thermal derating -----------------------------------------------------------


def wpe_derated(wpe_ref: float, t_ref_c: float, t_c: float, slope_per_k: float) -> float:
    """Wall-plug efficiency at temperature t_c, first-order linear derating.

        wpe(T) = wpe_ref * (1 - slope_per_k * (t_c - t_ref_c)),

    with wpe_ref the efficiency at the reference temperature t_ref_c (both
    Celsius) and slope_per_k >= 0 the fractional efficiency loss per kelvin
    (typical DFB/laser-engine values are of order 0.005-0.02 /K near room
    temperature, from the T0/T1 characteristic-temperature roll-off of
    threshold and slope efficiency linearized about t_ref_c). Below the
    reference temperature the efficiency is allowed to exceed wpe_ref but
    is capped at the physical bound of 1.0. Raises ValueError if the linear
    model derates the efficiency to <= 0 — that is outside the model's
    validity range, not a prediction.

    Feed the result into :class:`siphon.link.Laser` as
    ``Laser(wpe=wpe_derated(...), ...)``. This module deliberately does NOT
    solve the electro-thermal self-heating loop (wpe depends on T_junction,
    which depends on dissipation, which depends on wpe): the junction /
    case temperature is an input the caller must supply from a thermal
    model or measurement.
    """
    if not 0.0 < wpe_ref <= 1.0:
        raise ValueError("wpe_ref must be in (0, 1]")
    if slope_per_k < 0.0:
        raise ValueError("slope_per_k must be >= 0")
    _celsius_to_kelvin(t_ref_c, "t_ref_c")
    _celsius_to_kelvin(t_c, "t_c")
    wpe = wpe_ref * (1.0 - slope_per_k * (t_c - t_ref_c))
    if wpe <= 0.0:
        raise ValueError(
            f"linear derating gives wpe <= 0 at t_c = {t_c} C; "
            "outside the first-order model's validity range"
        )
    return min(wpe, 1.0)
