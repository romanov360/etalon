"""Optical link-budget engine for IM-DD datacom interconnects.

Closed-form power and energy budgets for intensity-modulated direct-detect
links (NRZ and PAM-M), the arithmetic a co-packaged-optics system architect
runs before anyone opens a circuit simulator.

Conventions
-----------
* Optical power in dBm (mW where named), losses/penalties in dB, symbol
  rates in GBd, energies in fJ/bit or pJ/bit as named.
* Modulation amplitude is tracked as *outer* OMA (optical modulation
  amplitude, P_top - P_bottom); receiver sensitivity is likewise an OMA.
* The receiver baseline is thermal-noise-limited: the TIA input-referred
  current noise sets the sensitivity floor. Shot noise (including dark
  current) is modeled on top as a self-consistent dB penalty evaluated at
  the top eye level (see :func:`shot_penalty_db`); ISI is neglected.
  RIN, shot noise, finite extinction ratio, and crosstalk are layered on
  as separate dB penalties, following standard budgeting practice
  (Agrawal, *Fiber-Optic Communication Systems*, 4th ed., ch. 4; IEEE
  802.3 link-budget spreadsheets). Because RIN and shot noise are both
  signal-dependent, their separate dB penalties sum OPTIMISTICALLY (the
  classic "adding dB penalties is conservative" argument holds only for
  signal-independent noises); ``LinkBudget`` solves the joint thermal +
  RIN + shot requirement exactly and reports the correction as a separate
  ``noise_interaction_db`` waterfall row.

Validity: architecture-level numbers, not compliance signoff. There is no
TDECQ machinery and no bandwidth/equalization modeling; equalization and
ISI effects must be entered as explicit penalties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.special import erfc, erfcinv

from .constants import ELEMENTARY_CHARGE_C, db_to_linear, dbm_to_mw, mw_to_dbm

# Noise bandwidth of an optimized baud-rate receiver, as a fraction of the
# symbol rate. 0.75 * baud is the conventional value for a receiver with
# ~0.5-0.7 * baud 3-dB bandwidth (e.g. IEEE 802.3 sensitivity math).
NOISE_BANDWIDTH_FRACTION = 0.75


# --- BER <-> Q ------------------------------------------------------------


def q_from_ber(target_ber: float, levels: int) -> float:
    """Per-eye Q factor required for a target pre-FEC BER with PAM-M.

    Uses the equal-eye PAM-M symbol error rate SER = 2 (1 - 1/M) Q(q),
    Q(x) = erfc(x / sqrt 2) / 2 the Gaussian tail, together with the
    Gray-coding approximation BER = SER / log2(M) (one bit flip per symbol
    error; standard and accurate at low error rates).
    """
    _check_levels(levels)
    if not 0.0 < target_ber < 0.5:
        raise ValueError("target_ber must be in (0, 0.5)")
    ser = target_ber * math.log2(levels)
    tail = ser / (2.0 * (1.0 - 1.0 / levels))
    if tail >= 0.5:
        raise ValueError("target BER is above the PAM-M random-guess floor")
    return float(math.sqrt(2.0) * erfcinv(2.0 * tail))


def ber_from_q(q: float, levels: int) -> float:
    """Pre-FEC BER of PAM-M at per-eye Q factor q (inverse of q_from_ber)."""
    _check_levels(levels)
    ser = 2.0 * (1.0 - 1.0 / levels) * 0.5 * erfc(q / math.sqrt(2.0))
    return float(ser / math.log2(levels))


def _check_levels(levels: int) -> None:
    if levels < 2 or 2 ** int(round(math.log2(levels))) != levels:
        raise ValueError("levels must be a power of two >= 2 (2=NRZ, 4=PAM4, ...)")


# --- components -----------------------------------------------------------


@dataclass(frozen=True)
class Laser:
    """CW laser line.

    Parameters
    ----------
    power_dbm : optical power of one wavelength line, in dBm, referenced at
        the plane named by ``launch_reference``.
    wpe : wall-plug efficiency, optical out / electrical in, 0 < wpe <= 1.
        Include TEC/driver overhead here if it should count against energy.
    rin_db_hz : relative intensity noise in dB/Hz (e.g. -145).
    launch_reference : documentation of the reference plane for
        ``power_dbm``, e.g. "fiber" (in-fiber, external laser source) or
        "chip" (on/into-chip, integrated or butt-coupled laser). Coupling
        losses downstream of this plane belong in the link path.
    """

    power_dbm: float
    wpe: float
    rin_db_hz: float
    launch_reference: str = "fiber"

    def __post_init__(self):
        if not 0.0 < self.wpe <= 1.0:
            raise ValueError("wpe must be in (0, 1]")


@dataclass(frozen=True)
class Modulator:
    """Intensity modulator (MZM, EAM, or ring) for OOK/PAM.

    ``insertion_loss_db`` is the on-state (peak-level) loss;
    ``extinction_ratio_db`` the outer extinction ratio P_top/P_bottom.
    ``energy_fj_per_bit`` covers modulator + driver switching energy.
    """

    insertion_loss_db: float
    extinction_ratio_db: float
    energy_fj_per_bit: float

    def __post_init__(self):
        if self.extinction_ratio_db <= 0.0:
            raise ValueError("extinction_ratio_db must be > 0 dB")

    @property
    def modulation_loss_db(self) -> float:
        """Average-power penalty of OOK/PAM modulation, in dB.

        For M equally spaced levels between P_bot and P_top the average
        power is (P_top + P_bot)/2, so relative to the CW (peak) power
        P_top the loss is -10 log10((1 + 1/ER)/2), independent of M.
        3.01 dB at infinite ER.
        """
        er = db_to_linear(self.extinction_ratio_db)
        if math.isinf(er):
            return 10.0 * math.log10(2.0)
        return -10.0 * math.log10((1.0 + 1.0 / er) / 2.0)


@dataclass(frozen=True)
class Photodiode:
    """PIN photodiode.

    ``responsivity_a_per_w`` in A/W. ``dark_current_na`` (nA) contributes
    shot noise 2 q_e I_d f_n in :func:`shot_penalty_db`; it is usually
    negligible against TIA noise but is carried through exactly.
    """

    responsivity_a_per_w: float
    dark_current_na: float = 10.0

    def __post_init__(self):
        if self.responsivity_a_per_w <= 0.0:
            raise ValueError("responsivity must be positive")


@dataclass(frozen=True)
class Tia:
    """Transimpedance amplifier front end.

    ``input_noise_pa_per_sqrt_hz`` is the input-referred current noise
    density (flat approximation), ``bandwidth_ghz`` the 3-dB bandwidth,
    ``energy_fj_per_bit`` the RX analog front-end energy.

    CAUTION: ``bandwidth_ghz`` is descriptive only — no sensitivity
    formula consumes it. The noise bandwidth is always
    0.75 * baud (NOISE_BANDWIDTH_FRACTION), so budgets computed at symbol
    rates well beyond the TIA's stated bandwidth silently assume a
    receiver rescaled to that baud (there is no bandwidth/ISI modeling).
    """

    input_noise_pa_per_sqrt_hz: float
    bandwidth_ghz: float
    energy_fj_per_bit: float

    def __post_init__(self):
        if self.input_noise_pa_per_sqrt_hz <= 0.0 or self.bandwidth_ghz <= 0.0:
            raise ValueError("noise density and bandwidth must be positive")


@dataclass(frozen=True)
class Signaling:
    """Line signaling: symbol rate, PAM order, and target pre-FEC BER.

    Default target_ber = 2.4e-4 is the KP4 RS(544,514) FEC threshold
    (IEEE 802.3bj); levels = 4 is PAM4, 2 is NRZ.
    """

    rate_gbd: float
    levels: int = 4
    target_ber: float = 2.4e-4

    def __post_init__(self):
        if self.rate_gbd <= 0.0:
            raise ValueError("rate_gbd must be positive")
        _check_levels(self.levels)
        if not 0.0 < self.target_ber < 0.5:
            raise ValueError("target_ber must be in (0, 0.5)")

    @property
    def bits_per_symbol(self) -> float:
        return math.log2(self.levels)

    @property
    def bit_rate_gbps(self) -> float:
        """Line bit rate in Gb/s (baud * log2(M))."""
        return self.rate_gbd * self.bits_per_symbol

    @property
    def q_factor(self) -> float:
        """Per-eye Q required for target_ber (see :func:`q_from_ber`)."""
        return q_from_ber(self.target_ber, self.levels)


# --- sensitivity and penalties ---------------------------------------------


def receiver_sensitivity_oma_dbm(pd: Photodiode, tia: Tia, sig: Signaling) -> float:
    """Thermal-noise-limited receiver sensitivity as outer OMA, in dBm.

    With M equally spaced levels, eye spacing OMA/(M-1), Gaussian TIA noise
    of rms i_n at each level and the decision thresholds mid-eye, hitting
    per-eye Q = q requires

        OMA_outer = 2 q (M-1) i_n / R,      i_n = S_i sqrt(f_n),

    with responsivity R, input noise density S_i, and noise bandwidth
    f_n = 0.75 * baud (see NOISE_BANDWIDTH_FRACTION). Assumes no shot
    noise, no signal-dependent noise, and no ISI; RIN, extinction, and
    crosstalk are separate penalties (see LinkBudget.sensitivity_oma_dbm).
    """
    q = sig.q_factor
    f_n_hz = NOISE_BANDWIDTH_FRACTION * sig.rate_gbd * 1e9
    i_n_a = tia.input_noise_pa_per_sqrt_hz * 1e-12 * math.sqrt(f_n_hz)
    oma_w = 2.0 * q * (sig.levels - 1) * i_n_a / pd.responsivity_a_per_w
    return float(mw_to_dbm(oma_w * 1e3))


def er_penalty_db(er_db: float) -> float:
    """Extinction-ratio power penalty, dB: 10 log10((ER+1)/(ER-1)).

    The classic penalty relating average power to eye opening for finite
    outer extinction ratio ER (linear); 0 dB at infinite ER (Agrawal,
    4th ed., eq. 4.6.4). Applied on the sensitivity side of the budget as
    the conventional, conservative allocation for extinction-related eye
    closure (residual "0"-level power, threshold offset).
    """
    if er_db <= 0.0:
        raise ValueError("extinction ratio must be > 0 dB")
    if math.isinf(er_db):
        return 0.0
    er = db_to_linear(er_db)
    return 10.0 * math.log10((er + 1.0) / (er - 1.0))


def rin_penalty_db(
    rin_db_hz: float,
    rate_gbd: float,
    q: float,
    levels: int,
    extinction_ratio_db: float = math.inf,
) -> float:
    """Laser RIN power penalty, dB, for a thermal-limited PAM-M receiver.

    Relative intensity noise is evaluated at the top level, whose power at
    outer extinction ratio ER (linear) is P_top = OMA * ER/(ER-1) — equal
    to OMA only at infinite ER. With rms relative noise
    r = sqrt(RIN * f_n), f_n = 0.75 * baud, requiring the top eye
    (spacing OMA/(M-1)) to still reach Q = q against the combined
    thermal + RIN noise inflates the needed OMA by

        penalty = -5 log10(1 - (2 q r' (M-1))^2),    r' = r * ER/(ER-1)

    (same structure as Agrawal 4th ed., eq. 4.7.3, generalized to PAM-M
    with the noise evaluated at the top level). At the low ERs real
    modulators run (4-6 dB) the ER/(ER-1) factor is 1.4-1.7x and roughly
    doubles the penalty versus evaluating at OMA; the default infinite ER
    reproduces the classic P_top ~ OMA form. Raises ValueError when
    2 q r' (M-1) >= 1: RIN alone then makes the target BER unreachable at
    any power.
    """
    _check_levels(levels)
    if extinction_ratio_db <= 0.0:
        raise ValueError("extinction ratio must be > 0 dB")
    er = db_to_linear(extinction_ratio_db)
    top_over_oma = 1.0 if math.isinf(er) else er / (er - 1.0)
    f_n_hz = NOISE_BANDWIDTH_FRACTION * rate_gbd * 1e9
    r = math.sqrt(db_to_linear(rin_db_hz) * f_n_hz) * top_over_oma
    x = 2.0 * q * r * (levels - 1)
    if x >= 1.0:
        raise ValueError(
            f"RIN {rin_db_hz} dB/Hz makes target unreachable at {rate_gbd} GBd "
            f"PAM{levels} (2*q*r*(M-1) = {x:.2f} >= 1)"
        )
    return -5.0 * math.log10(1.0 - x * x)


def shot_penalty_db(
    pd: Photodiode,
    tia: Tia,
    sig: Signaling,
    extinction_ratio_db: float = math.inf,
) -> float:
    """Shot-noise (and dark-current) power penalty, dB, on the thermal budget.

    The thermal-limited baseline requires an outer OMA (watts) of
    u0 = 2 q (M-1) i_n / R, with i_n = S_i sqrt(f_n) the rms TIA current
    noise and f_n = 0.75 * baud (NOISE_BANDWIDTH_FRACTION). Photocurrent
    shot noise is signal-dependent, so it is evaluated self-consistently
    at the sensitivity point, at the top eye level — whose power at outer
    extinction ratio ER (linear) is P_top = OMA * k, k = ER/(ER-1)
    (k = 1 at infinite ER, the same top-eye convention as
    :func:`rin_penalty_db`). With elementary charge q_e and dark current
    I_d (A), the top eye reaches Q = q when the required OMA u satisfies

        u R / (2 (M-1)) = q sqrt( i_n^2 + 2 q_e (R k u + I_d) f_n ),

    quadratic in u:  a u^2 - b u - c = 0 with

        a = (R / (2 q (M-1)))^2,   b = 2 q_e R k f_n,
        c = i_n^2 + 2 q_e I_d f_n,
        u = (b + sqrt(b^2 + 4 a c)) / (2 a),

    and the penalty is 10 log10(u / u0) >= 0 dB. This is the SINGLE-noise
    penalty (thermal + shot only). Because shot and RIN are both
    signal-dependent, adding their separate dB penalties UNDERSTATES the
    joint requirement (each solve undercounts the variance at the larger
    joint operating point); ``LinkBudget`` therefore re-sums the variances
    jointly and books the positive difference as
    ``noise_interaction_db``. Raises ValueError for extinction ratio
    <= 0 dB. Assumes a PIN photodiode (no avalanche excess noise) and
    Gaussian statistics; architecture-level accuracy, not signoff.
    """
    if extinction_ratio_db <= 0.0:
        raise ValueError("extinction ratio must be > 0 dB")
    er = db_to_linear(extinction_ratio_db)
    k = 1.0 if math.isinf(er) else er / (er - 1.0)
    q = sig.q_factor
    m = sig.levels
    r = pd.responsivity_a_per_w
    q_e = ELEMENTARY_CHARGE_C
    i_d_a = pd.dark_current_na * 1e-9
    f_n_hz = NOISE_BANDWIDTH_FRACTION * sig.rate_gbd * 1e9
    i_n_a = tia.input_noise_pa_per_sqrt_hz * 1e-12 * math.sqrt(f_n_hz)
    u0_w = 2.0 * q * (m - 1) * i_n_a / r
    a = (r / (2.0 * q * (m - 1))) ** 2
    b = 2.0 * q_e * r * k * f_n_hz
    c = i_n_a**2 + 2.0 * q_e * i_d_a * f_n_hz
    u_w = (b + math.sqrt(b * b + 4.0 * a * c)) / (2.0 * a)
    return 10.0 * math.log10(u_w / u0_w)


def crosstalk_penalty_db(agg_crosstalk_db: float) -> float:
    """Power penalty for aggregate in-band crosstalk, dB.

    For total crosstalk power a fraction X (dB, negative) of the signal,
    the eye closes by (1 - X_linear):  penalty = -10 log10(1 - 10^(X/10)).
    Coherent-beat enhancement is not modeled; treat as a floor.
    """
    if agg_crosstalk_db >= 0.0:
        raise ValueError("aggregate crosstalk must be < 0 dB")
    return -10.0 * math.log10(1.0 - db_to_linear(agg_crosstalk_db))


# --- link budget ------------------------------------------------------------


@dataclass(frozen=True)
class LossElement:
    """A named passive loss in the optical path (loss_db >= 0, in dB)."""

    name: str
    loss_db: float


@dataclass
class LinkBudget:
    """End-to-end IM-DD link budget in the OMA domain.

    Composition (all dB):

        launched OMA  = laser power - modulator IL - modulation loss
                        + 10 log10(2 (ER-1)/(ER+1))   [avg -> outer OMA]
        received OMA  = launched OMA - sum(path losses)
        required OMA  = thermal-limited sensitivity OMA
                        + RIN penalty + shot-noise penalty
                        + RIN x shot interaction + explicit penalties
        margin        = received OMA - required OMA

    Shot noise (with dark current) is modeled via :func:`shot_penalty_db`,
    evaluated at the top eye level like the RIN penalty. Both are
    signal-dependent, so the joint thermal + RIN + shot requirement is
    solved exactly (one quadratic, see ``noise_interaction_db``); the
    interaction term is what the independent dB sum would have missed
    (it is >= 0: independent summing is optimistic here, not
    conservative).

    Finite extinction ratio is charged once, at the transmitter (the
    avg -> OMA conversion inside ``launched_oma_dbm``) — see
    ``sensitivity_oma_dbm`` for why :func:`er_penalty_db` must not also be
    added to an OMA-domain sensitivity.

    ``penalties_db`` holds named explicit allocations (ISI/TDECQ, MPI,
    crosstalk via :func:`crosstalk_penalty_db`, aging, ...).
    """

    laser: Laser
    modulator: Modulator
    path: list[LossElement]
    photodiode: Photodiode
    tia: Tia
    signaling: Signaling
    penalties_db: dict[str, float] | None = None

    # -- transmit side -----------------------------------------------------

    @property
    def _avg_to_oma_db(self) -> float:
        """10 log10(OMA / P_avg) = 10 log10(2 (ER-1)/(ER+1)); +3.01 dB at inf ER."""
        er = db_to_linear(self.modulator.extinction_ratio_db)
        if math.isinf(er):
            return 10.0 * math.log10(2.0)
        return 10.0 * math.log10(2.0 * (er - 1.0) / (er + 1.0))

    @property
    def average_launch_power_dbm(self) -> float:
        """Average optical power leaving the modulator, dBm."""
        return (
            self.laser.power_dbm
            - self.modulator.insertion_loss_db
            - self.modulator.modulation_loss_db
        )

    @property
    def launched_oma_dbm(self) -> float:
        """ER-limited outer OMA leaving the modulator, dBm."""
        return self.average_launch_power_dbm + self._avg_to_oma_db

    # -- channel -------------------------------------------------------------

    @property
    def path_loss_db(self) -> float:
        return sum(e.loss_db for e in self.path)

    @property
    def received_oma_dbm(self) -> float:
        """Outer OMA at the photodiode, dBm."""
        return self.launched_oma_dbm - self.path_loss_db

    # -- receive side ---------------------------------------------------------

    @property
    def _penalties(self) -> dict[str, float]:
        return dict(self.penalties_db or {})

    @property
    def rin_penalty_db(self) -> float:
        return rin_penalty_db(
            self.laser.rin_db_hz,
            self.signaling.rate_gbd,
            self.signaling.q_factor,
            self.signaling.levels,
            extinction_ratio_db=self.modulator.extinction_ratio_db,
        )

    @property
    def shot_penalty_db(self) -> float:
        """Shot-noise + dark-current penalty at the modulator's ER, dB."""
        return shot_penalty_db(
            self.photodiode,
            self.tia,
            self.signaling,
            extinction_ratio_db=self.modulator.extinction_ratio_db,
        )

    @property
    def _joint_noise_penalty_db(self) -> float:
        """Exact joint thermal + RIN + shot required-OMA inflation, dB.

        Solves the top-eye condition with ALL noise variances summed at
        the joint operating point (k = ER/(ER-1), RIN linear per Hz):

            u R / (2 (M-1)) = q sqrt( i_n^2 + 2 q_e (R k u + I_d) f_n
                                      + (R k u)^2 RIN f_n )

        which is still quadratic in the required OMA u:

            (a - g) u^2 - b u - c = 0,   g = (R k)^2 RIN f_n,

        with a, b, c as in :func:`shot_penalty_db`. g/a = x^2 with x the
        RIN unreachability parameter of :func:`rin_penalty_db`, so a > g
        exactly when the RIN-only target is reachable. Returns
        10 log10(u / u0) against the thermal-only baseline u0.
        """
        sig = self.signaling
        q = sig.q_factor
        m = sig.levels
        r = self.photodiode.responsivity_a_per_w
        er = db_to_linear(self.modulator.extinction_ratio_db)
        k = 1.0 if math.isinf(er) else er / (er - 1.0)
        f_n_hz = NOISE_BANDWIDTH_FRACTION * sig.rate_gbd * 1e9
        i_n_a = self.tia.input_noise_pa_per_sqrt_hz * 1e-12 * math.sqrt(f_n_hz)
        q_e = ELEMENTARY_CHARGE_C
        a = (r / (2.0 * q * (m - 1))) ** 2
        b = 2.0 * q_e * r * k * f_n_hz
        c = i_n_a**2 + 2.0 * q_e * self.photodiode.dark_current_na * 1e-9 * f_n_hz
        g = (r * k) ** 2 * db_to_linear(self.laser.rin_db_hz) * f_n_hz
        if g >= a:
            raise ValueError(
                f"RIN {self.laser.rin_db_hz} dB/Hz makes the target "
                f"unreachable at {sig.rate_gbd} GBd PAM{m}"
            )
        u0_w = 2.0 * q * (m - 1) * i_n_a / r
        u_w = (b + math.sqrt(b * b + 4.0 * (a - g) * c)) / (2.0 * (a - g))
        return 10.0 * math.log10(u_w / u0_w)

    @property
    def noise_interaction_db(self) -> float:
        """RIN x shot interaction: joint solve minus the two separate penalties.

        Always >= 0 — both noises are signal-dependent, so each separate
        penalty undercounts the variance at the larger joint operating
        point. Small at the shipped presets (~0.06 dB DR4, ~0.0002 dB
        CPO) but grows to ~1 dB at high baud / low ER / -140 dB/Hz RIN.
        """
        return self._joint_noise_penalty_db - self.rin_penalty_db - self.shot_penalty_db

    @property
    def sensitivity_oma_dbm(self) -> float:
        """Required OMA at the receiver, dBm.

        Thermal-limited sensitivity (:func:`receiver_sensitivity_oma_dbm`)
        plus the exact joint thermal + RIN + shot noise requirement
        (attributed across the RIN row, the shot row, and the
        ``noise_interaction_db`` row of the waterfall) plus every entry of
        ``penalties_db``.

        The extinction-ratio effect is NOT added here: this budget is carried
        in the OMA domain, where finite ER is already charged once at the
        transmitter via the avg->OMA conversion (``launched_oma_dbm``).
        :func:`er_penalty_db` — the Agrawal (ER+1)/(ER-1) factor — is the
        conversion between an average-power sensitivity and this fixed OMA
        requirement; applying it on top of an OMA budget would double-count
        ER (it is exactly the same factor as the launch-side conversion).
        """
        s = receiver_sensitivity_oma_dbm(self.photodiode, self.tia, self.signaling)
        s += self._joint_noise_penalty_db
        s += sum(self._penalties.values())
        return s

    @property
    def margin_db(self) -> float:
        """Unallocated link margin, dB (received OMA - required OMA)."""
        return self.received_oma_dbm - self.sensitivity_oma_dbm

    # -- report -----------------------------------------------------------

    def report(self) -> str:
        """Aligned plain-text waterfall of the whole budget."""
        sig = self.signaling
        fmt = "PAM4" if sig.levels == 4 else ("NRZ" if sig.levels == 2 else f"PAM{sig.levels}")
        width = 66
        lines = [
            f"{fmt} @ {sig.rate_gbd:g} GBd ({sig.bit_rate_gbps:g} Gb/s), "
            f"target BER {sig.target_ber:.1e} (q = {sig.q_factor:.2f})",
            "-" * width,
        ]

        def row(label: str, delta_db: float | None, level_dbm: float | None):
            d = f"{delta_db:+7.2f} dB " if delta_db is not None else " " * 11
            p = f"{level_dbm:+8.2f} dBm" if level_dbm is not None else ""
            lines.append(f"  {label:<38}{d} {p}")

        p = self.laser.power_dbm
        row(f"laser line power ({self.laser.launch_reference})", None, p)
        p -= self.modulator.insertion_loss_db
        row("modulator insertion loss", -self.modulator.insertion_loss_db, p)
        p -= self.modulator.modulation_loss_db
        row("modulation loss (CW -> avg)", -self.modulator.modulation_loss_db, p)
        p += self._avg_to_oma_db
        row(
            f"avg -> outer OMA (ER {self.modulator.extinction_ratio_db:.1f} dB)",
            self._avg_to_oma_db,
            p,
        )
        row("launched OMA", None, p)
        for elem in self.path:
            p -= elem.loss_db
            row(elem.name, -elem.loss_db, p)
        row("received OMA", None, p)
        lines.append("-" * width)
        s = receiver_sensitivity_oma_dbm(self.photodiode, self.tia, sig)
        row("thermal-limited sensitivity (OMA)", None, s)
        d = self.rin_penalty_db
        s += d
        row("RIN penalty", +d, s)
        d = self.shot_penalty_db
        s += d
        row("shot noise penalty (incl. dark)", +d, s)
        d = self.noise_interaction_db
        s += d
        row("RIN x shot interaction", +d, s)
        for name, pen in self._penalties.items():
            s += pen
            row(f"penalty: {name}", +pen, s)
        row("required OMA at receiver", None, s)
        lines.append("-" * width)
        m = self.margin_db
        lines.append(f"  {'MARGIN':<38}{'':11} {m:+8.2f} dB")
        return "\n".join(lines)


# --- energy ------------------------------------------------------------------


def energy_per_bit_pj(
    link: LinkBudget,
    n_lanes: int,
    laser_shared_by: int = 1,
    laser_overhead_mw_per_device: float = 0.0,
    thermal_tuning_mw_per_lane: float = 0.0,
    serdes_pj_per_bit: float = 4.0,
) -> dict[str, float]:
    """Electrical energy per transported bit, pJ/bit, by contributor.

    Returns {"laser", "modulator", "tia", "tuning", "serdes", "total"},
    all in pJ/bit averaged over the aggregate bit rate of ``n_lanes``
    identical lanes (1 mW / 1 Gb/s = 1 pJ/bit).

    * laser: every lane must receive link.laser.power_dbm at the launch
      reference plane, so the emitted optical power scales with n_lanes
      regardless of how many physical devices supply it — energy
      conservation forbids feeding N lanes at full per-line power from one
      per-line-power emitter. Electrical = optical / wpe. Sharing a source
      (``laser_shared_by`` lanes per device, e.g. a split CW or comb)
      amortizes only ``laser_overhead_mw_per_device`` (control, TEC,
      monitoring), not the per-line optical power itself.
    * modulator / tia: the components' energy_fj_per_bit.
    * tuning: per-lane thermal tuning power (ring heaters etc.), mW.
    * serdes: host-side serdes/DSP allocation, pJ/bit (dominant in
      pluggables with full DSP; small for XSR-class CPO die-to-optics).
    """
    if n_lanes < 1 or laser_shared_by < 1:
        raise ValueError("n_lanes and laser_shared_by must be >= 1")
    if laser_overhead_mw_per_device < 0:
        raise ValueError("laser_overhead_mw_per_device must be >= 0")
    lane_gbps = link.signaling.bit_rate_gbps
    total_gbps = n_lanes * lane_gbps
    n_lasers = math.ceil(n_lanes / laser_shared_by)
    laser_mw = (
        n_lanes * dbm_to_mw(link.laser.power_dbm) / link.laser.wpe
        + n_lasers * laser_overhead_mw_per_device
    )
    out = {
        "laser": laser_mw / total_gbps,
        "modulator": link.modulator.energy_fj_per_bit * 1e-3,
        "tia": link.tia.energy_fj_per_bit * 1e-3,
        "tuning": thermal_tuning_mw_per_lane / lane_gbps,
        "serdes": serdes_pj_per_bit,
    }
    out["total"] = sum(out.values())
    return out


# --- presets -------------------------------------------------------------------


def preset_pluggable_dr4() -> LinkBudget:
    """One lane of a 400G-DR4-class pluggable: 106.25 Gb/s PAM4 over 500 m SMF.

    Silicon-photonic TX with integrated CW DFB and MZM, grating-coupled.
    All numbers are ILLUSTRATIVE 2026-plausible values assembled from
    public module specs and conference literature, not any vendor's data.
    """
    return LinkBudget(
        laser=Laser(
            power_dbm=12.0,  # CW DFB butt-coupled into the PIC, per lane
            wpe=0.12,  # DFB incl. driver overhead, uncooled
            rin_db_hz=-140.0,  # decent DFB; EMLs/DFBs spec -135..-145
            launch_reference="chip",
        ),
        modulator=Modulator(
            insertion_loss_db=5.0,  # Si depletion MZM, on-state, incl. phase-shifter loss
            extinction_ratio_db=5.0,  # outer ER at 53 GBd with driver swing limits
            energy_fj_per_bit=300.0,  # MZM + CMOS driver
        ),
        path=[
            LossElement("tx grating coupler", 1.75),  # "3.5 dB GC losses" split TX/RX
            LossElement("tx connector", 0.25),
            LossElement("500 m SMF", 0.2),  # ~0.4 dB/km O-band
            LossElement("rx connector", 0.25),
            LossElement("rx grating coupler", 1.75),
        ],
        photodiode=Photodiode(responsivity_a_per_w=0.8),  # Ge-on-Si PIN at 1310 nm
        tia=Tia(
            input_noise_pa_per_sqrt_hz=16.0,  # 100G-class linear TIA
            bandwidth_ghz=40.0,
            energy_fj_per_bit=700.0,  # linear TIA + AGC
        ),
        signaling=Signaling(rate_gbd=53.125, levels=4, target_ber=2.4e-4),  # KP4
        penalties_db={
            "isi_equalization": 1.0,  # residual ISI after RX FFE (TDECQ-style)
            "mpi": 0.3,  # multi-path interference over connectors
            "temp_aging": 0.5,  # end-of-life / temperature allocation
        },
    )


def preset_cpo_optical_io() -> LinkBudget:
    """One lane of a CPO/optical-I/O style link: 32 Gb/s NRZ, ring-based.

    Remote external laser source (ELS) delivered over fiber; microring
    modulator on the compute package; XSR-class electrical interface.
    All numbers are ILLUSTRATIVE 2026-plausible values.
    """
    return LinkBudget(
        laser=Laser(
            power_dbm=6.5,  # per wavelength line, in fiber at the ELS output
            wpe=0.10,  # ELS module incl. cooling and control
            rin_db_hz=-150.0,  # low-RIN CW DFB, key enabler for rings
            launch_reference="fiber",
        ),
        modulator=Modulator(
            insertion_loss_db=1.5,  # ring through-port on-state loss
            extinction_ratio_db=4.0,  # ring ER at speed with modest drive swing
            energy_fj_per_bit=50.0,  # microring + low-swing driver
        ),
        path=[
            LossElement("laser fiber-to-chip coupler", 1.5),  # edge coupler at TX PIC
            LossElement("tx on-chip routing", 0.5),
            LossElement("tx chip-to-fiber coupler", 1.5),
            LossElement("jumper fiber + connectors", 0.3),
            LossElement("rx fiber-to-chip coupler", 1.5),
            LossElement("rx on-chip routing", 0.5),
        ],
        photodiode=Photodiode(responsivity_a_per_w=1.0),  # waveguide Ge PD
        tia=Tia(
            input_noise_pa_per_sqrt_hz=12.0,  # low-BW NRZ TIA
            bandwidth_ghz=25.0,
            energy_fj_per_bit=250.0,
        ),
        signaling=Signaling(rate_gbd=32.0, levels=2, target_ber=2.4e-4),  # NRZ, KP4
        penalties_db={
            "wdm_crosstalk": 0.3,  # adjacent ring channels (cf. crosstalk_penalty_db)
            "wavelength_drift": 0.5,  # thermal-lock residual on rings
        },
    )
