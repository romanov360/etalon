"""PAM eye-closure (ISI) penalty of a passive optical filter response.

The E-O-E bridge between the frequency-domain circuit solver and the link
budget: take a composed complex field response S21(lambda) — a demux
passband, grating-coupler ripple, a filter cascade from
:class:`etalon.circuit.Circuit` — and compute the worst-case PAM eye
closure it inflicts on an intensity-modulated signal, as a dB penalty
ready for ``LinkBudget.penalties_db``:

    t = circuit.transmission(wl, 'in', 'drop')          # complex S21(wl)
    r = filter_isi_penalty_db(wl, t, center_wl_um=1.311, rate_gbd=32.0,
                              levels=2)
    budget.penalties_db['demux_isi'] = r.penalty_db

Scope (the module's contract)
-----------------------------
* PASSIVE LINEAR FIELD FILTERING ONLY, downstream of the modulator. The
  circuit solver's S(lambda) is a CW steady-state response, so a DRIVEN
  ring modulator is out of scope: a modulated ring is a time-varying
  cavity whose modulation response is not its static linewidth. Feed this
  function demuxes, couplers, and other passives the signal traverses
  after being modulated.
* Chirp-free ideal intensity modulation: E_in(t) = sqrt(P(t)) with P(t)
  the ideal rectangular PAM-M waveform, M levels equally spaced between
  P0 and P1 with outer extinction ratio ER = P1/P0. No transmitter rise
  time, no laser chirp, no receiver equalization — with an FFE-based
  receiver treat the result as an upper bound on the residual ISI.
* Budgeting-grade eye closure, NOT TDECQ compliance: architecture-level
  numbers, not signoff.

Method
------
1. Baseband mapping: carrier fc = C/center_wl_um with
   C = 2.99792458e14 um*Hz. The periodic simulation holds
   N = M**memory_symbols symbols of samples_per_symbol samples each; FFT
   bin k (numpy fftfreq layout) maps to optical frequency fc + f_k, i.e.
   wavelength C/(fc + f_k). The supplied s21(wl_um) is interpolated
   linearly (on real and imaginary parts separately) onto those
   wavelengths; the grid must cover the whole simulation band — no
   extrapolation, a too-narrow grid raises ValueError stating the
   required span in nm.
2. Pattern: a cyclic de Bruijn sequence B(M, memory_symbols), in which
   every possible symbol subsequence of length memory_symbols appears
   exactly once. Circular FFT filtering of a cyclic pattern is exact (no
   edge transients), and the eye search is exhaustive over every pattern
   the filter memory can see — deterministic, no random-pattern luck.
3. The FIELD is propagated, e = ifft(fft(sqrt(P)) * H_baseband), then
   square-law detected, I(t) = |e|^2 — exact within the chirp-free model.
   The intensity is never filtered directly; the coherent cross-terms of
   the field convolution matter.
4. Flat loss is normalized out: I(t) is divided by |H(fc)|^2 and
   ``insertion_loss_db = -20*log10(|H(fc)|)`` is reported SEPARATELY.
   Book each number exactly once: the IL as a LossElement in the link
   path (verify your path actually contains a loss element for this
   filter — presets may not), the penalty in ``penalties_db``. Adding
   the IL in both places double-counts it. The penalty is eye closure
   BEYOND flat loss.
5. Bulk delay is removed before the eye search: the detected intensity is
   aligned to the transmitted pattern by circular cross-correlation (the
   de Bruijn sequence makes the peak unique), so ANY magnitude of group
   delay — routing waveguides, long paths — is absorbed exactly as
   receiver clock recovery would, and reported as ``bulk_delay_ui``.
6. Eye: at each of the samples_per_symbol sampling phases (scanning
   +/- half a UI about the alignment), one sample per symbol is taken and
   grouped by transmitted level; sub-eye opening j is
   min(samples of level j+1) - max(samples of level j), and the worst
   sub-eye is minimized over j. The sampling phase maximizing the worst
   sub-eye is chosen, and
   penalty_db = 10*log10(ideal_spacing / worst_opening) with
   ideal_spacing = (P1 - P0)/(M - 1) under the same normalization. A
   closed eye (worst_opening <= 0) returns penalty_db = math.inf.

Conventions: wavelengths in um, symbol rates in GBd, penalties/losses in
dB; S21 is a complex FIELD amplitude in the library-wide
exp(-1j*2*pi*neff*L/wl) delay convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Speed of light expressed so that f[Hz] = C_UM_HZ / wl[um].
C_UM_HZ = 2.99792458e14

# Cap on the de Bruijn pattern length M**memory_symbols (symbols); with the
# default 16 samples/symbol this bounds the FFT at ~1M points.
MAX_PATTERN_SYMBOLS = 65536


def _check_levels(levels: int) -> None:
    """Same convention as etalon.link: PAM order is a power of two >= 2."""
    if levels < 2 or 2 ** int(round(math.log2(levels))) != levels:
        raise ValueError("levels must be a power of two >= 2 (2=NRZ, 4=PAM4, ...)")


def de_bruijn_sequence(alphabet: int, subsequence: int) -> np.ndarray:
    """Cyclic de Bruijn sequence B(k, n) over symbols 0..k-1, length k**n.

    Every one of the k**n possible symbol subsequences of length n appears
    exactly once as a cyclic window — the property that makes the eye
    search of :func:`filter_isi_penalty_db` exhaustive for a filter with
    n symbols of memory. Standard recursive (Lyndon-word concatenation)
    construction; deterministic.

    Parameters
    ----------
    alphabet : number of symbols k >= 2.
    subsequence : window length n >= 1.
    """
    k, n = int(alphabet), int(subsequence)
    if k < 2 or n < 1:
        raise ValueError("need alphabet >= 2 and subsequence >= 1")
    if k**n > MAX_PATTERN_SYMBOLS:
        raise ValueError(
            f"de Bruijn length {k}**{n} = {k**n} exceeds the "
            f"{MAX_PATTERN_SYMBOLS}-symbol cap"
        )
    a = [0] * (k * n)
    seq: list[int] = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                seq.extend(a[1 : p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return np.asarray(seq, dtype=np.intp)


def _pam_power_levels(levels: int, extinction_ratio_db: float) -> np.ndarray:
    """M power levels equally spaced on [P0, 1], P1/P0 = outer ER (linear)."""
    if not extinction_ratio_db > 0.0:
        raise ValueError("extinction_ratio_db must be > 0 dB")
    p0 = 0.0 if math.isinf(extinction_ratio_db) else 10.0 ** (-extinction_ratio_db / 10.0)
    return np.linspace(p0, 1.0, levels)


def _baseband_response(
    wl_um: np.ndarray,
    s21: np.ndarray,
    center_wl_um: float,
    f_hz: np.ndarray,
) -> np.ndarray:
    """Interpolate s21(wl) onto the baseband FFT bins fc + f_hz.

    Linear interpolation on real and imaginary parts separately; raises
    ValueError (stating the required span in nm) if the simulation band
    falls outside the supplied grid — never extrapolates.
    """
    fc = C_UM_HZ / center_wl_um
    wl_needed = C_UM_HZ / (fc + f_hz)
    lo, hi = float(wl_needed.min()), float(wl_needed.max())
    if lo < wl_um[0] or hi > wl_um[-1]:
        raise ValueError(
            f"s21 grid [{wl_um[0]:.6f}, {wl_um[-1]:.6f}] um does not cover the "
            f"simulation band [{lo:.6f}, {hi:.6f}] um: a span of at least "
            f"{(hi - lo) * 1e3:.4f} nm around {center_wl_um:.6f} um is "
            "required (refusing to extrapolate)"
        )
    # Density guard: linear interpolation of real/imag parts silently
    # attenuates a phasor that rotates fast between grid points (long
    # routing paths). Require < pi/2 of phase step between adjacent user
    # samples inside the band, ignoring steps at deep nulls where the
    # amplitude (and hence the interpolation error) is negligible.
    in_band = (wl_um >= lo) & (wl_um <= hi)
    idx = np.flatnonzero(in_band)
    if idx.size >= 2:
        seg = slice(max(idx[0] - 1, 0), min(idx[-1] + 2, wl_um.size))
        s_seg = s21[seg]
        amp_floor = 0.01 * float(np.abs(s_seg).max())
        dphi = np.abs(np.angle(s_seg[1:] * np.conj(s_seg[:-1])))
        significant = np.minimum(np.abs(s_seg[1:]), np.abs(s_seg[:-1])) > amp_floor
        if significant.any():
            worst_step = float(dphi[significant].max())
            if worst_step > 0.5 * np.pi:
                raise ValueError(
                    f"s21 grid too coarse: the phase rotates up to "
                    f"{worst_step:.2f} rad between adjacent wavelength samples "
                    f"inside the simulation band (limit pi/2). Resample s21 on "
                    f"a grid at least {math.ceil(worst_step / (0.25 * np.pi))}x "
                    "denser — long routing paths rotate the phase quickly."
                )
    return np.interp(wl_needed, wl_um, s21.real) + 1j * np.interp(
        wl_needed, wl_um, s21.imag
    )


def _eye_at_labels(
    samples: np.ndarray, labels: np.ndarray, levels: int
) -> tuple[np.ndarray, np.ndarray]:
    """(openings, worst-per-phase) of the (n_symbols, sps) sample matrix."""
    n_phases = samples.shape[1]
    lo = np.empty((levels, n_phases))
    hi = np.empty((levels, n_phases))
    for j in range(levels):
        rows = samples[labels == j]
        lo[j] = rows.min(axis=0)
        hi[j] = rows.max(axis=0)
    openings = lo[1:] - hi[:-1]  # (levels-1, n_phases)
    return openings, openings.min(axis=0)


def _best_eye(
    intensity: np.ndarray,
    symbols: np.ndarray,
    levels: int,
    sps: int,
    max_symbol_offset: int,
) -> tuple[int, int, np.ndarray, float]:
    """Exhaustive eye search over symbol offsets and sampling phases.

    Returns (best symbol offset q, best phase index, sub-eye openings,
    worst opening). Sample row i is labeled with symbol i - q; searching
    q in [-max_symbol_offset, max_symbol_offset] as well as the sps
    sampling phases makes the search robust to the eye optimum sitting
    whole symbols away from the nominal alignment (asymmetric filtered
    pulses — a ring's stored energy places the optimum well after the
    pulse's correlation peak). Sub-eye opening j = min(samples of level
    j+1) - max(samples of level j); the result maximizes the worst
    sub-eye. Ties prefer the smallest |q| (q, then -q), then the lowest
    phase index — deterministic.
    """
    samples = intensity.reshape(-1, sps)  # (n_symbols, sps)
    best: tuple[float, int, int, np.ndarray | None] = (-np.inf, 0, 0, None)
    order = [0]
    for q in range(1, max_symbol_offset + 1):
        order.extend((q, -q))
    for q in order:
        openings, worst = _eye_at_labels(samples, np.roll(symbols, q), levels)
        phase = int(np.argmax(worst))
        if worst[phase] > best[0]:
            best = (float(worst[phase]), q, phase, openings[:, phase].copy())
    worst_val, q, phase, best_openings = best
    # order always contains at least q=0, and worst[phase] > -inf holds on
    # any real float, so best is replaced at least once — best_openings is
    # never actually None here, just untraceable to mypy through the loop.
    assert best_openings is not None
    return q, phase, best_openings, worst_val


def _aligned_eye(
    field: np.ndarray,
    h: np.ndarray,
    h0: complex,
    powers: np.ndarray,
    symbols: np.ndarray,
    levels: int,
    sps: int,
    max_symbol_offset: int,
    rate_gbd: float,
    fs_hz: float,
    f_hz: np.ndarray,
) -> tuple[int, int, np.ndarray, float, float]:
    """Bulk-delay-aligned detection and exhaustive eye search.

    Bulk-delay removal: a composed response carries group delay (routing
    waveguides, long paths) that can exceed one UI. The eye search labels
    samples by transmitted symbol, so without alignment an integer-UI
    delay mislabels every sample and reports a fake closed eye — and even
    a fractional-UI delay parks the eye between grid samples, faking a
    small penalty for a penalty-free allpass. The bulk delay is estimated
    from the phase slope of the intensity-vs-ideal cross spectrum over
    the low signal harmonics, taken as magnitude-weighted adjacent-bin
    phase DIFFERENCES — each lies in (-pi, pi], so no unwrap is needed
    (unwrap corrupts at zero-magnitude bins, which the weighting zeroes
    out); exact for a pure delay. The delay is removed on the FIELD (a
    pure linear phase — allpass, distorts nothing) so the eye lands on
    the sampling grid, and :func:`_best_eye` additionally scans integer
    symbol offsets, because asymmetric filtered pulses (rings) park the
    eye optimum whole symbols away from any correlation-based alignment.

    Returns (symbol offset, phase index, openings, worst opening, shift
    in samples applied before the search).
    """
    spectrum = np.fft.fft(field)
    intensity = np.abs(np.fft.ifft(spectrum * h)) ** 2 / abs(h0) ** 2

    ideal = np.repeat(powers[symbols], sps)
    cross = np.fft.fft(intensity) * np.conj(np.fft.fft(ideal))
    band = (f_hz > 0.0) & (f_hz <= 0.75 * rate_gbd * 1e9)
    ck = cross[band][np.argsort(f_hz[band])]
    pair = ck[1:] * np.conj(ck[:-1])
    wgt = np.abs(pair)
    df_hz = fs_hz / field.size  # uniform FFT bin spacing
    slope = float(np.sum(wgt * np.angle(pair)) / np.sum(wgt)) / df_hz  # rad/Hz
    tau_samples = -slope / (2.0 * np.pi) * fs_hz
    shift = tau_samples - sps // 2
    h_aligned = h * np.exp(2j * np.pi * f_hz * (shift / fs_hz))
    intensity = np.abs(np.fft.ifft(spectrum * h_aligned)) ** 2 / abs(h0) ** 2

    q_sym, phase, openings, worst = _best_eye(
        intensity, symbols, levels, sps, max_symbol_offset
    )
    return q_sym, phase, openings, worst, shift


@dataclass(frozen=True)
class IsiResult:
    """Eye-closure result of :func:`filter_isi_penalty_db`.

    Attributes
    ----------
    penalty_db:
        Worst-sub-eye closure beyond flat loss, dB (>= ~0; math.inf if the
        eye is closed). This is the number for ``LinkBudget.penalties_db``.
    insertion_loss_db:
        Flat loss at the carrier, -20*log10(|H(fc)|), dB. Reported
        separately because it normally already sits in the link path
        budget — do NOT book both this IL and the penalty from here twice.
    sampling_phase_ui:
        Chosen sampling phase within the symbol, in [0, 1) UI (the
        fractional part of ``bulk_delay_ui``).
    eye_openings:
        Sub-eye openings (level j -> j+1, ascending) at the chosen phase,
        normalized so the ideal spacing is (P1 - P0)/(M - 1) with P1 = 1.
    levels, rate_gbd, memory_symbols:
        Echo of the simulation settings.
    bulk_delay_ui:
        Total clock offset absorbed before the eye search, in UI (modulo
        the cyclic pattern length): the response's aggregate group delay,
        found by cross-correlation. Any magnitude of pure delay is
        penalty-free, as clock recovery makes it in a real receiver.
    """

    penalty_db: float
    insertion_loss_db: float
    sampling_phase_ui: float
    eye_openings: tuple[float, ...]
    levels: int
    rate_gbd: float
    memory_symbols: int
    bulk_delay_ui: float = 0.0

    def report(self) -> str:
        """Compact plain-text summary."""
        fmt = (
            "PAM4"
            if self.levels == 4
            else ("NRZ" if self.levels == 2 else f"PAM{self.levels}")
        )
        eyes = ", ".join(f"{o:.4f}" for o in self.eye_openings)
        pen = "inf (eye closed)" if math.isinf(self.penalty_db) else f"{self.penalty_db:.3f} dB"
        return "\n".join(
            [
                f"filter ISI penalty ({fmt} @ {self.rate_gbd:g} GBd, "
                f"de Bruijn memory {self.memory_symbols}):",
                f"  ISI penalty (beyond flat loss) : {pen}",
                f"  insertion loss at carrier      : {self.insertion_loss_db:.3f} dB "
                "(book once, in the path budget)",
                f"  best sampling phase            : {self.sampling_phase_ui:.4f} UI",
                f"  bulk group delay absorbed      : {self.bulk_delay_ui:.4f} UI",
                f"  sub-eye openings (normalized)  : {eyes}",
            ]
        )


def filter_isi_penalty_db(
    wl_um,
    s21,
    center_wl_um: float,
    rate_gbd: float,
    levels: int = 4,
    extinction_ratio_db: float = math.inf,
    samples_per_symbol: int = 16,
    memory_symbols: int = 6,
) -> IsiResult:
    """ISI (eye-closure) penalty of a passive filter S21(lambda) on PAM-M.

    Time-domain, exhaustive, deterministic: a cyclic de Bruijn
    B(levels, memory_symbols) PAM pattern is intensity-modulated onto the
    carrier at ``center_wl_um``, the FIELD sqrt(P(t)) is filtered by the
    supplied response via circular FFT (exact for a cyclic pattern), and
    the square-law-detected eye is searched over every sampling phase and
    every symbol subsequence the filter memory can see. See the module
    docstring for the full method, scope (passive linear filtering only,
    chirp-free modulation), and normalization (flat loss reported
    separately as ``insertion_loss_db`` — do not double-count it).

    Parameters
    ----------
    wl_um : 1-D array of wavelengths in um, strictly monotonic (either
        direction), covering the simulation band
        [C/(fc + f_max), C/(fc - f_max)], f_max ~ samples_per_symbol/2 *
        rate_gbd; ValueError states the required span in nm otherwise.
    s21 : complex field transmission at ``wl_um`` (same shape), e.g.
        ``circuit.transmission(wl_um, 'in', 'drop')``.
    center_wl_um : carrier wavelength in um; must lie inside ``wl_um``.
    rate_gbd : symbol rate in GBd (> 0).
    levels : PAM order M, power of two >= 2 (2=NRZ, 4=PAM4, ...).
    extinction_ratio_db : outer ER = P1/P0 in dB (> 0; default infinite).
    samples_per_symbol : time resolution and sampling-phase granularity
        (integer >= 4).
    memory_symbols : filter memory covered exhaustively, in symbols
        (integer >= 2); the pattern holds levels**memory_symbols symbols
        (capped at 65536). Increase until the penalty stops moving (a ring
        with photon lifetime ~ n UI needs memory_symbols > n).

    Intended flow::

        t = circuit.transmission(wl, 'in', 'drop')
        r = filter_isi_penalty_db(wl, t, center_wl_um=1.311, rate_gbd=32.0,
                                  levels=2)
        budget.penalties_db['demux_isi'] = r.penalty_db
    """
    wl = np.asarray(wl_um, dtype=float)
    t21 = np.asarray(s21, dtype=complex)
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError("wl_um must be a 1-D array with at least 2 points")
    if t21.shape != wl.shape:
        raise ValueError(
            f"s21 shape {t21.shape} does not match wl_um shape {wl.shape}"
        )
    if not np.all(np.isfinite(wl)) or not np.all(np.isfinite(t21)):
        raise ValueError("wl_um and s21 must be finite (no NaN/inf)")
    d = np.diff(wl)
    if np.all(d < 0):
        wl, t21 = wl[::-1], t21[::-1]
    elif not np.all(d > 0):
        raise ValueError("wl_um must be strictly monotonic")
    if not wl[0] < center_wl_um < wl[-1]:
        raise ValueError(
            f"center_wl_um = {center_wl_um} um lies outside the supplied grid "
            f"[{wl[0]}, {wl[-1]}] um"
        )
    if rate_gbd <= 0.0:
        raise ValueError("rate_gbd must be positive")
    _check_levels(levels)
    sps = samples_per_symbol
    if int(sps) != sps or sps < 4:
        raise ValueError("samples_per_symbol must be an integer >= 4")
    sps = int(sps)
    mem = memory_symbols
    if int(mem) != mem or mem < 2:
        raise ValueError("memory_symbols must be an integer >= 2")
    mem = int(mem)
    if levels**mem > MAX_PATTERN_SYMBOLS:
        raise ValueError(
            f"pattern length {levels}**{mem} = {levels**mem} exceeds the "
            f"{MAX_PATTERN_SYMBOLS}-symbol cap; reduce memory_symbols"
        )

    powers = _pam_power_levels(levels, extinction_ratio_db)
    symbols = de_bruijn_sequence(levels, mem)
    field = np.repeat(np.sqrt(powers[symbols]), sps)

    fs_hz = sps * rate_gbd * 1e9  # sample rate; duration N*T, T = 1/rate_gbd
    f_hz = np.fft.fftfreq(field.size, d=1.0 / fs_hz)
    h = _baseband_response(wl, t21, center_wl_um, f_hz)
    h0 = h[0]  # bin 0 is exactly the carrier fc
    if abs(h0) == 0.0:
        raise ValueError(
            "|S21| is zero at center_wl_um: flat-loss normalization and the "
            "ISI penalty are undefined at a transmission null"
        )

    q_sym, phase, openings, worst, shift = _aligned_eye(
        field, h, h0, powers, symbols, levels, sps, mem, rate_gbd, fs_hz, f_hz
    )
    ideal_spacing = (powers[-1] - powers[0]) / (levels - 1)
    if worst <= 0.0:
        penalty = math.inf
    else:
        penalty = 10.0 * math.log10(ideal_spacing / worst)

    total_samples = (shift + q_sym * sps + phase) % field.size
    return IsiResult(
        penalty_db=penalty,
        insertion_loss_db=-20.0 * math.log10(abs(h0)),
        sampling_phase_ui=(total_samples % sps) / sps,
        eye_openings=tuple(float(o) for o in openings),
        levels=levels,
        rate_gbd=rate_gbd,
        memory_symbols=mem,
        bulk_delay_ui=total_samples / sps,
    )
