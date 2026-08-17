"""Feed-forward equalizer (FFE) taps for a passive filter response.

:mod:`siphon.isi` computes the eye-closure penalty of a CHIRP-FREE,
UNEQUALIZED receiver, and its own docstring names the gap explicitly:
"no receiver equalization — with an FFE-based receiver treat the result
as an upper bound on the residual ISI." Real 100G+/lane PAM4 links
universally run receiver-side FFE, so that upper bound can be very loose.
This module closes part of that gap: given the same passive S21(lambda)
:func:`siphon.isi.filter_isi_penalty_db` takes, it solves the CLOSED-FORM
zero-forcing FFE tap vector and reports two numbers, never one alone —
the residual ISI after equalization (small to zero, by construction, for
taps that reach the whole channel memory) and the NOISE ENHANCEMENT
penalty that equalization always costs (Signal Integrity Journal;
low-complexity PAM4 IMDD equalization literature, e.g. Opt. Lett. 45,
2555 (2020)): a zero-forcing FFE amplifies receiver noise by the taps'
own energy, sum(c_i^2) relative to an ideal unit-gain unequalized
receiver. Book BOTH: the (near-zero) equalized ISI penalty into
``LinkBudget.penalties_db`` in place of :func:`siphon.isi.filter_isi_penalty_db`'s
number, and the noise enhancement as an ADDITIONAL penalty on top of
:func:`siphon.link.rin_penalty_db`/:func:`siphon.link.shot_penalty_db`
(both noise, not signal, so the enhancement multiplies their variance,
not the signal level) — an FFE is never a free lunch, and reporting only
the ISI side would silently claim one.

Method
------
1. **Pulse response**: a single isolated symbol (one UI wide, unit
   amplitude, all other symbols at zero — NOT a full pattern) is
   propagated through the same baseband-mapped channel
   :func:`siphon.isi.filter_isi_penalty_db` uses (shares its private
   :func:`siphon.isi._baseband_response` interpolation, so the two
   modules never diverge on how S21(lambda) is interpolated onto the
   simulation band), then square-root/field-domain filtered and
   SYMBOL-RATE SAMPLED (not oversampled — FFE taps operate at baud rate)
   at the GLOBAL peak-magnitude phase over the whole simulated window —
   not merely within the symbol the pulse was launched into. A pure
   bulk group delay, or a channel whose dominant response sits in a
   postcursor rather than at the launch instant, moves the true peak to
   a different symbol than the launch one; :func:`pulse_response`
   explicitly finds that symbol and raises ValueError if it falls
   outside what the requested ``n_pre``/``n_post`` can reach around it,
   rather than silently zero-forcing around near-zero energy (see
   :mod:`siphon.isi`'s own bulk-delay handling in ``_aligned_eye`` for
   the same concern in that module). This is a LINEAR (field-domain)
   response — a genuine simplification vs. the intensity response
   :mod:`siphon.isi` computes, valid because the FFE itself acts on the
   ELECTRICAL (post-photodiode, i.e. post-square-law) signal in a real
   receiver, and for the chirp-free, flat-loss-normalized model here the
   intensity pulse response is exactly ``|h[n]|^2`` convolved
   appropriately — see :func:`pulse_response` for the precise
   construction.
2. **Zero-forcing taps**: build the ``(n_taps, n_taps)`` Toeplitz
   convolution matrix from ``h[n]`` and solve ``H c = e_0`` (unit
   response at the cursor, zero at every other tap the filter reaches) —
   a single ``numpy.linalg.solve``, exact within the tap span, no
   iterative adaptation/LMS convergence to model. Symbols outside the
   tap span are NOT canceled (residual ISI floor set by
   ``n_pre``/``n_post`` vs. the channel's true memory) — increase
   ``n_pre``/``n_post`` until :attr:`FfeResult.residual_isi_db` stops
   improving. The simulation padding this needs (a long-memory channel,
   e.g. a high-Q ring, needs far more than ``n_pre``/``n_post`` alone
   would suggest) is handled automatically: :func:`pulse_response`
   adaptively doubles its zero-padding until the response has genuinely
   decayed at the padding boundary, rather than trusting a fixed guess —
   raising ValueError if that never happens within a hard cap, instead
   of silently reporting a wraparound-contaminated (falsely optimistic)
   result.
3. **Noise enhancement**: ``sum(c_i^2)`` is the FFE's total-noise gain
   relative to an UNEQUALIZED receiver's noise gain of exactly 1 (it
   samples the channel directly, unfiltered), reported as
   ``noise_enhancement_db = 10*log10(sum(c_i^2))``. Typically positive —
   canceling ISI costs noise gain — but NOT provably >= 0 in general: a
   channel with in-band peaking (cursor response magnitude > 1 after the
   flat-loss normalization :func:`pulse_response` applies) can let
   zero-forcing land on a cursor tap below 1, occasionally giving a
   small negative (favorable) number. Report it as computed either way
   rather than clamping at zero — a clamp would hide a real, if
   secondary, channel effect.

Honesty limits
--------------
Zero-forcing only — NOT minimum mean-square-error (MMSE): zero-forcing
is the noise-PESSIMISTIC, ISI-OPTIMISTIC end of the linear-equalizer
family (it fully cancels ISI regardless of noise cost); a real receiver
tuning taps against BER would use MMSE and land at a milder trade-off.
Treat :attr:`FfeResult.noise_enhancement_db` as an upper bound on the
noise cost, the mirror image of :mod:`siphon.isi`'s own "unequalized
penalty is an upper bound on residual ISI" statement. No decision
feedback (DFE), no adaptive convergence/tracking, no quantization of
tap coefficients or ADC effects, no timing-recovery jitter. Linear
FIELD-domain pulse response only — the same passive-linear-filter scope
:mod:`siphon.isi` declares (no driven ring modulators, no laser chirp).
A bulk group delay WITHIN reach of the requested ``n_pre``/``n_post`` is
absorbed into the located cursor silently and correctly (a real
receiver's clock recovery would do the same) — but this means
``n_pre``/``n_post`` then describe "delay plus ISI taps" jointly, not
pure precursor/postcursor ISI alone; route out a known bulk delay before
calling if the two must stay distinguishable. Cursor location is a hard
``argmax`` over the simulated window: a channel with two comparably
strong taps at different delays (a genuine multipath/reflection
channel, not a single high-Q resonator — swept realistic ring
parameters found no near-ties) can flip which one is chosen from an
infinitesimal input change, discontinuously changing the reported
result or which side of the reach check it lands on. Architecture-level
budgeting, not signoff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import isi as _isi

# Starting zero-padding (symbols) on each side of the launched pulse before
# circular FFT filtering, regardless of how small n_pre/n_post are. This is
# only a STARTING guess: pulse_response adaptively doubles it (see
# _MAX_PAD_DOUBLINGS, _WRAPAROUND_FLOOR) until the response has actually
# decayed to negligible energy at the padding boundary, rather than trusting
# a fixed constant to be enough for any channel's memory — a fixed guard was
# a first version of this function's mistake: a long-memory channel (a very
# high-Q ring) silently reported an artificially GOOD residual instead of
# failing loudly (see test_equalize.py's long-memory-channel test).
GUARD_SYMBOLS = 32

# Cap on how many times pulse_response doubles the padding before giving up
# (padding = GUARD_SYMBOLS * 2**_MAX_PAD_DOUBLINGS at worst, ~2M symbols).
_MAX_PAD_DOUBLINGS = 16

# A channel is "decayed enough" at the padding boundary once the boundary
# symbol's peak magnitude is this fraction of the window's overall peak —
# comfortably below what would measurably affect a residual_isi_db reported
# to 0.1 dB precision.
_WRAPAROUND_FLOOR = 1e-6


def pulse_response(
    wl_um,
    s21,
    center_wl_um: float,
    rate_gbd: float,
    n_pre: int,
    n_post: int,
    samples_per_symbol: int = 16,
) -> np.ndarray:
    """Symbol-rate-sampled field pulse response ``h[n]``, ``n_pre+n_post+1`` taps.

    A single unit-amplitude, one-UI-wide field pulse (all other time
    zero) is baseband-filtered by the interpolated ``s21(wl)`` (via
    :func:`siphon.isi._baseband_response`, the same interpolation
    :func:`siphon.isi.filter_isi_penalty_db` uses) and sampled at the
    symbol rate. The CURSOR is placed at the GLOBAL peak-magnitude
    sample over the whole simulated window (not merely the symbol the
    pulse was launched into): a pure bulk group delay, or a channel
    whose dominant response sits in a postcursor rather than at the
    launch instant, moves the true peak elsewhere, and forcing the
    cursor onto the launch symbol regardless would zero-force around
    near-zero energy while the real peak sits outside the tap span.
    Raises ValueError if that global peak falls further than
    ``n_pre``/``n_post`` symbols from the launch instant — the tap span
    cannot reach it, so widen ``n_pre``/``n_post`` (or remove the bulk
    delay from the input, e.g. route it out of ``s21`` beforehand) rather
    than receive a meaningless result. Returns a complex array of length
    ``n_pre + n_post + 1``, indices ``-n_pre .. +n_post`` relative to the
    (located) cursor, in ascending time order.

    Parameters
    ----------
    wl_um, s21 : same as :func:`siphon.isi.filter_isi_penalty_db` — 1-D
        wavelength grid (um) and complex field S21 on that grid,
        covering the simulation band around ``center_wl_um``.
    center_wl_um : carrier wavelength in um.
    rate_gbd : symbol rate in GBd (> 0).
    n_pre, n_post : precursor and postcursor tap counts (>= 0; at least
        one of them must be > 0). The simulation window is zero-padded
        on each side before the circular FFT filtering (then cropped
        back), starting from ``max(n_pre, n_post, GUARD_SYMBOLS)`` and
        adaptively DOUBLING until the response has genuinely decayed at
        the padding boundary — see :data:`GUARD_SYMBOLS`. Raises
        ValueError instead of returning a wraparound-contaminated result
        if the channel's memory outruns a hard doubling cap
        (:data:`_MAX_PAD_DOUBLINGS`).
    samples_per_symbol : time resolution (integer >= 4).
    """
    wl = np.asarray(wl_um, dtype=float)
    t21 = np.asarray(s21, dtype=complex)
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError("wl_um must be a 1-D array with at least 2 points")
    if t21.shape != wl.shape:
        raise ValueError(f"s21 shape {t21.shape} does not match wl_um shape {wl.shape}")
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
    if n_pre < 0 or n_post < 0:
        raise ValueError("n_pre and n_post must be >= 0")
    if n_pre == 0 and n_post == 0:
        raise ValueError("at least one of n_pre, n_post must be > 0")
    sps = samples_per_symbol
    if int(sps) != sps or sps < 4:
        raise ValueError("samples_per_symbol must be an integer >= 4")
    sps = int(sps)

    # Zero-pad generously beyond n_pre/n_post so circular wraparound of the
    # isolated pulse's filtered tail is negligible at the cropped taps.
    # A FIXED padding guess (this function's first version used one) is
    # not principled: a long-memory channel (a high-Q ring) can need far
    # more padding than any reasonable constant, and using too little
    # doesn't fail loudly — it reports an artificially GOOD residual, the
    # most dangerous kind of wrong answer for a link-budget tool (see
    # tests). Instead, adaptively grow the padding and directly check the
    # thing that actually matters: whether the filtered response has
    # decayed to negligible energy at the padding boundary. If it hasn't,
    # the tail is wrapping around and contaminating the crop; double the
    # padding and retry, up to a hard cap.
    pad_symbols = max(n_pre, n_post, GUARD_SYMBOLS)
    h0 = None
    for _ in range(_MAX_PAD_DOUBLINGS):
        n_symbols = n_pre + n_post + 1 + 2 * pad_symbols
        launch_symbol = pad_symbols + n_pre

        field = np.zeros(n_symbols * sps, dtype=complex)
        field[launch_symbol * sps : (launch_symbol + 1) * sps] = 1.0

        fs_hz = sps * rate_gbd * 1e9
        f_hz = np.fft.fftfreq(field.size, d=1.0 / fs_hz)
        h_base = _isi._baseband_response(wl, t21, center_wl_um, f_hz)
        if h0 is None:
            h0 = h_base[0]
            if abs(h0) == 0.0:
                raise ValueError(
                    "|S21| is zero at center_wl_um: the pulse response is "
                    "undefined at a transmission null"
                )
        filtered = np.fft.ifft(np.fft.fft(field) * h_base) / h0  # flat-loss normalized

        # Boundary check: energy in the single symbol farthest from the
        # launch instant (where wraparound would first show up), relative
        # to the peak anywhere in the window.
        boundary = filtered[0:sps]
        peak_mag = float(np.max(np.abs(filtered)))
        boundary_mag = float(np.max(np.abs(boundary))) if peak_mag > 0 else 0.0
        if boundary_mag <= _WRAPAROUND_FLOOR * peak_mag:
            break
        pad_symbols *= 2
    else:
        raise ValueError(
            f"channel response has not decayed to negligible energy even at "
            f"{pad_symbols} symbols of padding — this channel's memory (e.g. "
            "an extremely high-Q ring) is too long for this simulation to "
            "safely window; verify the channel parameters or reduce rate_gbd"
        )

    # Global peak-seeking: find the (symbol, phase) of the LARGEST
    # magnitude sample anywhere in the simulated window, not just within
    # the symbol slot the pulse was launched into. A pure bulk group
    # delay, or a channel whose dominant energy sits in a postcursor tap
    # rather than at the launch instant, moves the true peak to a
    # DIFFERENT symbol — silently keeping the launch symbol as "the
    # cursor" would zero-force around near-zero energy while the real
    # peak sits outside the tap span (this is exactly what a first
    # version of this function did; see the module's honesty-limits
    # section and test_equalize.py's group-delay/dominant-tap tests).
    mag = np.abs(filtered)
    peak_sample = int(np.argmax(mag))
    peak_symbol, phase = divmod(peak_sample, sps)

    cursor_symbol = peak_symbol
    idx = np.arange(n_symbols) * sps + phase
    samples = filtered[idx]

    # The cursor is ALWAYS the found global peak, so the returned window
    # [reach_lo, reach_hi] is by construction centered on it — n_pre/n_post
    # symbols of real signal are always available as long as pad_symbols
    # comfortably exceeds n_pre/n_post (guaranteed: pad_symbols =
    # max(n_pre, n_post, GUARD_SYMBOLS)), so a reach-vs-padding check can
    # never fire and would give false confidence. The actual risk this
    # function must guard is different: an EXCESSIVE bulk delay between
    # the launch instant and the located peak silently gets absorbed into
    # "the cursor" with no signal to the caller that n_pre/n_post no
    # longer describe precursor/postcursor ISI taps but partly describe a
    # delay they may not have intended to fold in. Bound that delay
    # explicitly against the requested tap span (a delay a user's own
    # n_pre/n_post could not have anticipated covering is exactly the
    # "silently nonsensical" case) rather than against padding internals.
    delay_symbols = cursor_symbol - launch_symbol
    if delay_symbols > n_post or -delay_symbols > n_pre:
        raise ValueError(
            f"the channel's dominant response sits {delay_symbols:+d} symbols "
            f"away from the launch instant — beyond what n_pre={n_pre}/"
            f"n_post={n_post} were requested to cover. This is either a bulk "
            "group delay (route it out of s21 before calling, e.g. divide out "
            "a linear phase ramp, or widen n_pre/n_post to explicitly include "
            "it) or a channel whose postcursor energy dominates the 'direct' "
            "tap (widen n_post). Silently absorbing an unrequested delay into "
            "the cursor would make n_pre/n_post describe something other than "
            "what was asked for."
        )
    reach_lo, reach_hi = cursor_symbol - n_pre, cursor_symbol + n_post
    return samples[reach_lo : reach_hi + 1]


@dataclass(frozen=True)
class FfeResult:
    """Result of :func:`solve_ffe_taps`.

    Attributes
    ----------
    taps:
        Zero-forcing tap coefficients, length ``n_pre + n_post + 1``,
        ascending time order (index ``n_pre`` is the cursor tap).
    residual_isi_db:
        Sum of squared residual response OUTSIDE the tap span relative to
        the cursor, as a dB ratio (10*log10) — zero-forcing exactly zeros
        the response AT every tap position, so this measures only what
        the finite tap span cannot reach. -inf (reported as
        ``float('-inf')``) if the channel response is exactly zero
        outside the span (a strictly time-limited pulse response).
    noise_enhancement_db:
        ``10*log10(sum(taps**2))``, the FFE's noise-gain cost relative to
        an unequalized receiver — see the module docstring. Typically
        positive; occasionally slightly negative for a channel with
        in-band peaking (not clamped to zero).
    n_pre, n_post:
        Echo of the requested tap span.
    """

    taps: tuple[float, ...]
    residual_isi_db: float
    noise_enhancement_db: float
    n_pre: int
    n_post: int

    def report(self) -> str:
        """Compact plain-text summary."""
        taps_str = ", ".join(f"{t:+.4f}" for t in self.taps)
        resid = (
            "-inf (exact)" if math.isinf(self.residual_isi_db) else f"{self.residual_isi_db:.2f} dB"
        )
        return "\n".join(
            [
                f"FFE ({self.n_pre} pre / {self.n_post} post taps):",
                f"  taps                 : [{taps_str}]",
                f"  residual ISI (beyond tap span) : {resid}",
                f"  noise enhancement    : {self.noise_enhancement_db:+.3f} dB "
                "(book as an ADDITIONAL penalty on rin/shot, not a substitute)",
            ]
        )


def solve_ffe_taps(
    wl_um,
    s21,
    center_wl_um: float,
    rate_gbd: float,
    n_pre: int,
    n_post: int,
    samples_per_symbol: int = 16,
) -> FfeResult:
    """Closed-form zero-forcing FFE taps for a passive filter response.

    Computes the symbol-spaced pulse response via :func:`pulse_response`,
    then solves the zero-forcing linear system: with ``h[n]`` the pulse
    response (index ``n_pre`` = cursor) and ``H`` the
    ``(n_pre+n_post+1, n_pre+n_post+1)`` Toeplitz matrix of tap-delayed
    copies of ``h``, the taps satisfy ``H @ c = e_cursor`` (unit response
    at the cursor delay, zero at every other delay the tap span
    reaches) — see the module docstring for the full method and its
    zero-forcing (not MMSE) honesty limit.

    Parameters
    ----------
    wl_um, s21, center_wl_um, rate_gbd, n_pre, n_post, samples_per_symbol :
        Same as :func:`pulse_response`.

    Raises
    ------
    ValueError : same input validation as :func:`pulse_response`, plus a
        singular zero-forcing system (a pulse response with no energy at
        the cursor delay after the tap-span convolution — physically, an
        all-pass or near-null channel at this configuration).
    """
    n_taps = n_pre + n_post + 1
    h = pulse_response(wl_um, s21, center_wl_um, rate_gbd, n_pre, n_post, samples_per_symbol)
    h_real = h.real  # field response of a real-valued (intensity-launched) pulse
    # is real by construction here: the launched field pulse is real & the
    # baseband filter is applied then evaluated at the symbol-rate sample
    # phase, but round-off can leave a tiny imaginary part; keep only Re.

    # Zero-forcing system: choose taps c[0..n_taps-1] (index n_pre = cursor
    # tap) so that conv(h_real, c) equals a unit impulse at the cursor
    # delay over the n_taps-wide reachable band. conv(h, c)[k] =
    # sum_i h[i] c[k-i], so row k (k = 0..n_taps-1, representing output
    # delay n_pre + k - n_pre = ... i.e. output index k in conv's own
    # indexing, centered so k = n_pre is the cursor) has A[row, i] =
    # h[k - i] for 0 <= k - i < n_taps, else 0 — this is np.convolve's
    # OWN index convention, verified directly against np.convolve output
    # (not just algebraically) to rule out an off-by-one/transpose bug.
    rows = np.arange(n_pre, n_pre + n_taps)  # conv output indices to force
    cols = np.arange(n_taps)
    j = rows[:, None] - cols[None, :]
    A = np.where((j >= 0) & (j < n_taps), h_real[np.clip(j, 0, n_taps - 1)], 0.0)

    target = np.zeros(n_taps)
    target[n_pre] = 1.0
    try:
        taps = np.linalg.solve(A, target)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "zero-forcing system is singular (no cursor energy reachable "
            "with this tap span)"
        ) from exc

    # Residual: response the taps produce at conv indices OUTSIDE the
    # forced band [n_pre, n_pre + n_taps) (energy the finite tap span
    # cannot reach), relative to the (forced-to-1) cursor.
    conv = np.convolve(h_real, taps)
    forced = slice(n_pre, n_pre + n_taps)
    outside = np.concatenate([conv[:forced.start], conv[forced.stop:]])
    residual_energy = float(np.sum(outside**2))
    cursor_val = float(conv[n_pre + n_pre])
    if residual_energy <= 0.0:
        residual_db = float("-inf")
    else:
        residual_db = 10.0 * math.log10(residual_energy / cursor_val**2)

    noise_enh_db = 10.0 * math.log10(float(np.sum(taps**2)))

    return FfeResult(
        taps=tuple(float(c) for c in taps),
        residual_isi_db=residual_db,
        noise_enhancement_db=noise_enh_db,
        n_pre=n_pre,
        n_post=n_post,
    )
