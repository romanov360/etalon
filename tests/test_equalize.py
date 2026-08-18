"""Tests for etalon.equalize (closed-form zero-forcing FFE taps).

Anchors are analytic: a flat/dispersionless channel gives an exact delta
pulse response and a trivial (identity) FFE; a single-postcursor-echo
channel H(f) = 1 + alpha*exp(-2j*pi*f*T) has a known normalized pulse
response [1/(1+alpha), alpha/(1+alpha)] and a known geometric-series
zero-forcing tap ratio (-alpha per tap, the exact IIR inverse of a
one-pole echo).
"""

import math

import numpy as np
import pytest

from etalon import equalize

C_UM_HZ = 2.99792458e14


def _flat_channel(amp: float = 0.8, n: int = 4001):
    wl = np.linspace(1.50, 1.60, n)
    s21 = np.full(n, complex(amp, 0.0))
    return wl, s21


def _echo_channel(alpha: float, rate_gbd: float, n: int = 20001):
    """H(f) = 1 + alpha*exp(-2j*pi*f*T), T = 1/rate_gbd — one-postcursor echo."""
    wl = np.linspace(1.50, 1.60, n)
    fc = C_UM_HZ / 1.55
    f = C_UM_HZ / wl - fc
    t = 1.0 / (rate_gbd * 1e9)
    h = 1.0 + alpha * np.exp(-2j * np.pi * f * t)
    return wl, h.astype(complex)


def _delayed(wl, s21, rate_gbd: float, delay_ui: float):
    """Apply a pure bulk group delay of delay_ui symbols to a channel."""
    fc = C_UM_HZ / 1.55
    f = C_UM_HZ / wl - fc
    t = 1.0 / (rate_gbd * 1e9)
    return (np.asarray(s21) * np.exp(-2j * np.pi * f * delay_ui * t)).astype(complex)


class TestPulseResponse:
    def test_flat_channel_is_exact_delta(self):
        wl, s21 = _flat_channel()
        h = equalize.pulse_response(wl, s21, 1.55, 32.0, n_pre=2, n_post=2)
        assert h.size == 5
        assert abs(h[2] - 1.0) < 1e-9  # cursor, flat-loss normalized
        for i in (0, 1, 3, 4):
            assert abs(h[i]) < 1e-9

    def test_echo_channel_matches_analytic_normalized_response(self):
        alpha = 0.3
        rate = 32.0
        wl, s21 = _echo_channel(alpha, rate)
        h = equalize.pulse_response(wl, s21, 1.55, rate, n_pre=1, n_post=1)
        # normalized by h0 = H(fc) = 1 + alpha
        expected_cursor = 1.0 / (1.0 + alpha)
        expected_post = alpha / (1.0 + alpha)
        assert h[1].real == pytest.approx(expected_cursor, abs=2e-3)
        assert h[2].real == pytest.approx(expected_post, abs=2e-3)
        assert h[0].real == pytest.approx(0.0, abs=2e-3)

    def test_peak_seeking_alignment_finds_cursor(self):
        # Even with a big n_pre/n_post window, the cursor should land on
        # the actual peak-magnitude sample, not an arbitrary window center.
        wl, s21 = _flat_channel()
        h = equalize.pulse_response(wl, s21, 1.55, 32.0, n_pre=5, n_post=1)
        assert int(np.argmax(np.abs(h))) == 5  # cursor index = n_pre

    def test_rejects_bad_grid(self):
        wl, s21 = _flat_channel()
        with pytest.raises(ValueError):
            equalize.pulse_response(wl, s21, 1.55, 32.0, n_pre=0, n_post=0)  # both zero
        with pytest.raises(ValueError):
            equalize.pulse_response(wl, s21, 1.55, 32.0, n_pre=-1, n_post=2)
        with pytest.raises(ValueError):
            equalize.pulse_response(wl, s21, 1.55, 0.0, n_pre=1, n_post=1)
        with pytest.raises(ValueError):
            equalize.pulse_response(wl, s21, 1.55, 32.0, n_pre=1, n_post=1, samples_per_symbol=2)

    def test_center_outside_grid_raises(self):
        wl, s21 = _flat_channel()
        with pytest.raises(ValueError):
            equalize.pulse_response(wl, s21, 1.70, 32.0, n_pre=1, n_post=1)

    def test_null_at_carrier_raises(self):
        wl, s21 = _flat_channel(amp=0.0)
        with pytest.raises(ValueError, match="transmission null"):
            equalize.pulse_response(wl, s21, 1.55, 32.0, n_pre=1, n_post=1)

    def test_bulk_delay_within_reach_is_absorbed_correctly(self):
        # A 1-UI bulk delay on an otherwise-ISI-free channel: the cursor
        # should relocate to the delayed symbol (not stay at the launch
        # instant with near-zero energy), and the returned response
        # should still be a clean, correctly-normalized delta. A denser
        # grid than the other tests' default is needed here: a 1-UI phase
        # ramp at 32 GBd rotates fast enough that _baseband_response's
        # density guard needs headroom (see test_bulk_delay_beyond_tap_span
        # _raises for what happens when it doesn't have enough).
        rate = 32.0
        wl, s21 = _flat_channel(amp=0.8, n=40_001)
        delayed = _delayed(wl, s21, rate, delay_ui=1.0)
        h = equalize.pulse_response(wl, delayed, 1.55, rate, n_pre=2, n_post=2)
        assert int(np.argmax(np.abs(h))) == 2  # cursor re-centered on the peak
        assert abs(h[2] - 1.0) < 1e-3
        for i in (0, 1, 3, 4):
            assert abs(h[i]) < 1e-3

    def test_bulk_delay_beyond_tap_span_raises(self):
        # A delay too large for the tap span to reach must raise -- either
        # via the delay-vs-span check itself, or (as here, at a delay
        # steep enough to also break the density guard) via
        # _baseband_response's own coarse-grid guard. Either is an honest
        # failure, not silent garbage, which is what this test verifies.
        rate = 32.0
        wl, s21 = _flat_channel(amp=0.8, n=40_001)
        delayed = _delayed(wl, s21, rate, delay_ui=5.0)
        with pytest.raises(ValueError, match="symbols away from the launch|grid too coarse"):
            equalize.pulse_response(wl, delayed, 1.55, rate, n_pre=1, n_post=1)

    def test_dominant_postcursor_tap_relocates_cursor(self):
        # A channel where the "echo" tap is louder than the direct tap:
        # the located cursor must move to the true peak, not stay pinned
        # at the launch instant.
        alpha = 5.0
        rate = 32.0
        wl, s21 = _echo_channel(alpha, rate)
        h = equalize.pulse_response(wl, s21, 1.55, rate, n_pre=1, n_post=1)
        assert int(np.argmax(np.abs(h))) == 1  # re-centered on the true peak
        assert abs(h[2]) < 1e-5  # nothing left beyond the (relocated) cursor

    def test_far_dominant_postcursor_tap_raises(self):
        alpha = 5.0
        rate = 32.0
        wl, s21 = _echo_channel(alpha, rate)
        # n_post=0: the dominant tap sits one symbol AFTER the launch
        # instant, which n_post=0 cannot reach.
        with pytest.raises(ValueError, match="symbols away from the launch"):
            equalize.pulse_response(wl, s21, 1.55, rate, n_pre=1, n_post=0)



class TestSolveFfeTaps:
    def test_flat_channel_trivial_taps(self):
        wl, s21 = _flat_channel()
        result = equalize.solve_ffe_taps(wl, s21, 1.55, 32.0, n_pre=2, n_post=2)
        assert result.taps[2] == pytest.approx(1.0, abs=1e-6)  # cursor tap
        for i in (0, 1, 3, 4):
            assert result.taps[i] == pytest.approx(0.0, abs=1e-6)
        assert result.noise_enhancement_db == pytest.approx(0.0, abs=1e-4)
        # exactly zero response outside the tap span -> deeply negative dB
        # (float noise, not literally -inf, since the channel isn't exactly
        # bandlimited in floating point after FFT filtering)
        assert result.residual_isi_db < -100.0

    def test_echo_channel_geometric_tap_ratio(self):
        # Exact IIR inverse of H(z) = 1 + alpha*z^-1 (normalized) is
        # 1/(1+alpha*z^-1) = sum_k (-alpha)^k z^-k -- taps should alternate
        # sign with ratio -alpha between consecutive terms.
        alpha = 0.25
        rate = 28.0
        wl, s21 = _echo_channel(alpha, rate)
        result = equalize.solve_ffe_taps(wl, s21, 1.55, rate, n_pre=0, n_post=4)
        taps = np.array(result.taps)
        ratios = taps[1:] / taps[:-1]
        assert np.allclose(ratios, -alpha, atol=0.02)

    def test_more_taps_reduces_residual_isi(self):
        alpha = 0.3
        rate = 32.0
        wl, s21 = _echo_channel(alpha, rate)
        r2 = equalize.solve_ffe_taps(wl, s21, 1.55, rate, n_pre=0, n_post=2)
        r5 = equalize.solve_ffe_taps(wl, s21, 1.55, rate, n_pre=0, n_post=5)
        assert r5.residual_isi_db < r2.residual_isi_db

    def test_noise_enhancement_positive_for_typical_isi_channel(self):
        alpha = 0.3
        rate = 32.0
        wl, s21 = _echo_channel(alpha, rate)
        result = equalize.solve_ffe_taps(wl, s21, 1.55, rate, n_pre=0, n_post=3)
        assert result.noise_enhancement_db > 0.0

    def test_forward_reconstruction_matches_zero_forcing_target(self):
        # Direct algebraic check: convolving the pulse response with the
        # solved taps must reproduce a unit impulse at the cursor delay
        # across the whole forced band, independent of the module's own
        # residual/report bookkeeping.
        alpha = 0.4
        rate = 32.0
        n_pre, n_post = 2, 2
        wl, s21 = _echo_channel(alpha, rate)
        h = equalize.pulse_response(wl, s21, 1.55, rate, n_pre, n_post)
        result = equalize.solve_ffe_taps(wl, s21, 1.55, rate, n_pre, n_post)
        conv = np.convolve(h.real, np.array(result.taps))
        n_taps = n_pre + n_post + 1
        forced = conv[n_pre : n_pre + n_taps]
        expected = np.zeros(n_taps)
        expected[n_pre] = 1.0
        assert np.allclose(forced, expected, atol=1e-6)

    def test_report_runs(self):
        wl, s21 = _flat_channel()
        result = equalize.solve_ffe_taps(wl, s21, 1.55, 32.0, n_pre=1, n_post=1)
        text = result.report()
        assert "FFE" in text and "noise enhancement" in text

    def test_asymmetric_taps(self):
        alpha = 0.2
        rate = 32.0
        wl, s21 = _echo_channel(alpha, rate)
        result = equalize.solve_ffe_taps(wl, s21, 1.55, rate, n_pre=3, n_post=1)
        assert len(result.taps) == 5
        assert result.n_pre == 3 and result.n_post == 1

    def test_ring_component_end_to_end(self):
        from etalon.components import RingAllPass

        ring = RingAllPass(
            circumference_um=200.0, neff0=2.4, ng=4.2, kappa_power=0.1,
            loss_db_per_cm=3.0, wl0_um=1.55,
        )
        wl = np.linspace(1.540, 1.560, 8001)
        s21 = ring.s_params(wl)[:, 1, 0]
        result = equalize.solve_ffe_taps(wl, s21, 1.55, 32.0, n_pre=3, n_post=3)
        assert len(result.taps) == 7
        assert np.isfinite(result.noise_enhancement_db)
        assert result.residual_isi_db < -20.0  # well-suppressed with 7 taps

    def test_cursor_invariant_holds_for_flat_and_echo_channels(self):
        # argmax(|h|) must land exactly at index n_pre for any channel
        # the module accepts (it is a postcondition of the cursor-locating
        # logic, not merely a convenient special case).
        rate = 32.0
        wl_flat, s21_flat = _flat_channel()
        for n_pre, n_post in [(0, 3), (2, 2), (4, 1)]:
            h = equalize.pulse_response(wl_flat, s21_flat, 1.55, rate, n_pre, n_post)
            assert int(np.argmax(np.abs(h))) == n_pre

        wl_echo, s21_echo = _echo_channel(0.3, rate)
        for n_pre, n_post in [(0, 2), (1, 3)]:
            h = equalize.pulse_response(wl_echo, s21_echo, 1.55, rate, n_pre, n_post)
            assert int(np.argmax(np.abs(h))) == n_pre

    def test_high_q_ring_pulse_response_stable_under_more_padding(self):
        # A long-memory (high-Q) channel: the SAME small tap span must
        # report the same pulse response regardless of how much padding
        # pulse_response's internal doubling settles on -- i.e. the
        # adaptive padding must genuinely converge, not merely stop at an
        # arbitrary fixed guess. Compare against a manually-forced, much
        # heavier minimum padding via a large n_post request (which also
        # raises the internal starting padding, since it is
        # max(n_pre, n_post, GUARD_SYMBOLS)) sliced back down to the same
        # small span. THIS is the test with real regression coverage for
        # the adaptive-padding fix: reverting to a fixed/insufficient
        # padding scheme makes h_small silently diverge from
        # h_heavy_padded's first 3 taps well outside atol (confirmed by
        # deliberately hobbling the padding logic during development).
        from etalon.components import RingAllPass

        ring = RingAllPass(
            circumference_um=200.0, neff0=2.4, ng=4.2, kappa_power=0.003,
            loss_db_per_cm=0.2, wl0_um=1.55,
        )
        wl = np.linspace(1.545, 1.555, 40_001)
        s21 = ring.s_params(wl)[:, 1, 0]
        rate = 32.0
        h_small = equalize.pulse_response(wl, s21, 1.55, rate, n_pre=0, n_post=2)
        h_heavy_padded = equalize.pulse_response(wl, s21, 1.55, rate, n_pre=0, n_post=600)
        assert np.allclose(h_small, h_heavy_padded[:3], atol=1e-4)

    def test_high_q_ring_residual_reflects_real_slow_tail(self):
        # This ring has a fast near-cursor decay but a slowly-decaying
        # oscillatory tail out to 100+ symbols (a real high-Q effect, not
        # a padding artifact -- verified BY the sibling
        # test_high_q_ring_pulse_response_stable_under_more_padding,
        # which is the one with actual regression coverage for the
        # adaptive-padding fix: at n_post=400 here, pad_symbols already
        # starts at max(0, 400, GUARD_SYMBOLS)=400 on the FIRST attempt,
        # so this test alone would NOT catch a regression to fixed/
        # insufficient padding -- it only checks that a genuinely wider
        # tap span reaches materially more of a real (non-artifactual)
        # slow tail, a physical-behavior check, not a padding-correctness
        # check.
        from etalon.components import RingAllPass

        ring = RingAllPass(
            circumference_um=200.0, neff0=2.4, ng=4.2, kappa_power=0.003,
            loss_db_per_cm=0.2, wl0_um=1.55,
        )
        wl = np.linspace(1.545, 1.555, 40_001)
        s21 = ring.s_params(wl)[:, 1, 0]
        rate = 32.0
        short = equalize.solve_ffe_taps(wl, s21, 1.55, rate, n_pre=0, n_post=2)
        long = equalize.solve_ffe_taps(wl, s21, 1.55, rate, n_pre=0, n_post=400)
        # the long tap span must reach a materially better (more negative)
        # residual than the short one -- the tail is real ISI a 2-tap FFE
        # cannot see.
        assert long.residual_isi_db < short.residual_isi_db - 10.0
