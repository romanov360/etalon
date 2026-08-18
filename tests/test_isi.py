"""Tests for etalon.isi (filter ISI / eye-closure penalty).

Physics anchors: a flat filter closes nothing, a pure delay is absorbed by
the sampling-phase search, amplitude scale lands entirely in the insertion
loss, a narrowing low-pass closes the eye monotonically, and the exhaustive
de Bruijn eye is never beaten (shown worse) by a long random pattern.
"""

import math

import numpy as np
import pytest

from etalon import isi
from etalon.components import RingAddDrop
from etalon.isi import (
    C_UM_HZ,
    IsiResult,
    de_bruijn_sequence,
    filter_isi_penalty_db,
)

CENTER = 1.55
RATE = 32.0


def flat_grid(value=0.7 * np.exp(1j * 0.3), n=101):
    """A generously wide flat-response grid around CENTER."""
    wl = np.linspace(1.50, 1.60, n)
    return wl, np.full(n, value, dtype=complex)


def analytic_grid(h_of_f, rate_gbd, sps, span_factor=1.2, n=400_001):
    """Sample an analytic baseband response H(f) onto a wavelength grid."""
    fc = C_UM_HZ / CENTER
    fmax = span_factor * sps * rate_gbd * 1e9 / 2.0
    wl = np.linspace(C_UM_HZ / (fc + fmax), C_UM_HZ / (fc - fmax), n)
    f = C_UM_HZ / wl - fc
    return wl, h_of_f(f).astype(complex)


def single_pole(f3db_hz):
    return lambda f: 1.0 / (1.0 + 1j * f / f3db_hz)


# --- (5) de Bruijn property -------------------------------------------------


class TestDeBruijn:
    @pytest.mark.parametrize("m,n", [(2, 8), (4, 5)])
    def test_all_cyclic_windows_distinct(self, m, n):
        seq = de_bruijn_sequence(m, n)
        assert seq.size == m**n
        assert seq.min() == 0 and seq.max() == m - 1
        doubled = np.concatenate([seq, seq[: n - 1]])
        windows = {tuple(doubled[i : i + n]) for i in range(seq.size)}
        assert len(windows) == m**n  # every length-n word appears exactly once

    def test_levels_balanced(self):
        seq = de_bruijn_sequence(4, 5)
        counts = np.bincount(seq, minlength=4)
        assert np.all(counts == 4**4)  # each symbol appears M^(n-1) times

    def test_cap(self):
        with pytest.raises(ValueError, match="cap"):
            de_bruijn_sequence(4, 9)  # 4**9 = 262144 > 65536

    def test_bad_args(self):
        with pytest.raises(ValueError):
            de_bruijn_sequence(1, 4)
        with pytest.raises(ValueError):
            de_bruijn_sequence(2, 0)


# --- (1) flat filter ----------------------------------------------------------


class TestFlatFilter:
    def test_zero_penalty_and_exact_il(self):
        wl, s21 = flat_grid(0.7 * np.exp(1j * 0.3))
        r = filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=4)
        assert r.penalty_db == pytest.approx(0.0, abs=1e-9)
        assert r.insertion_loss_db == pytest.approx(-20.0 * math.log10(0.7), abs=1e-9)

    def test_result_fields(self):
        wl, s21 = flat_grid()
        r = filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=4)
        assert isinstance(r, IsiResult)
        assert 0.0 <= r.sampling_phase_ui < 1.0
        assert len(r.eye_openings) == 3
        assert np.allclose(r.eye_openings, 1.0 / 3.0, atol=1e-12)
        assert (r.levels, r.rate_gbd, r.memory_symbols) == (4, RATE, 6)

    def test_report_string(self):
        wl, s21 = flat_grid()
        r = filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=2)
        text = r.report()
        assert "NRZ" in text and "insertion loss" in text and "UI" in text


# --- (2) pure delay -----------------------------------------------------------


class TestPureDelay:
    def test_sampling_phase_absorbs_delay(self):
        # tau = 0.3 UI; sps = 20 makes it an integer number of samples, so
        # the circular shift is exact and only interpolation noise remains.
        sps = 20
        tau_s = 0.3 / (RATE * 1e9)
        wl, s21 = analytic_grid(
            lambda f: np.exp(-2j * np.pi * f * tau_s), RATE, sps
        )
        for levels in (2, 4):
            r = filter_isi_penalty_db(
                wl, s21, CENTER, RATE, levels=levels, samples_per_symbol=sps
            )
            assert abs(r.penalty_db) < 1e-6
            assert abs(r.insertion_loss_db) < 1e-6


# --- (3) amplitude scale invariance --------------------------------------------


class TestScaleInvariance:
    def test_scale_moves_il_not_penalty(self):
        wl, s21 = analytic_grid(single_pole(0.7 * RATE * 1e9), RATE, 16)
        ra = filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=2)
        rb = filter_isi_penalty_db(wl, 0.1 * s21, CENTER, RATE, levels=2)
        assert rb.insertion_loss_db - ra.insertion_loss_db == pytest.approx(
            20.0, abs=1e-9
        )
        assert abs(rb.penalty_db - ra.penalty_db) < 1e-9


# --- (4) single-pole low-pass ----------------------------------------------------


class TestSinglePoleLowPass:
    def test_penalty_monotonic_in_bandwidth(self):
        penalties = []
        for x in (2.0, 1.0, 0.7, 0.5):
            wl, s21 = analytic_grid(single_pole(x * RATE * 1e9), RATE, 16)
            r = filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=2)
            penalties.append(r.penalty_db)
        assert all(b > a for a, b in zip(penalties, penalties[1:]))

    def test_wideband_limit_negligible(self):
        wl, s21 = analytic_grid(single_pole(50.0 * RATE * 1e9), RATE, 16)
        r = filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=2)
        # residual is band-edge truncation noise of the periodic spectrum,
        # not ISI; measured ~6e-3 dB.
        assert abs(r.penalty_db) < 0.02


# --- (6) exhaustiveness beats random ---------------------------------------------


class TestExhaustiveVsRandom:
    def test_random_pattern_never_worse(self):
        # Gaussian low-pass: negligible energy at the simulation band edge,
        # so the effective discrete filter is FFT-length independent and the
        # comparison isolates pattern coverage. PAM4 with memory 6 gives
        # 4096 distinct windows; the de Bruijn pattern hits every one.
        sps, levels, mem = 16, 4, 6
        f0 = 1.0 * RATE * 1e9
        wl, s21 = analytic_grid(
            lambda f: np.exp(-((f / f0) ** 2)), RATE, sps,
            span_factor=1.05, n=4_000_001,
        )
        r_db = filter_isi_penalty_db(
            wl, s21, CENTER, RATE, levels=levels,
            samples_per_symbol=sps, memory_symbols=mem,
        )
        assert math.isfinite(r_db.penalty_db)

        powers = isi._pam_power_levels(levels, math.inf)
        ideal = (powers[-1] - powers[0]) / (levels - 1)

        def raw_penalty(symbols):
            # identical UNALIGNED detection on both sides: the alignment
            # refinement is pattern-dependent at the sub-mdB level, so the
            # pattern-coverage claim is tested at a fixed common treatment
            # (the Gaussian filter has no bulk delay to remove)
            field = np.repeat(np.sqrt(powers[symbols]), sps)
            f_hz = np.fft.fftfreq(field.size, d=1.0 / (sps * RATE * 1e9))
            h = isi._baseband_response(wl, s21, CENTER, f_hz)
            e = np.fft.ifft(np.fft.fft(field) * h)
            intensity = np.abs(e) ** 2 / abs(h[0]) ** 2
            _, _, _, worst = isi._best_eye(intensity, symbols, levels, sps, 0)
            return 10.0 * math.log10(ideal / worst)

        penalty_debruijn_raw = raw_penalty(isi.de_bruijn_sequence(levels, mem))
        # the engine's aligned result agrees with the raw path to within
        # the sub-sample alignment refinement
        assert r_db.penalty_db == pytest.approx(penalty_debruijn_raw, abs=0.05)
        for seed in (0, 42):
            rng = np.random.default_rng(seed)
            symbols = rng.integers(0, levels, 8192)
            assert raw_penalty(symbols) <= penalty_debruijn_raw + 1e-9


# --- (7) ring add-drop memory adequacy --------------------------------------------


class TestRingDropPort:
    @staticmethod
    def _drop_response():
        # Resonance snapped exactly onto 1.55 um: L = m * wl0 / neff0.
        neff0, ng = 2.4, 4.2
        m = round(60.0 * neff0 / CENTER)
        ring = RingAddDrop(
            circumference_um=m * CENTER / neff0,
            neff0=neff0,
            ng=ng,
            kappa1_power=0.05,
            kappa2_power=0.05,
            loss_db_per_cm=2.0,
            wl0_um=CENTER,
        )
        wl = np.linspace(1.540, 1.560, 40_001)
        s = ring.s_params(wl)
        return wl, s[:, ring.ports.index("drop"), ring.ports.index("in")]

    def test_memory_convergence_and_rate_ordering(self):
        wl, drop = self._drop_response()
        kw = dict(levels=2, samples_per_symbol=32)
        p32_m6 = filter_isi_penalty_db(
            wl, drop, CENTER, 32.0, memory_symbols=6, **kw
        ).penalty_db
        p32_m7 = filter_isi_penalty_db(
            wl, drop, CENTER, 32.0, memory_symbols=7, **kw
        ).penalty_db
        p8 = filter_isi_penalty_db(
            wl, drop, CENTER, 8.0, memory_symbols=6, **kw
        ).penalty_db
        assert abs(p32_m7 - p32_m6) < 0.15  # memory 6 already converged
        assert p32_m6 > p8 >= 0.0  # narrowband filtering hurts faster bauds


# --- (8) finite extinction ratio ----------------------------------------------------


class TestFiniteExtinctionRatio:
    def test_flat_filter_still_zero(self):
        wl, s21 = flat_grid()
        r = filter_isi_penalty_db(
            wl, s21, CENTER, RATE, levels=2, extinction_ratio_db=4.0
        )
        assert r.penalty_db == pytest.approx(0.0, abs=1e-9)

    def test_lowpass_finite_and_positive(self):
        wl, s21 = analytic_grid(single_pole(0.7 * RATE * 1e9), RATE, 16)
        r_er = filter_isi_penalty_db(
            wl, s21, CENTER, RATE, levels=2, extinction_ratio_db=4.0
        )
        r_inf = filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=2)
        assert math.isfinite(r_er.penalty_db)
        assert r_er.penalty_db > 0.0
        # Verified numerically: at 4 dB ER the penalty is close to (here
        # slightly below) the infinite-ER value; assert only the closeness.
        assert r_er.penalty_db == pytest.approx(r_inf.penalty_db, abs=0.05)


# --- closed eye ---------------------------------------------------------------------


class TestClosedEye:
    def test_returns_inf_not_nan(self):
        # PAM4 through a 0.6*baud Gaussian: the eye is closed.
        f0 = 0.6 * RATE * 1e9
        wl, s21 = analytic_grid(lambda f: np.exp(-((f / f0) ** 2)), RATE, 16)
        r = filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=4)
        assert r.penalty_db == math.inf
        assert isinstance(r.penalty_db, float)
        assert "eye closed" in r.report()


# --- (9) grid span / input validation ----------------------------------------------


class TestValidation:
    def test_insufficient_span_states_required_nm(self):
        wl = np.linspace(1.5495, 1.5505, 101)  # ~1 nm; band needs ~4 nm
        s21 = np.ones(101, dtype=complex)
        with pytest.raises(ValueError, match="nm"):
            filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=2)

    def test_descending_grid_accepted(self):
        wl, s21 = analytic_grid(single_pole(0.7 * RATE * 1e9), RATE, 16)
        ra = filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=2)
        rb = filter_isi_penalty_db(wl[::-1], s21[::-1], CENTER, RATE, levels=2)
        assert rb.penalty_db == pytest.approx(ra.penalty_db, abs=1e-12)
        assert rb.insertion_loss_db == pytest.approx(ra.insertion_loss_db, abs=1e-12)

    def test_non_monotonic_wl(self):
        wl = np.array([1.50, 1.56, 1.54, 1.60])
        s21 = np.ones(4, dtype=complex)
        with pytest.raises(ValueError, match="monotonic"):
            filter_isi_penalty_db(wl, s21, CENTER, RATE)

    def test_shape_mismatch(self):
        wl, s21 = flat_grid()
        with pytest.raises(ValueError, match="shape"):
            filter_isi_penalty_db(wl, s21[:-1], CENTER, RATE)

    def test_center_outside_grid(self):
        wl, s21 = flat_grid()
        with pytest.raises(ValueError, match="outside"):
            filter_isi_penalty_db(wl, s21, 1.31, RATE)

    def test_bad_levels(self):
        wl, s21 = flat_grid()
        for levels in (1, 3, 6):
            with pytest.raises(ValueError, match="power of two"):
                filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=levels)

    def test_bad_rate(self):
        wl, s21 = flat_grid()
        with pytest.raises(ValueError, match="rate_gbd"):
            filter_isi_penalty_db(wl, s21, CENTER, 0.0)

    def test_bad_extinction_ratio(self):
        wl, s21 = flat_grid()
        for er in (0.0, -3.0):
            with pytest.raises(ValueError, match="extinction"):
                filter_isi_penalty_db(wl, s21, CENTER, RATE, extinction_ratio_db=er)

    def test_bad_samples_per_symbol(self):
        wl, s21 = flat_grid()
        for sps in (3, 7.5):
            with pytest.raises(ValueError, match="samples_per_symbol"):
                filter_isi_penalty_db(wl, s21, CENTER, RATE, samples_per_symbol=sps)

    def test_bad_memory_symbols(self):
        wl, s21 = flat_grid()
        with pytest.raises(ValueError, match="memory_symbols"):
            filter_isi_penalty_db(wl, s21, CENTER, RATE, memory_symbols=1)

    def test_pattern_cap(self):
        wl, s21 = flat_grid()
        with pytest.raises(ValueError, match="cap"):
            filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=4, memory_symbols=9)

    def test_null_at_carrier(self):
        wl, s21 = flat_grid(0.0)
        with pytest.raises(ValueError, match="zero"):
            filter_isi_penalty_db(wl, s21, CENTER, RATE)


class TestReviewRegressions:
    """Pins for the 2026-07-24 adversarial review of this module.

    R-isi-1 (high, all 3 reviewers): bulk group delay >= 1 UI mislabeled
    every sample and returned penalty = inf for a physically penalty-free
    pure delay. Now removed by cross-correlation alignment.
    R-isi-2 (low): coarse wavelength grids silently distorted fast-rotating
    phasors (long paths); now a ValueError names the required density.
    R-isi-3 (low): NaN in s21 propagated silently into the penalty.
    """

    def test_multi_ui_pure_delay_is_penalty_free(self):
        # delays of exactly and beyond one UI: previously inf, truly ~0
        t_symbol = 1.0 / (RATE * 1e9)
        for tau_ui in (1.0, 1.3, 2.5):
            tau = tau_ui * t_symbol
            wl, s21 = analytic_grid(
                lambda f: np.exp(-2j * np.pi * f * tau), RATE, 16
            )
            r = filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=2)
            # integer-UI delays align exactly (phase-slope estimator is
            # exact); fractional-UI delays carry a sub-milli-dB residual
            # from square-law Gibbs ringing sampled a fraction of a sample
            # off-grid — the pinned bug was penalty = inf
            tol = 1e-6 if tau_ui == round(tau_ui) else 1e-3
            assert abs(r.penalty_db) < tol, tau_ui
            # on a wide-open eye every (symbol-offset, phase) combination
            # ties — FFT noise picks among equally valid operating points
            # up to a symbol away — so the reported delay is only pinned to
            # within the search granularity
            assert r.bulk_delay_ui == pytest.approx(tau_ui, abs=1.0 + 1.0 / 16)

    def test_delay_invariance_of_a_real_filter(self):
        # a filter's penalty must not depend on how far away it sits
        f3 = 0.7 * RATE * 1e9
        t_symbol = 1.0 / (RATE * 1e9)

        def pole(f):
            return 1.0 / (1.0 + 1j * f / f3)

        wl, s21 = analytic_grid(pole, RATE, 16)
        p_ref = filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=2).penalty_db
        for tau_ui in (0.6, 0.95, 5.0):
            wl, s21 = analytic_grid(
                lambda f: pole(f) * np.exp(-2j * np.pi * f * tau_ui * t_symbol),
                RATE,
                16,
            )
            p = filter_isi_penalty_db(wl, s21, CENTER, RATE, levels=2).penalty_db
            # invariant to sub-sample alignment precision (parabolic
            # refinement of the correlation peak, ~0.1 sample residual);
            # the pinned bug was 0.5 -> 2.2 -> 13.8 -> inf dB
            assert p == pytest.approx(p_ref, abs=5e-3), tau_ui

    def test_ring_plus_routing_straight_matches_ring_alone(self):
        # the reviewers' plausible-use case: demux ring behind ordinary routing
        from etalon.components import Straight

        wl = np.linspace(1.545, 1.555, 60_001)
        ring = RingAddDrop(
            circumference_um=60.0, neff0=2.4, ng=4.2,
            kappa1_power=0.05, kappa2_power=0.05,
        )
        drop = ring.s_params(wl)[:, 3, 0]
        res_wl = wl[np.argmax(np.abs(drop))]
        p_ring = filter_isi_penalty_db(wl, drop, float(res_wl), RATE, levels=2)
        for length_um in (1000.0, 2000.0):
            route = Straight(length_um=length_um, neff0=2.4, ng=4.2)
            s21 = drop * route.s_params(wl)[:, 1, 0]
            p_both = filter_isi_penalty_db(wl, s21, float(res_wl), RATE, levels=2)
            assert math.isfinite(p_both.penalty_db)
            assert p_both.penalty_db == pytest.approx(p_ring.penalty_db, abs=0.02)

    def test_coarse_grid_raises_instead_of_fabricating_penalty(self):
        from etalon.components import Straight

        route = Straight(length_um=670.0, neff0=2.4, ng=4.2)
        fine_wl = np.linspace(1.50, 1.60, 200_001)
        fine = filter_isi_penalty_db(
            fine_wl, route.s_params(fine_wl)[:, 1, 0], CENTER, RATE, levels=2
        )
        # pure delay, dense grid: ~0 to sub-sample alignment precision
        assert abs(fine.penalty_db) < 0.01
        coarse_wl = np.linspace(1.50, 1.60, 201)
        with pytest.raises(ValueError, match="denser"):
            filter_isi_penalty_db(
                coarse_wl, route.s_params(coarse_wl)[:, 1, 0], CENTER, RATE, levels=2
            )

    def test_nonfinite_s21_raises(self):
        wl, s21 = flat_grid()
        s21[50] = np.nan
        with pytest.raises(ValueError, match="finite"):
            filter_isi_penalty_db(wl, s21, CENTER, RATE)
