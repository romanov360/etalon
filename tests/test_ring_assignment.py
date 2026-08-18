"""Tests for etalon.wdm.optimize_ring_assignment (barrel-shift tuning optimizer).

Physics anchors are analytic or independently re-derived (brute-force
enumeration in plain Python), never the code's own output.
"""

import numpy as np
import pytest

from etalon import wdm


def brute_force_best(offsets, fsr, efficiency, bidirectional):
    """Independent re-derivation: plain-Python enumeration of all rotations.

    Returns (best_rotation, best_total_mw, naive_total_mw).
    """
    n = len(offsets)
    delta = fsr / n
    totals = []
    for r in range(n):
        total = 0.0
        for off in offsets:
            x = (r * delta - off) % fsr
            h = min(x, fsr - x) if bidirectional else x
            total += h / efficiency
        totals.append(total)
    best = min(range(n), key=lambda r: totals[r])
    return best, totals, totals[0]


class TestAnalyticAnchors:
    def test_zero_offsets_rotation_zero_zero_power(self):
        res = wdm.optimize_ring_assignment([0.0] * 5, fsr_nm=16.0)
        assert res.rotation == 0
        assert res.per_ring_mw == (0.0,) * 5
        assert res.total_mw == 0.0
        assert res.mean_mw_per_ring == 0.0
        assert res.naive_total_mw == 0.0

    def test_common_mode_offset_absorbed_by_rotation(self):
        # N = 8 rings, FSR = 16 nm -> delta = 2 nm. Pure common-mode
        # offset delta_c = 7.9 nm (~FSR/2) on every ring; efficiency 1
        # nm/mW so powers read in nm.
        n, fsr, delta_c = 8, 16.0, 7.9
        delta = fsr / n
        res = wdm.optimize_ring_assignment([delta_c] * n, fsr_nm=fsr, efficiency_nm_per_mw=1.0)
        # Analytic optimum: rotation r = 4 leaves residual (4*2 - 7.9) = 0.1 nm
        # on every ring -> total 0.8 mW. Naive r = 0: (-7.9) mod 16 = 8.1 nm
        # per ring -> 64.8 mW.
        assert res.rotation == 4
        assert res.total_mw == pytest.approx(n * 0.1)
        assert res.naive_total_mw == pytest.approx(n * 8.1)
        # Quantization bound: residual per ring < delta, so total <= N * delta.
        assert res.total_mw <= n * delta
        # Several-fold drop versus the naive assignment (here ~80x).
        assert res.total_mw * 5.0 < res.naive_total_mw

    def test_n2_case_by_hand(self):
        # fsr = 10, N = 2 -> delta = 5; offsets = [1, 2] nm; efficiency 0.25.
        #   r=0: (0-1)%10 = 9, (0-2)%10 = 8   -> 17 nm
        #   r=1: (5-1)%10 = 4, (5-2)%10 = 3   ->  7 nm  (best)
        # Powers = shift / 0.25 = 4x the nm values.
        res = wdm.optimize_ring_assignment(
            [1.0, 2.0], fsr_nm=10.0, efficiency_nm_per_mw=0.25
        )
        assert res.rotation == 1
        assert res.per_ring_mw == pytest.approx((16.0, 12.0))
        assert res.total_mw == pytest.approx(28.0)
        assert res.mean_mw_per_ring == pytest.approx(14.0)
        assert res.naive_total_mw == pytest.approx(68.0)

    def test_n2_case_by_hand_bidirectional(self):
        # Same inputs, nearest-order tuning:
        #   r=0: min(9,1)=1, min(8,2)=2 -> 3 nm  (best)
        #   r=1: min(4,6)=4, min(3,7)=3 -> 7 nm
        res = wdm.optimize_ring_assignment(
            [1.0, 2.0], fsr_nm=10.0, efficiency_nm_per_mw=0.25, bidirectional=True
        )
        assert res.rotation == 0
        assert res.per_ring_mw == pytest.approx((4.0, 8.0))
        assert res.total_mw == pytest.approx(12.0)


class TestRandomizedProperties:
    def test_optimal_never_exceeds_naive_and_matches_brute_force(self):
        # In the red-shift-only model rotation totals differ only by
        # multiples of the FSR, so exact ties are common; compare achieved
        # power, not the tie-broken rotation index.
        rng = np.random.default_rng(20260723)
        fsr = 12.8
        for n in (1, 2, 3, 8, 16):
            for _ in range(10):
                offsets = rng.uniform(-fsr, fsr, size=n)
                for bidir in (False, True):
                    res = wdm.optimize_ring_assignment(
                        offsets, fsr_nm=fsr, bidirectional=bidir
                    )
                    assert res.total_mw <= res.naive_total_mw + 1e-9
                    best_r, totals, naive_total = brute_force_best(
                        list(offsets), fsr, wdm.TUNING_EFFICIENCY_NM_PER_MW, bidir
                    )
                    best_total = totals[best_r]
                    # Returned rotation achieves the brute-force optimum.
                    assert totals[res.rotation] == pytest.approx(best_total, rel=1e-9)
                    assert res.total_mw == pytest.approx(best_total, rel=1e-9)
                    assert res.naive_total_mw == pytest.approx(naive_total, rel=1e-12)
                    assert res.mean_mw_per_ring == pytest.approx(res.total_mw / n)
                    assert sum(res.per_ring_mw) == pytest.approx(res.total_mw)

    def test_bidirectional_never_worse_than_red_shift_only(self):
        # min(x, fsr - x) <= x for every ring at every rotation, so the
        # bidirectional optimum can never exceed the red-shift-only one.
        rng = np.random.default_rng(42)
        fsr = 16.0
        for n in (2, 4, 8):
            for _ in range(20):
                offsets = rng.uniform(-fsr / 2, fsr / 2, size=n)
                red = wdm.optimize_ring_assignment(offsets, fsr_nm=fsr)
                bidir = wdm.optimize_ring_assignment(
                    offsets, fsr_nm=fsr, bidirectional=True
                )
                assert bidir.total_mw <= red.total_mw + 1e-9

    def test_adding_delta_shifts_rotation_by_one_red_only(self):
        # Adding exactly delta to every offset is a relabeling of the same
        # physical situation: best rotation advances by 1 (mod N), total
        # power unchanged. Red-shift-only rotation totals tie in multiples
        # of the FSR, so use offsets clustered inside one delta bin, where
        # the optimum r = k (the bin index) is provably unique.
        # fsr = 16, N = 8 -> delta = 2.0 (exact in fp).
        rng = np.random.default_rng(7)
        n, fsr = 8, 16.0
        delta = fsr / n
        for k in (1, 2, 5, n - 2):
            offsets = (k - 1) * delta + delta * rng.uniform(0.05, 0.95, size=n)
            base = wdm.optimize_ring_assignment(offsets, fsr_nm=fsr)
            shifted = wdm.optimize_ring_assignment(offsets + delta, fsr_nm=fsr)
            assert base.rotation == k
            assert shifted.rotation == (base.rotation + 1) % n
            assert shifted.total_mw == pytest.approx(base.total_mw, rel=1e-12)

    def test_adding_delta_shifts_rotation_by_one_bidirectional(self):
        # Bidirectional totals have no FSR-multiple tie structure, so the
        # optimum is generically unique and the shift-by-1 property holds
        # for arbitrary seeded offsets (including wraparound r = N-1 -> 0).
        rng = np.random.default_rng(11)
        n, fsr = 8, 16.0
        delta = fsr / n
        for _ in range(20):
            offsets = rng.uniform(-fsr / 2, fsr / 2, size=n)
            base = wdm.optimize_ring_assignment(offsets, fsr_nm=fsr, bidirectional=True)
            shifted = wdm.optimize_ring_assignment(
                offsets + delta, fsr_nm=fsr, bidirectional=True
            )
            assert shifted.rotation == (base.rotation + 1) % n
            assert shifted.total_mw == pytest.approx(base.total_mw, rel=1e-12)


class TestValidation:
    def test_invalid_fsr(self):
        with pytest.raises(ValueError):
            wdm.optimize_ring_assignment([0.1], fsr_nm=0.0)
        with pytest.raises(ValueError):
            wdm.optimize_ring_assignment([0.1], fsr_nm=-1.0)

    def test_invalid_efficiency(self):
        with pytest.raises(ValueError):
            wdm.optimize_ring_assignment([0.1], fsr_nm=10.0, efficiency_nm_per_mw=0.0)

    def test_empty_offsets(self):
        with pytest.raises(ValueError):
            wdm.optimize_ring_assignment([], fsr_nm=10.0)

    def test_non_finite_offsets(self):
        with pytest.raises(ValueError):
            wdm.optimize_ring_assignment([0.1, np.nan], fsr_nm=10.0)
        with pytest.raises(ValueError):
            wdm.optimize_ring_assignment([np.inf], fsr_nm=10.0)

    def test_non_1d_offsets(self):
        with pytest.raises(ValueError):
            wdm.optimize_ring_assignment([[0.1, 0.2]], fsr_nm=10.0)
