"""Tests for etalon.thermal (ring-to-ring thermal crosstalk)."""

import numpy as np
import pytest

from etalon import thermal, wdm


class TestRingLayout:
    def test_uniform_positions(self):
        layout = thermal.RingLayout.uniform(4, 20.0)
        assert layout.positions_um == (0.0, 20.0, 40.0, 60.0)
        assert layout.n == 4

    def test_pitch_matrix_symmetric_zero_diagonal(self):
        layout = thermal.RingLayout.uniform(3, 10.0)
        pm = layout.pitch_matrix_um()
        assert np.allclose(np.diag(pm), 0.0)
        assert np.allclose(pm, pm.T)
        assert np.allclose(pm, [[0, 10, 20], [10, 0, 10], [20, 10, 0]])

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            thermal.RingLayout(())

    def test_invalid_uniform_args(self):
        with pytest.raises(ValueError):
            thermal.RingLayout.uniform(0, 10.0)
        with pytest.raises(ValueError):
            thermal.RingLayout.uniform(3, 0.0)

    def test_nonfinite_rejected(self):
        with pytest.raises(ValueError):
            thermal.RingLayout((0.0, float("nan")))


class TestCrosstalkKernel:
    def test_zero_distance_is_one(self):
        assert thermal.crosstalk_kernel(np.array([0.0]), decay_um=20.0)[0] == pytest.approx(1.0)

    def test_decays_exponentially(self):
        decay = 25.0
        dx = np.array([0.0, decay, 2 * decay])
        k = thermal.crosstalk_kernel(dx, decay_um=decay)
        assert np.allclose(k, [1.0, np.exp(-1.0), np.exp(-2.0)])

    def test_monotonic_decreasing(self):
        dx = np.linspace(0, 200, 50)
        k = thermal.crosstalk_kernel(dx, decay_um=30.0)
        assert np.all(np.diff(k) <= 0)

    def test_invalid_decay(self):
        with pytest.raises(ValueError):
            thermal.crosstalk_kernel(np.array([1.0]), decay_um=0.0)

    def test_negative_distance_rejected(self):
        with pytest.raises(ValueError):
            thermal.crosstalk_kernel(np.array([-1.0]), decay_um=10.0)


class TestCouplingMatrix:
    def test_diagonal_zeroed(self):
        layout = thermal.RingLayout.uniform(4, 15.0)
        k = thermal.coupling_matrix(layout, decay_um=20.0)
        assert np.allclose(np.diag(k), 0.0)

    def test_symmetric_for_symmetric_layout(self):
        layout = thermal.RingLayout.uniform(5, 12.0)
        k = thermal.coupling_matrix(layout, decay_um=18.0)
        assert np.allclose(k, k.T)

    def test_matches_kernel_off_diagonal(self):
        layout = thermal.RingLayout((0.0, 10.0))
        k = thermal.coupling_matrix(layout, decay_um=10.0)
        assert k[0, 1] == pytest.approx(np.exp(-1.0))
        assert k[1, 0] == pytest.approx(np.exp(-1.0))

    def test_tighter_pitch_couples_more(self):
        tight = thermal.coupling_matrix(thermal.RingLayout.uniform(2, 5.0), decay_um=20.0)
        loose = thermal.coupling_matrix(thermal.RingLayout.uniform(2, 50.0), decay_um=20.0)
        assert tight[0, 1] > loose[0, 1]


class TestSolveCoupledPowers:
    def test_isolated_rings_match_naive(self):
        # decay_um << pitch: crosstalk vanishes, coupled == naive P = d/eff.
        layout = thermal.RingLayout.uniform(3, 1000.0)
        target = [1.0, 2.0, 3.0]
        eff = 0.25
        result = thermal.solve_coupled_powers(target, layout, decay_um=1.0,
                                               tuning_efficiency_nm_per_mw=eff)
        expected = np.array(target) / eff
        assert np.allclose(result.heater_mw, expected, atol=1e-6)
        assert np.allclose(result.naive_mw, expected)

    def test_two_ring_hand_solved(self):
        # Two rings, symmetric target and layout -> closed form by hand.
        # target = eff * (I + K) @ P, K = [[0, k],[k, 0]], symmetric target d.
        # By symmetry P0 = P1 = p: eff*p*(1+k) = d -> p = d/(eff*(1+k)).
        layout = thermal.RingLayout((0.0, 10.0))
        decay = 10.0
        eff = 0.25
        d = 2.0
        result = thermal.solve_coupled_powers([d, d], layout, decay_um=decay,
                                               tuning_efficiency_nm_per_mw=eff)
        k = np.exp(-1.0)
        expected_p = d / (eff * (1.0 + k))
        assert result.heater_mw[0] == pytest.approx(expected_p, rel=1e-9)
        assert result.heater_mw[1] == pytest.approx(expected_p, rel=1e-9)
        # Verify the fixed point directly: own contribution + neighbor shift = target.
        own_shift = result.heater_mw[0] * eff
        assert own_shift + result.neighbor_shift_nm[0] == pytest.approx(d, rel=1e-9)

    def test_coupled_power_less_than_naive_for_aligned_targets(self):
        # When neighbors need heating in the same direction, crosstalk helps:
        # each ring "borrows" some of its neighbor's heat, so coupled < naive.
        layout = thermal.RingLayout.uniform(4, 8.0)
        target = [3.0, 3.0, 3.0, 3.0]
        result = thermal.solve_coupled_powers(target, layout, decay_um=15.0)
        assert result.total_mw < result.naive_total_mw

    def test_asymmetric_targets_can_increase_power_for_one_ring(self):
        # Ring 0 needs zero shift but sits next to a ring that needs a lot:
        # ring 0 receives unwanted heat and would need negative correction,
        # which is unreachable -> ValueError.
        layout = thermal.RingLayout((0.0, 5.0))
        with pytest.raises(ValueError, match="unreachable"):
            thermal.solve_coupled_powers([0.0, 100.0], layout, decay_um=50.0)

    def test_negative_target_rejected(self):
        layout = thermal.RingLayout.uniform(2, 10.0)
        with pytest.raises(ValueError):
            thermal.solve_coupled_powers([-1.0, 1.0], layout, decay_um=10.0)

    def test_wrong_length_rejected(self):
        layout = thermal.RingLayout.uniform(3, 10.0)
        with pytest.raises(ValueError):
            thermal.solve_coupled_powers([1.0, 2.0], layout, decay_um=10.0)

    def test_invalid_efficiency(self):
        layout = thermal.RingLayout.uniform(2, 10.0)
        with pytest.raises(ValueError):
            thermal.solve_coupled_powers([1.0, 1.0], layout, decay_um=10.0,
                                          tuning_efficiency_nm_per_mw=0.0)

    def test_default_efficiency_matches_wdm(self):
        layout = thermal.RingLayout.uniform(2, 1000.0)
        result = thermal.solve_coupled_powers([1.0, 1.0], layout, decay_um=1.0)
        assert result.naive_mw[0] == pytest.approx(
            wdm.tuning_power_mw(1.0), rel=1e-6
        )

    def test_naive_mw_matches_wdm_per_ring(self):
        # Every ring's naive_mw, not just a symmetric case, should equal
        # wdm.tuning_power_mw of that ring's own target (naive_mw is
        # defined independent of coupling).
        target = [0.3, 1.7, 0.9, 2.4, 1.1]
        layout = thermal.RingLayout.uniform(5, 20.0)
        result = thermal.solve_coupled_powers(target, layout, decay_um=10.0)
        expected = [wdm.tuning_power_mw(d) for d in target]
        assert np.allclose(result.naive_mw, expected, rtol=1e-9)

    def test_realistic_scale_eight_ring_bank(self):
        # Matches examples/08_thermal_crosstalk.py's scale/regime: a dense
        # DWDM bank where crosstalk is non-negligible but still lockable.
        rng = np.random.default_rng(2026)
        n = 8
        target = np.abs(rng.uniform(0.05, 1.5, n))
        layout = thermal.RingLayout.uniform(n, 30.0)
        result = thermal.solve_coupled_powers(target, layout, decay_um=8.0)
        assert len(result.heater_mw) == n
        assert all(p >= 0 for p in result.heater_mw)
        # Forward-check: reconstructing each ring's total shift from the
        # solved powers must reproduce the target exactly.
        eff = wdm.TUNING_EFFICIENCY_NM_PER_MW
        k = thermal.coupling_matrix(layout, 8.0)
        power = np.array(result.heater_mw)
        reconstructed = eff * power + eff * (k @ power)
        assert np.allclose(reconstructed, target, atol=1e-9)

    def test_report_runs(self):
        layout = thermal.RingLayout.uniform(3, 20.0)
        result = thermal.solve_coupled_powers([1.0, 1.5, 2.0], layout, decay_um=10.0)
        text = result.report()
        assert "thermal crosstalk-coupled tuning" in text
        assert "total:" in text

    def test_single_ring_reduces_to_naive(self):
        layout = thermal.RingLayout((0.0,))
        result = thermal.solve_coupled_powers([1.0], layout, decay_um=10.0)
        assert result.heater_mw[0] == pytest.approx(result.naive_mw[0])
        assert result.neighbor_shift_nm[0] == pytest.approx(0.0)

    def test_coincident_positions_raise_singular(self):
        # pitch = 0 -> full coupling -> (I + K) singular for two rings.
        layout = thermal.RingLayout((5.0, 5.0))
        with pytest.raises(ValueError, match="singular"):
            thermal.solve_coupled_powers([1.0, 2.0], layout, decay_um=10.0)

    def test_weak_coupling_matches_screening_bound(self):
        # In the weak-coupling limit, the coupled solve's neighbor_shift_nm
        # should agree with the uncoupled screening bound evaluated at the
        # (near-identical) naive powers.
        layout = thermal.RingLayout.uniform(4, 100.0)
        decay = 5.0
        target = [1.0, 2.0, 1.5, 0.8]
        eff = 0.25
        result = thermal.solve_coupled_powers(target, layout, decay, eff)
        screen = thermal.worst_case_neighbor_shift_nm(
            np.array(result.naive_mw), layout, decay, eff
        )
        assert np.allclose(result.neighbor_shift_nm, screen, atol=1e-6)
        assert np.allclose(result.heater_mw, result.naive_mw, atol=1e-6)


class TestWorstCaseNeighborShift:
    def test_matches_manual_computation(self):
        layout = thermal.RingLayout((0.0, 10.0, 20.0))
        decay = 10.0
        eff = 0.25
        heater = [4.0, 0.0, 0.0]
        shift = thermal.worst_case_neighbor_shift_nm(heater, layout, decay_um=decay,
                                                       tuning_efficiency_nm_per_mw=eff)
        k = thermal.coupling_matrix(layout, decay_um=decay)
        expected = k @ np.array(heater) * eff
        assert np.allclose(shift, expected)
        # Ring 0 sees none of its own heat via crosstalk (diagonal zeroed).
        assert shift[0] == pytest.approx(0.0)
        assert shift[1] > shift[2] > 0.0  # closer ring gets more crosstalk

    def test_zero_heaters_zero_shift(self):
        layout = thermal.RingLayout.uniform(4, 15.0)
        shift = thermal.worst_case_neighbor_shift_nm([0.0] * 4, layout, decay_um=10.0)
        assert np.allclose(shift, 0.0)

    def test_wrong_length_rejected(self):
        layout = thermal.RingLayout.uniform(3, 10.0)
        with pytest.raises(ValueError):
            thermal.worst_case_neighbor_shift_nm([1.0, 2.0], layout, decay_um=10.0)

    def test_invalid_efficiency(self):
        layout = thermal.RingLayout.uniform(2, 10.0)
        with pytest.raises(ValueError):
            thermal.worst_case_neighbor_shift_nm([1.0, 1.0], layout, decay_um=10.0,
                                                   tuning_efficiency_nm_per_mw=-1.0)
