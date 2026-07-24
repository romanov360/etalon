"""Tests for siphon.wdm (channel plans, tuning power, crosstalk)."""

import numpy as np
import pytest

from siphon import wdm
from siphon.wdm import ChannelPlan


C_UM_GHZ = 2.99792458e5


class TestChannelPlan:
    def test_cwdm4_wavelengths_exact(self):
        plan = ChannelPlan.cwdm4()
        assert plan.centers_um == (1.271, 1.291, 1.311, 1.331)
        assert np.allclose(plan.spacing_nm(), 20.0)

    def test_lr4_wavelengths(self):
        plan = ChannelPlan.lr4()
        assert plan.centers_um == (1.29556, 1.30005, 1.30458, 1.30914)
        # LAN-WDM is an 800 GHz frequency grid
        f_ghz = plan.as_frequencies_thz() * 1e3
        assert np.allclose(np.diff(f_ghz), -800.0, atol=2.0)

    def test_dwdm_100ghz_spacing_near_1550(self):
        plan = ChannelPlan.dwdm(1.55, 100.0, 8)
        # d(lambda) = spacing * wl^2 / c = 100 * 1.55^2 / 2.99792458e5 um
        expected_nm = 100.0 * 1.55**2 / C_UM_GHZ * 1e3
        assert expected_nm == pytest.approx(0.8013, abs=1e-3)
        assert np.allclose(plan.spacing_nm(), expected_nm, rtol=1e-2)

    def test_dwdm_centered_on_frequency(self):
        plan = ChannelPlan.dwdm(1.55, 100.0, 5)
        f_thz = plan.as_frequencies_thz()
        f_center = C_UM_GHZ / 1.55 * 1e-3
        # odd channel count: middle channel sits exactly on the center frequency
        assert sorted(f_thz)[2] == pytest.approx(f_center, rel=1e-12)
        assert np.allclose(np.abs(np.diff(f_thz)), 0.1, rtol=1e-12)

    def test_frequency_wavelength_round_trip(self):
        plan = ChannelPlan.cwdm4()
        wl_back = C_UM_GHZ / (plan.as_frequencies_thz() * 1e3)
        assert np.allclose(wl_back, plan.centers_um, rtol=1e-14)

    def test_sorted_and_validated(self):
        plan = ChannelPlan((1.331, 1.271))
        assert plan.centers_um == (1.271, 1.331)
        with pytest.raises(ValueError):
            ChannelPlan(())
        with pytest.raises(ValueError):
            ChannelPlan((1.31, -1.29))

    def test_span_nm(self):
        assert ChannelPlan.cwdm4().span_nm() == pytest.approx(60.0)


class TestRingBankLimit:
    def test_channel_count_arithmetic(self):
        # 16 nm FSR, 1.6 nm spacing -> 10 slots, minus 1 guard = 9
        assert wdm.ring_bank_channel_count_limit(16.0, 1.6) == 9
        assert wdm.ring_bank_channel_count_limit(16.0, 1.6, guard_channels=0) == 10
        assert wdm.ring_bank_channel_count_limit(16.0, 1.6, guard_channels=2) == 8

    def test_clips_at_zero(self):
        assert wdm.ring_bank_channel_count_limit(1.0, 2.0) == 0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            wdm.ring_bank_channel_count_limit(-1.0, 1.0)
        with pytest.raises(ValueError):
            wdm.ring_bank_channel_count_limit(16.0, 1.6, guard_channels=-1)


class TestThermalTuning:
    def test_tuning_power_heat_only_half_fsr(self):
        # heat-only heaters red-shift to the next order: mean detune FSR/2.
        # FSR = 16 nm -> mean detune 8 nm; at 0.25 nm/mW that is 32 mW
        assert wdm.tuning_power_mw(8.0, 0.25) == pytest.approx(32.0)
        assert wdm.expected_tuning_power_mw(16.0, 0.25) == pytest.approx(32.0)

    def test_expected_tuning_bidirectional_is_quarter_fsr(self):
        fsr = 12.8
        assert wdm.expected_tuning_power_mw(fsr) == pytest.approx(
            wdm.tuning_power_mw(fsr / 2.0)
        )
        assert wdm.expected_tuning_power_mw(fsr, bidirectional=True) == pytest.approx(
            wdm.tuning_power_mw(fsr / 4.0)
        )

    def test_detune_sign_ignored(self):
        assert wdm.tuning_power_mw(-2.0) == wdm.tuning_power_mw(2.0)

    def test_default_efficiency(self):
        assert wdm.TUNING_EFFICIENCY_NM_PER_MW == 0.25
        assert wdm.tuning_power_mw(0.25) == pytest.approx(1.0)

    def test_invalid_efficiency(self):
        with pytest.raises(ValueError):
            wdm.tuning_power_mw(1.0, 0.0)


class TestResonanceShift:
    def test_known_value_at_1550(self):
        # dlambda/dT = wl * (dneff/dT)/ng = 1550 nm * 1.581e-4 / 4.2 = 0.0583 nm/K
        shift = wdm.resonance_shift_nm_per_k(1.55, 4.2)
        assert shift == pytest.approx(1550.0 * 1.86e-4 * 0.85 / 4.2, rel=1e-12)
        assert 0.04 < shift < 0.09  # typical Si microring: ~50-80 pm/K

    def test_explicit_dneff_dt(self):
        assert wdm.resonance_shift_nm_per_k(1.3, 4.0, dneff_dT=2.0e-4) == pytest.approx(
            1300.0 * 2.0e-4 / 4.0
        )

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            wdm.resonance_shift_nm_per_k(-1.55, 4.2)
        with pytest.raises(ValueError):
            wdm.resonance_shift_nm_per_k(1.55, 0.0)


class TestCrosstalk:
    def test_aggregation_known_value(self):
        # 3 aggressors at -30 dB each: 10 log10(3e-3) = -25.229 dB
        xt = wdm.aggregate_crosstalk_db(30.0, 3)
        assert xt == pytest.approx(-25.229, abs=1e-3)

    def test_single_aggressor_is_isolation(self):
        assert wdm.aggregate_crosstalk_db(25.0, 1) == pytest.approx(-25.0)

    def test_doubling_aggressors_adds_3db(self):
        xt1 = wdm.aggregate_crosstalk_db(30.0, 4)
        xt2 = wdm.aggregate_crosstalk_db(30.0, 8)
        assert xt2 - xt1 == pytest.approx(10.0 * np.log10(2.0))

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            wdm.aggregate_crosstalk_db(-3.0, 2)
        with pytest.raises(ValueError):
            wdm.aggregate_crosstalk_db(30.0, 0)


class TestLaserGridCheck:
    def test_fits_comfortably(self):
        plan = ChannelPlan.lr4()  # span ~13.6 nm
        wdm.laser_grid_check(plan, ring_fsr_nm=20.0)  # no warning, no raise

    def test_raises_when_span_exceeds_fsr(self):
        plan = ChannelPlan.cwdm4()  # span 60 nm
        with pytest.raises(ValueError, match="exceeds ring FSR"):
            wdm.laser_grid_check(plan, ring_fsr_nm=50.0)

    def test_warns_when_span_near_fsr(self):
        plan = ChannelPlan.cwdm4()  # span 60 nm; 60/62 > 0.9
        with pytest.warns(UserWarning, match="guard band"):
            wdm.laser_grid_check(plan, ring_fsr_nm=62.0)

    def test_invalid_fsr(self):
        with pytest.raises(ValueError):
            wdm.laser_grid_check(ChannelPlan.cwdm4(), ring_fsr_nm=0.0)
