"""Regression tests for the adversarial review of the 2026-07-24 extensions.

Findings and transcripts: docs/research/raw/workflow-validation-review/.
R1 (high): fdmode returned pad-dependent slab-continuum artifacts for ribs.
R2 (med):  independent RIN + shot dB penalties are jointly OPTIMISTIC; the
           budget now solves the joint quadratic and books the interaction.
R3 (med):  extract died with the model's raw ValueError on unbounded fits.
R4 (med):  CommonDifferential truncation spuriously rejected valid setups.
"""

import math
import warnings

import numpy as np
import pytest

from siphon import extract, link, montecarlo as mc
from siphon.components import DirectionalCoupler
from siphon.fdmode import solve_modes
from siphon.waveguide import slab_neffs


# --- R2: joint RIN + shot noise solve ----------------------------------------


def test_joint_noise_matches_independent_bisection():
    # Solve u R/(2(M-1)) = q sqrt(i_n^2 + 2 q_e (R k u + I_d) f_n
    #                             + (R k u)^2 RIN f_n) from scratch.
    lb = link.preset_pluggable_dr4()
    sig = lb.signaling
    q, m = sig.q_factor, sig.levels
    r = lb.photodiode.responsivity_a_per_w
    er = 10.0 ** (lb.modulator.extinction_ratio_db / 10.0)
    k = er / (er - 1.0)
    f_n = 0.75 * sig.rate_gbd * 1e9
    i_n = lb.tia.input_noise_pa_per_sqrt_hz * 1e-12 * math.sqrt(f_n)
    q_e = 1.602176634e-19
    i_d = lb.photodiode.dark_current_na * 1e-9
    rin = 10.0 ** (lb.laser.rin_db_hz / 10.0)

    def gap(u):
        var = i_n**2 + 2.0 * q_e * (r * k * u + i_d) * f_n + (r * k * u) ** 2 * rin * f_n
        return u * r / (2.0 * (m - 1)) - q * math.sqrt(var)

    u0 = 2.0 * q * (m - 1) * i_n / r
    lo, hi = u0, 100.0 * u0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    joint_db = 10.0 * math.log10(lo / u0)
    booked = lb.rin_penalty_db + lb.shot_penalty_db + lb.noise_interaction_db
    assert booked == pytest.approx(joint_db, abs=1e-6)


def test_noise_interaction_is_nonnegative_and_matches_review_numbers():
    dr4, cpo = link.preset_pluggable_dr4(), link.preset_cpo_optical_io()
    # review measured ~0.064 dB optimism at the DR4 point, ~0.0002 at CPO
    assert dr4.noise_interaction_db == pytest.approx(0.064, abs=0.01)
    assert 0.0 <= cpo.noise_interaction_db < 0.001
    assert "RIN x shot interaction" in dr4.report()
    # zero RIN -> zero interaction (joint solve degenerates to shot quadratic)
    quiet = link.LinkBudget(
        laser=link.Laser(power_dbm=6.0, wpe=0.1, rin_db_hz=-math.inf),
        modulator=dr4.modulator,
        path=list(dr4.path),
        photodiode=dr4.photodiode,
        tia=dr4.tia,
        signaling=dr4.signaling,
    )
    assert quiet.noise_interaction_db == pytest.approx(0.0, abs=1e-12)


# --- R1: fdmode rib slab-continuum filtering ----------------------------------


def test_rib_modes_are_pad_invariant_and_above_slab_line():
    kwargs = dict(slab_um=0.09, dx_um=0.03, n_modes=5)
    a = solve_modes(0.5, 0.22, 1.55, pad_um=1.5, **kwargs)
    b = solve_modes(0.5, 0.22, 1.55, pad_um=2.5, **kwargs)
    assert len(a) == len(b) >= 1
    for ma, mb in zip(a, b):
        assert ma.neff == pytest.approx(mb.neff, abs=1e-5)
    # everything returned must lie above the residual slab's TE line —
    # below it is lateral radiation discretized by the Dirichlet wall
    from siphon import materials

    n_si = float(materials.index("si", 1.55))
    n_ox = float(materials.index("sio2", 1.55))
    slab_line = slab_neffs(n_si, n_ox, n_ox, 0.09, 1.55, "TE")[0]
    assert all(m.neff > slab_line for m in a)


# --- R3: extract fails loudly with bounds advice -------------------------------


def test_unbounded_fit_raises_informative_error():
    wl = np.linspace(1.54, 1.56, 11)
    with pytest.raises(ValueError, match="bounds"):
        extract.fit_transmission(
            DirectionalCoupler,
            {"coupling": 0.9},
            wl,
            np.full(11, 0.05),  # +0.05 dB "measured" pushes coupling past 1
            inport="in0",
            outport="out1",
        )


def test_fit_ring_add_drop_fixed_parameter():
    from siphon.components import RingAddDrop

    true = dict(
        circumference_um=200.0,
        kappa1_power=0.06,
        kappa2_power=0.05,
        loss_db_per_cm=2.5,
        neff0=2.35,
        ng=4.3,
    )
    ring = RingAddDrop(**true, wl0_um=1.55)
    wl = np.linspace(1.54, 1.56, 1200)
    s = ring.s_params(wl)
    thru = 10.0 * np.log10(np.abs(s[:, 1, 0]) ** 2)
    drop = 10.0 * np.log10(np.abs(s[:, 3, 0]) ** 2)
    res = extract.fit_ring_add_drop(
        wl,
        thru,
        drop,
        x0={"kappa1_power": 0.08, "kappa2_power": 0.04},
        circumference_um=200.0,
        fixed={"loss_db_per_cm": 2.5},
    )
    assert "loss_db_per_cm" not in res.params  # held, not fitted
    assert res.params["kappa1_power"] == pytest.approx(0.06, rel=0.02)
    assert res.params["kappa2_power"] == pytest.approx(0.05, rel=0.02)
    with pytest.raises(ValueError, match="both x0 and fixed"):
        extract.fit_ring_add_drop(
            wl, thru, drop,
            x0={"loss_db_per_cm": 2.0},
            circumference_um=200.0,
            fixed={"loss_db_per_cm": 2.5},
        )


# --- R4: CommonDifferential truncation semantics --------------------------------


def test_truncation_no_longer_rejects_valid_configs():
    rng = np.random.default_rng(7)
    # sigma_common = 0 with bounds excluding the mean: previously raised,
    # must now match the plain truncated Normal marginal
    cd = mc.CommonDifferential(mean=0.0, sigma_common=0.0, sigma_diff=1.0, low=2.0, high=3.0)
    x = cd.sample(4000, 4, rng)
    assert np.all((x >= 2.0) & (x <= 3.0))
    ref = mc.Normal("x", 0.0, 1.0, low=2.0, high=3.0).sample(16000, np.random.default_rng(8))
    assert x.mean() == pytest.approx(ref.mean(), abs=0.02)
    # the reviewer's second repro: 20% of mass in-bounds, previously raised
    cd2 = mc.CommonDifferential(mean=0.0, sigma_common=0.3, sigma_diff=2.0, low=1.5, high=4.0)
    y = cd2.sample(1000, 8, rng)
    assert np.all((y >= 1.5) & (y <= 4.0))


def test_truncation_preserves_common_mode_correlation():
    rng = np.random.default_rng(11)
    # bounds far away: correlation must equal sigma_c^2/(sigma_c^2+sigma_d^2)
    cd = mc.CommonDifferential(mean=0.0, sigma_common=1.0, sigma_diff=1.0, low=-50.0, high=50.0)
    x = cd.sample(20000, 2, rng)
    corr = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
    assert corr == pytest.approx(0.5, abs=0.03)
    # sigma_diff = 0 with bounds: exact truncated normal, identical across lanes
    cd0 = mc.CommonDifferential(mean=0.0, sigma_common=1.0, sigma_diff=0.0, low=0.5, high=2.0)
    z = cd0.sample(500, 3, rng)
    assert np.all(z[:, 0] == z[:, 1])
    assert np.all((z >= 0.5) & (z <= 2.0))
    # degenerate constant outside bounds still raises
    with pytest.raises(ValueError, match="outside"):
        mc.CommonDifferential(mean=0.0, sigma_common=0.0, sigma_diff=0.0, low=1.0, high=2.0).sample(
            10, 2, rng
        )
