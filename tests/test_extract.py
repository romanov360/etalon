"""Tests for siphon.extract — parameter recovery on synthetic spectra.

Physics anchoring: spectra are synthesized from known ground-truth
parameters, corrupted with seeded Gaussian noise, and the fit is required
to recover the truth (an independent identification round-trip, not a
comparison against the fitter's own output). The DirectionalCoupler case
also has closed-form targets (|S|^2 = coupling * 10^(-loss/10)).
"""

import numpy as np
import pytest

from siphon.circuit import Circuit
from siphon.components import DirectionalCoupler, RingAddDrop, Straight
from siphon.extract import FitResult, fit_ring_add_drop, fit_transmission

# --- ground truth ring (shared by several tests) ----------------------------

TRUE = {
    "kappa1_power": 0.08,
    "kappa2_power": 0.06,
    "loss_db_per_cm": 3.0,
    "neff0": 2.35,
    "ng": 4.1,
}
CIRC_UM = 200.0
WL0 = 1.55
NOISE_DB = 0.05
# FSR = wl^2 / (ng * L) ~ 2.93 nm; this grid spans ~3 FSRs around wl0.
WL = np.linspace(1.5455, 1.5545, 1501)


def synth_ring_spectra(seed=42, noise_db=NOISE_DB):
    """Noisy through/drop dB spectra of the ground-truth ring."""
    ring = RingAddDrop(circumference_um=CIRC_UM, wl0_um=WL0, **TRUE)
    s = ring.s_params(WL)
    i_in = ring.ports.index("in")
    thru = 10 * np.log10(np.abs(s[:, ring.ports.index("through"), i_in]) ** 2)
    drop = 10 * np.log10(np.abs(s[:, ring.ports.index("drop"), i_in]) ** 2)
    rng = np.random.default_rng(seed)
    return (
        thru + rng.normal(0.0, noise_db, WL.size),
        drop + rng.normal(0.0, noise_db, WL.size),
    )


# --- fit_ring_add_drop: parameter recovery -----------------------------------


@pytest.fixture(scope="module")
def ring_fit():
    thru_db, drop_db = synth_ring_spectra()
    x0 = {  # perturbed starting point, well off the truth
        "kappa1_power": 0.12,
        "kappa2_power": 0.04,
        "loss_db_per_cm": 5.0,
        "neff0": TRUE["neff0"] + 3e-4,  # ~40% of a resonance-order spacing
        "ng": TRUE["ng"] * 1.03,
    }
    return fit_ring_add_drop(
        WL, thru_db, drop_db, x0, circumference_um=CIRC_UM, wl0_um=WL0
    )


def test_ring_recovery_couplings_and_loss(ring_fit):
    p = ring_fit.params
    assert ring_fit.success
    assert p["kappa1_power"] == pytest.approx(TRUE["kappa1_power"], rel=0.03)
    assert p["kappa2_power"] == pytest.approx(TRUE["kappa2_power"], rel=0.03)
    assert p["loss_db_per_cm"] == pytest.approx(TRUE["loss_db_per_cm"], rel=0.05)


def test_ring_recovery_indices(ring_fit):
    p = ring_fit.params
    assert p["ng"] == pytest.approx(TRUE["ng"], rel=0.01)
    # neff0 is identifiable modulo the resonance order wl0/L; the fit starts
    # near truth so it must land on the true order, far tighter than 1 order.
    order = WL0 / CIRC_UM  # 7.75e-3
    assert abs(p["neff0"] - TRUE["neff0"]) < 0.02 * order


def test_ring_refit_residual_at_noise_floor(ring_fit):
    # rms misfit should approach the injected 0.05 dB noise (chi^2 ~ 1).
    assert 0.7 * NOISE_DB < ring_fit.residual_rms_db < 1.3 * NOISE_DB


def test_ring_noiseless_fit_is_essentially_exact():
    ring = RingAddDrop(circumference_um=CIRC_UM, wl0_um=WL0, **TRUE)
    s = ring.s_params(WL)
    thru = 10 * np.log10(np.abs(s[:, 1, 0]) ** 2)
    drop = 10 * np.log10(np.abs(s[:, 3, 0]) ** 2)
    fit = fit_ring_add_drop(
        WL,
        thru,
        drop,
        {**TRUE, "kappa1_power": 0.1, "loss_db_per_cm": 4.0},
        circumference_um=CIRC_UM,
        wl0_um=WL0,
    )
    for key, val in TRUE.items():
        assert fit.params[key] == pytest.approx(val, rel=1e-4), key
    assert fit.residual_rms_db < 1e-5


# --- fit_transmission: generic component -------------------------------------


def test_fit_directional_coupler_power_db():
    # cross power = coupling * 10^(-loss/10): closed form, wavelength-flat.
    wl = np.linspace(1.5, 1.6, 51)
    true_c, loss_db = 0.3, 0.5
    measured = np.full(wl.size, 10 * np.log10(true_c) - loss_db)
    fit = fit_transmission(
        DirectionalCoupler,
        {"coupling": 0.6},
        wl,
        measured,
        inport="in0",
        outport="out1",
        bounds={"coupling": (1e-4, 0.999)},
        loss_db=loss_db,  # fixed, not fitted
    )
    assert fit.success
    assert fit.params["coupling"] == pytest.approx(true_c, rel=1e-6)
    assert fit.residual_rms_db < 1e-8


def test_fit_directional_coupler_field_mag_domain():
    wl = np.linspace(1.5, 1.6, 21)
    true_c = 0.25
    measured = np.full(wl.size, np.sqrt(true_c))  # |S31| = sqrt(kappa)
    fit = fit_transmission(
        DirectionalCoupler,
        {"coupling": 0.5},
        wl,
        measured,
        inport="in0",
        outport="out1",
        domain="field_mag",
    )
    assert fit.params["coupling"] == pytest.approx(true_c, rel=1e-6)


def test_fit_transmission_on_circuit():
    # loss_db_total = loss_db_per_cm * L * 1e-4; flat dB spectrum.
    wl = np.linspace(1.54, 1.56, 41)
    true_loss = 2.5
    length_um = 5000.0
    measured = np.full(wl.size, -true_loss * length_um * 1e-4)

    def build(loss_db_per_cm):
        c = Circuit()
        c.add("wg", Straight(length_um, 2.4, 4.2, loss_db_per_cm))
        c.expose("a", ("wg", "in"))
        c.expose("b", ("wg", "out"))
        return c

    fit = fit_transmission(
        build, {"loss_db_per_cm": 1.0}, wl, measured, inport="a", outport="b"
    )
    assert fit.params["loss_db_per_cm"] == pytest.approx(true_loss, rel=1e-6)


def test_fit_transmission_multi_path_dict():
    # Joint through+cross fit of one coupler; solution must satisfy both.
    wl = np.linspace(1.5, 1.6, 31)
    true_c = 0.2
    measured = {
        ("in0", "out0"): np.full(wl.size, 10 * np.log10(1 - true_c)),
        ("in0", "out1"): np.full(wl.size, 10 * np.log10(true_c)),
    }
    fit = fit_transmission(DirectionalCoupler, {"coupling": 0.5}, wl, measured)
    assert fit.params["coupling"] == pytest.approx(true_c, rel=1e-6)


def test_fit_result_report_lists_params():
    fit = FitResult(
        params={"coupling": 0.3},
        cost=1.0,
        residual_rms_db=0.05,
        success=True,
        nfev=7,
        message="converged",
    )
    text = fit.report()
    assert "coupling" in text and "0.3" in text and "converged" in text


# --- error paths --------------------------------------------------------------


WL_SHORT = np.linspace(1.5, 1.6, 11)
MEAS_OK = np.zeros(WL_SHORT.size)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        fit_transmission(
            DirectionalCoupler,
            {"coupling": 0.5},
            WL_SHORT,
            np.zeros(WL_SHORT.size + 1),
            inport="in0",
            outport="out1",
        )


def test_unknown_component_port_raises():
    with pytest.raises(ValueError, match="unknown port"):
        fit_transmission(
            DirectionalCoupler,
            {"coupling": 0.5},
            WL_SHORT,
            MEAS_OK,
            inport="in0",
            outport="nope",
        )


def test_unknown_circuit_port_raises():
    def build(coupling):
        c = Circuit()
        c.add("dc", DirectionalCoupler(coupling))
        c.expose("a", ("dc", "in0"))
        c.expose("b", ("dc", "out1"))
        c.expose("c", ("dc", "in1"))
        c.expose("d", ("dc", "out0"))
        return c

    with pytest.raises(ValueError, match="external port"):
        fit_transmission(
            build, {"coupling": 0.5}, WL_SHORT, MEAS_OK, inport="a", outport="zz"
        )


def test_bounds_key_mismatch_raises():
    with pytest.raises(ValueError, match="not in params0"):
        fit_transmission(
            DirectionalCoupler,
            {"coupling": 0.5},
            WL_SHORT,
            MEAS_OK,
            inport="in0",
            outport="out1",
            bounds={"loss_db": (0, 1)},
        )


def test_x0_outside_bounds_raises():
    with pytest.raises(ValueError, match="outside bounds"):
        fit_transmission(
            DirectionalCoupler,
            {"coupling": 0.5},
            WL_SHORT,
            MEAS_OK,
            inport="in0",
            outport="out1",
            bounds={"coupling": (0.6, 0.9)},
        )


def test_params0_fixed_collision_raises():
    with pytest.raises(ValueError, match="both params0 and fixed"):
        fit_transmission(
            DirectionalCoupler,
            {"coupling": 0.5},
            WL_SHORT,
            MEAS_OK,
            inport="in0",
            outport="out1",
            coupling=0.4,
        )


def test_bad_domain_raises():
    with pytest.raises(ValueError, match="domain"):
        fit_transmission(
            DirectionalCoupler,
            {"coupling": 0.5},
            WL_SHORT,
            MEAS_OK,
            inport="in0",
            outport="out1",
            domain="watts",
        )


def test_empty_params0_raises():
    with pytest.raises(ValueError, match="params0"):
        fit_transmission(
            DirectionalCoupler, {}, WL_SHORT, MEAS_OK, inport="in0", outport="out1"
        )


def test_dict_measured_with_ports_raises():
    with pytest.raises(ValueError, match="do not"):
        fit_transmission(
            DirectionalCoupler,
            {"coupling": 0.5},
            WL_SHORT,
            {("in0", "out1"): MEAS_OK},
            inport="in0",
            outport="out1",
        )


def test_array_measured_without_ports_raises():
    with pytest.raises(ValueError, match="required"):
        fit_transmission(DirectionalCoupler, {"coupling": 0.5}, WL_SHORT, MEAS_OK)


def test_bad_wavelength_grid_raises():
    with pytest.raises(ValueError, match="wl_um"):
        fit_transmission(
            DirectionalCoupler,
            {"coupling": 0.5},
            np.array([1.55]),
            np.zeros(1),
            inport="in0",
            outport="out1",
        )
    with pytest.raises(ValueError, match="positive"):
        fit_transmission(
            DirectionalCoupler,
            {"coupling": 0.5},
            np.array([-1.5, 1.6]),
            np.zeros(2),
            inport="in0",
            outport="out1",
        )


def test_ring_bad_arguments_raise():
    thru = np.zeros(WL_SHORT.size)
    with pytest.raises(ValueError, match="circumference_um"):
        fit_ring_add_drop(WL_SHORT, thru, thru, circumference_um=-1.0)
    with pytest.raises(ValueError, match="x0 keys"):
        fit_ring_add_drop(
            WL_SHORT, thru, thru, {"radius": 5.0}, circumference_um=100.0
        )
    with pytest.raises(ValueError, match="bounds keys"):
        fit_ring_add_drop(
            WL_SHORT, thru, thru, circumference_um=100.0, bounds={"foo": (0, 1)}
        )
