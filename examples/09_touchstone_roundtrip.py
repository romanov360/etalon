#!/usr/bin/env python3
"""SiPhon example 09 — Touchstone round trip: the measured-data on-ramp.

Pretends a foundry PDK or wafer-probe VNA handed back a .s2p file for an
add-drop ring (synthesized here from known ground truth, exported to a
REAL Touchstone file on disk, in DB format like most VNA exports), then
reads that file back exactly as an external tool would produce it, and
calibrates a RingAddDrop model to it with siphon.extract — closing the
loop from "file on disk" to "calibrated SiPhon model" with no hand-built
numpy arrays in between.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CIRC_UM = 200.0
WL0_UM = 1.55
TRUE = {
    "kappa1_power": 0.08,
    "kappa2_power": 0.06,
    "loss_db_per_cm": 3.0,
    "neff0": 2.35,
    "ng": 4.1,
}
NOISE_DB = 0.05  # probe-trace noise, dB rms — see example 05
SEED = 7


def header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    try:
        from siphon.components import RingAddDrop
        from siphon.extract import fit_ring_add_drop
        from siphon import touchstone as ts
    except ImportError as exc:
        print(f"This example needs siphon.components/extract/touchstone ({exc}).")
        return 1

    header("SiPhon 09 — Touchstone file round trip")

    ring = RingAddDrop(circumference_um=CIRC_UM, wl0_um=WL0_UM, **TRUE)
    fsr_um = WL0_UM**2 / (TRUE["ng"] * CIRC_UM)
    wl = np.linspace(WL0_UM - 1.55 * fsr_um, WL0_UM + 1.55 * fsr_um, 601)
    s_full = ring.s_params(wl)  # (n_wl, 4, 4): ports in, through, add, drop

    # A wafer probe only ever gives you the ports actually landed: in and
    # drop, as a 2-port measurement. Slice those two out into a genuine
    # 2-port S-matrix before writing — this is what a real probe station
    # or PDK model export hands back, not the full 4-port device matrix.
    i_in, i_drop = ring.ports.index("in"), ring.ports.index("drop")
    s2 = s_full[:, [i_in, i_drop], :][:, :, [i_in, i_drop]]

    # A real probe trace carries measurement noise. Perturb the magnitude
    # in dB (VNA amplitude noise is roughly additive in log space) and
    # leave phase exact, then write THAT to disk — the file genuinely
    # differs from the noiseless model, same spirit as example 05.
    rng = np.random.default_rng(SEED)
    mag_db_noise = rng.normal(0.0, NOISE_DB, s2.shape)
    s2 = s2 * (10.0 ** (mag_db_noise / 20.0))

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "09_ring_probe.s2p"

    print(f"Device : add-drop ring, L = {CIRC_UM:.0f} um, FSR ~ {fsr_um * 1e3:.2f} nm")
    print(f"Writing a 2-port (in, drop) VNA-style export: {wl.size} points, "
          f"DB format, GHz, R=50 ohm -> {path.name}")
    ts.write_touchstone(path, wl, s2, ports=("in", "drop"), fmt="db", freq_unit="ghz")

    print(f"\nFile on disk: {path.stat().st_size} bytes. First data line:")
    with open(path) as f:
        for line in f:
            if not line.startswith(("!", "#")) and line.strip():
                print(f"  {line.strip()}")
                break
    print("  (freq_GHz  |S11|_dB  ang11  |S21|_dB  ang21  |S12|_dB  ang12  |S22|_dB  ang22)")

    header("Reading it back as an external tool would hand it to us")
    data = ts.read_touchstone(path)
    print(f"Parsed: {data.ports} ports, {data.freq_hz.size} points, "
          f"{data.freq_hz.min() / 1e9:.2f}-{data.freq_hz.max() / 1e9:.2f} GHz, "
          f"R = {data.reference_ohms:.0f} ohm")
    measured_db = data.transmission_db(wl, "in", "drop")
    noiseless_db = 10.0 * np.log10(np.abs(s_full[:, i_drop, i_in]) ** 2)
    rms_vs_noiseless = float(np.sqrt(np.mean((measured_db - noiseless_db) ** 2)))
    print(f"file vs. the noiseless model: {rms_vs_noiseless:.4f} dB rms "
          f"(this IS the injected {NOISE_DB} dB probe noise, not a file-format "
          "error — DB<->RI round-trips to float precision)")

    header("Calibrating a RingAddDrop model to the file (through port unavailable)")
    # Only in->drop is in the file — a realistic single-port-pair probe
    # measurement. fit_ring_add_drop wants both through and drop; since a
    # probe with only a drop tap is common, fit through the general
    # fit_transmission entry point on drop alone instead.
    from siphon.extract import fit_transmission

    x0 = {
        "kappa1_power": 0.09, "kappa2_power": 0.05,
        "loss_db_per_cm": 2.0, "neff0": 2.351, "ng": 4.2,
    }
    fit = fit_transmission(
        lambda **p: RingAddDrop(circumference_um=CIRC_UM, wl0_um=WL0_UM, **p),
        params0=x0,
        wl_um=wl,
        measured=measured_db,
        inport="in",
        outport="drop",
        bounds={
            "kappa1_power": (1e-4, 0.9), "kappa2_power": (1e-4, 0.9),
            "loss_db_per_cm": (0.0, 1e3), "neff0": (1.0, 4.5), "ng": (1.0, 8.0),
        },
    )
    print(f"converged: {fit.success}, residual rms = {fit.residual_rms_db:.4f} dB "
          f"(injected noise floor: {NOISE_DB} dB; drop-only fit, so expect a "
          "looser trade-off between kappa1/kappa2/loss than the through+drop "
          "joint fit in example 05)")
    print()
    print(fit.report())

    print("\nThis is the strategic point of siphon.touchstone: a foundry PDK export,")
    print("a VNA trace, or a wafer-probe measurement all speak this file format —")
    print("the path from 'file a fab handed you' to 'calibrated SiPhon model' is")
    print("now three function calls (read_touchstone -> transmission_db ->")
    print("fit_transmission), the same fitting machinery example 05 uses on")
    print("synthetic data.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
