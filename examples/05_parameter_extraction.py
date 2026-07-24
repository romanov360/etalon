#!/usr/bin/env python3
"""SiPhon example 05 — parameter extraction from a wafer probe trace.

Pretends a probe station measured the through- and drop-port power spectra
of an add-drop microring (synthesized here from known ground truth plus
0.05 dB rms Gaussian noise, spanning ~3 FSRs), then calibrates a
RingAddDrop model to the trace with siphon.extract.fit_ring_add_drop. The
script prints a true-vs-recovered parameter table with the fit's residual
rms against the injected noise floor, and saves an overlay plot (measured
points, fitted curves, residuals) to examples/out/.

The fit is architecture-level model calibration, not metrology: neff0 is
recovered only modulo the resonance order (the circumference is fixed from
layout), and ng is only identifiable because the trace spans several FSRs.

Units: um for wavelength/geometry, dB for transmission, dB/cm for loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Design/ground truth ------------------------------------------------------
CIRC_UM = 200.0          # ring circumference, known from layout (fixed in fit)
WL0_UM = 1.55            # dispersion expansion wavelength
TRUE = {
    "kappa1_power": 0.08,
    "kappa2_power": 0.06,
    "loss_db_per_cm": 3.0,
    "neff0": 2.35,
    "ng": 4.1,
}
NOISE_DB = 0.05          # probe-trace noise, dB rms
SEED = 7

# Chart palette (validated categorical slots, light mode)
SERIES = ["#2a78d6", "#eb6834"]      # through, drop
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"


def header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def table(headers: list[str], rows: list[list[str]], widths: list[int]) -> None:
    print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))


def synthesize_probe_trace(RingAddDrop):
    """Noisy through/drop dB spectra over ~3 FSRs of the ground-truth ring."""
    fsr_um = WL0_UM**2 / (TRUE["ng"] * CIRC_UM)          # FSR = wl^2/(ng L)
    span = 3.05 * fsr_um
    wl = np.linspace(WL0_UM - span / 2, WL0_UM + span / 2, 1501)
    ring = RingAddDrop(circumference_um=CIRC_UM, wl0_um=WL0_UM, **TRUE)
    s = ring.s_params(wl)
    i_in = ring.ports.index("in")
    thru_db = 10 * np.log10(np.abs(s[:, ring.ports.index("through"), i_in]) ** 2)
    drop_db = 10 * np.log10(np.abs(s[:, ring.ports.index("drop"), i_in]) ** 2)
    rng = np.random.default_rng(SEED)
    thru_db = thru_db + rng.normal(0.0, NOISE_DB, wl.size)
    drop_db = drop_db + rng.normal(0.0, NOISE_DB, wl.size)
    return wl, thru_db, drop_db, fsr_um


def model_spectra_db(RingAddDrop, params, wl):
    ring = RingAddDrop(circumference_um=CIRC_UM, wl0_um=WL0_UM, **params)
    s = ring.s_params(wl)
    thru = 10 * np.log10(np.maximum(np.abs(s[:, 1, 0]) ** 2, 1e-16))
    drop = 10 * np.log10(np.maximum(np.abs(s[:, 3, 0]) ** 2, 1e-16))
    return thru, drop


def maybe_plot(wl, thru_db, drop_db, fit_thru, fit_drop) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] matplotlib not installed; skipping the overlay figure")
        return

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)

    wl_nm = wl * 1e3
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(9, 5.6), height_ratios=[2.6, 1],
        sharex=True, facecolor=SURFACE,
    )
    for a in (ax, axr):
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, lw=0.8)
        for s in a.spines.values():
            s.set_color(GRID)
        a.tick_params(colors=INK_2)

    # measured points (subsampled so the markers stay readable) + fitted lines
    step = 12
    for data, fitted, color, name in (
        (thru_db, fit_thru, SERIES[0], "through"),
        (drop_db, fit_drop, SERIES[1], "drop"),
    ):
        ax.plot(
            wl_nm[::step], data[::step], "o", ms=3, mew=0, alpha=0.55,
            color=color, label=f"{name} (measured)",
        )
        ax.plot(wl_nm, fitted, color=color, lw=2.0, label=f"{name} (fit)")
    # direct labels near the right edge, in ink (identity also in the legend)
    ax.annotate("through", (wl_nm[-1], fit_thru[-1] + 1.5), ha="right",
                color=INK, fontsize=9)
    ax.annotate("drop", (wl_nm[-1], fit_drop[-1] + 1.5), ha="right",
                color=INK, fontsize=9)
    ax.set_ylabel("transmission (dB)", color=INK_2)
    ax.set_title("Add-drop ring: probe trace vs extracted model", color=INK)
    ax.legend(loc="center right", fontsize=8.5, framealpha=0.9)

    axr.plot(wl_nm, thru_db - fit_thru, color=SERIES[0], lw=1.0)
    axr.plot(wl_nm, drop_db - fit_drop, color=SERIES[1], lw=1.0)
    axr.axhline(0.0, color=INK_2, lw=0.8)
    axr.set_ylim(-4 * NOISE_DB, 4 * NOISE_DB)
    axr.set_xlabel("wavelength (nm)", color=INK_2)
    axr.set_ylabel("residual (dB)", color=INK_2)

    fig.tight_layout()
    path = out_dir / "05_parameter_extraction.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] saved {path}")


def main() -> int:
    try:
        from siphon.components import RingAddDrop
        from siphon.extract import fit_ring_add_drop
    except ImportError as exc:
        print("This example needs siphon.components and siphon.extract, which")
        print(f"are not built in this checkout yet ({exc}).")
        return 1

    header("SiPhon 05 — model calibration against a (synthetic) probe trace")
    wl, thru_db, drop_db, fsr_um = synthesize_probe_trace(RingAddDrop)
    print(f"Device : add-drop ring, L = {CIRC_UM:.0f} um (fixed from layout), "
          f"FSR ~ {fsr_um * 1e3:.2f} nm")
    print(f"Trace  : {wl.size} points, {wl[0] * 1e3:.1f}-{wl[-1] * 1e3:.1f} nm "
          f"(~3 FSRs), {NOISE_DB} dB rms noise, seed {SEED}")

    # Start from the *design* values, not the truth. Starting from the
    # designed kappa1 > kappa2 asymmetry matters: at 0.05 dB noise the
    # (kappa1, kappa2, loss) trade-off is a flat cost valley, and the fit
    # returns the valley point nearest x0 (see fit_ring_add_drop docstring).
    x0 = {
        "kappa1_power": 0.09,
        "kappa2_power": 0.05,
        "loss_db_per_cm": 2.0,
        "neff0": 2.351,   # design estimate; comb realigned by the coarse scan
        "ng": 4.2,
    }
    fit = fit_ring_add_drop(
        wl, thru_db, drop_db, x0, circumference_um=CIRC_UM, wl0_um=WL0_UM
    )

    header("True vs recovered parameters")
    rows = []
    for key in TRUE:
        t, x, r = TRUE[key], x0[key], fit.params[key]
        err = 100.0 * (r - t) / t
        rows.append([key, f"{t:.6g}", f"{x:.6g}", f"{r:.6g}", f"{err:+.3f}"])
    table(["parameter", "true", "start", "recovered", "err (%)"],
          rows, [15, 10, 10, 11, 8])
    print(f"\nresidual rms = {fit.residual_rms_db:.4f} dB "
          f"(injected noise floor: {NOISE_DB} dB) — "
          f"{'at the floor: the model explains the trace' if fit.residual_rms_db < 1.3 * NOISE_DB else 'ABOVE the floor: model/data mismatch'}")
    print(f"converged: {fit.success} after {fit.nfev} model evaluations")
    print()
    print(fit.report())

    print("\nIdentifiability notes (see siphon.extract docstrings):")
    print(f"  * neff0 is recovered modulo the resonance order "
          f"(wl0/L = {WL0_UM / CIRC_UM * 1e3:.2f} m-units per order);")
    print("    the layout circumference is held fixed because neff0*L is all")
    print("    the spectrum sees.")
    print("  * ng comes from the FSR — this works only because the trace spans")
    print("    ~3 FSRs; a single-resonance scan would return the initial guess.")
    print("  * (kappa1, kappa2, loss) share a flat cost valley at this noise")
    print("    level; starting from the design asymmetry picks the right end.")

    fit_thru, fit_drop = model_spectra_db(RingAddDrop, fit.params, wl)
    maybe_plot(wl, thru_db, drop_db, fit_thru, fit_drop)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
