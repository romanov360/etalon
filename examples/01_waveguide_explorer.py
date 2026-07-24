#!/usr/bin/env python3
"""SiPhon example 01 — waveguide explorer.

Sweeps the width of a standard 220 nm SOI strip waveguide at 1550 nm and
prints effective-index / group-index tables for the quasi-TE and quasi-TM
fundamental modes, locates the single-mode boundary with the effective index
method, and tabulates the bend-loss heuristic.

Uses only the core modules (siphon.materials, siphon.waveguide), so it runs
on a bare checkout. Units: um for wavelength/geometry, dB for loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from siphon import materials
from siphon.waveguide import Waveguide, bend_loss_db_per_90deg, slab_neffs

WL_UM = 1.55
HEIGHT_UM = 0.220  # standard SOI device layer

# Chart palette (validated categorical slots, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
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
    line = "  ".join(h.rjust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))


def lateral_mode_count(width_um: float, wl_um: float, mode: str = "TE") -> int:
    """Number of laterally guided modes of a strip, via the same EIM as Waveguide."""
    n_core = materials.index("si", wl_um)
    n_clad = materials.index("sio2", wl_um)
    vert_pol = "TE" if mode == "TE" else "TM"
    horiz_pol = "TM" if mode == "TE" else "TE"
    vert = slab_neffs(n_core, n_clad, n_clad, HEIGHT_UM, wl_um, vert_pol)
    if not vert:
        return 0
    return len(slab_neffs(vert[0], n_clad, n_clad, width_um, wl_um, horiz_pol))


def sweep(widths: np.ndarray) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {
        "neff_te": [], "ng_te": [], "neff_tm": [], "ng_tm": []
    }
    for w in widths:
        wg = Waveguide(width_um=float(w), height_um=HEIGHT_UM)
        for mode in ("TE", "TM"):
            try:
                neff = wg.neff(WL_UM, mode)
                ng = wg.group_index(WL_UM, mode)
            except ValueError:
                neff = ng = None
            out[f"neff_{mode.lower()}"].append(neff)
            out[f"ng_{mode.lower()}"].append(ng)
    return out


def find_boundary(mode: str) -> float | None:
    """Smallest width (5 nm grid) at which a second lateral mode is guided."""
    for w in np.arange(0.300, 1.2001, 0.005):
        if lateral_mode_count(float(w), WL_UM, mode) >= 2:
            return float(w)
    return None


def maybe_plot(widths: np.ndarray, data: dict[str, list[float | None]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    panels = [("n_eff", "neff"), ("n_g", "ng")]
    for ax, (label, key) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for i, mode in enumerate(("te", "tm")):
            y = np.array(
                [v if v is not None else np.nan for v in data[f"{key}_{mode}"]]
            )
            ax.plot(widths * 1e3, y, color=SERIES[i], lw=2)
            j = int(np.nanargmax(~np.isnan(y) * np.arange(len(y))))
            ax.annotate(
                mode.upper(), (widths[j] * 1e3, y[j]),
                xytext=(4, 0), textcoords="offset points",
                color=INK, fontsize=9, va="center",
            )
        ax.set_xlabel("width (nm)", color=INK_2)
        ax.set_ylabel(label, color=INK_2)
        ax.grid(color=GRID, lw=0.8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK_2)
    fig.suptitle(
        f"220 nm SOI strip at {WL_UM} um (effective index method)", color=INK
    )
    fig.tight_layout()
    path = out_dir / "01_waveguide_explorer.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] saved {path}")


def main() -> int:
    header("SiPhon 01 — SOI strip waveguide explorer")
    print(f"Geometry : Si strip, height {HEIGHT_UM * 1e3:.0f} nm, SiO2 clad/box")
    print(f"Wavelength: {WL_UM} um (C-band)")
    print(f"Materials : n_Si = {materials.n_si(WL_UM):.4f}, "
          f"n_SiO2 = {materials.n_sio2(WL_UM):.4f}")

    # --- width sweep -----------------------------------------------------
    header("Effective and group index vs width (fundamental modes)")
    widths = np.arange(0.35, 0.7001, 0.05)
    data = sweep(widths)

    def fmt(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "--"

    rows = []
    for i, w in enumerate(widths):
        rows.append([
            f"{w * 1e3:.0f}",
            fmt(data["neff_te"][i]), fmt(data["ng_te"][i]),
            fmt(data["neff_tm"][i]), fmt(data["ng_tm"][i]),
        ])
    table(
        ["width (nm)", "TE n_eff", "TE n_g", "TM n_eff", "TM n_g"],
        rows, [10, 9, 9, 9, 9],
    )
    print("\n'--' = mode not guided at this width (EIM finds no lateral mode).")
    print("Note the strong waveguide dispersion: n_g ~ 4.0 while n_eff is only")
    print("2.2-2.7 — the guided mode's confinement changes rapidly with lambda,")
    print("so d(n_eff)/d(lambda) is large and negative. n_g, not n_eff, is what")
    print("sets ring FSRs and true time delays.")

    # --- single-mode boundary ---------------------------------------------
    header("Single-mode boundary")
    for mode in ("TE", "TM"):
        b = find_boundary(mode)
        if b is None:
            print(f"{mode}: still single-mode up to 1.2 um (per EIM)")
        else:
            print(f"{mode}: second lateral mode appears at width ~ {b * 1e3:.0f} nm")
    print()
    print("Caveat: the EIM overestimates lateral confinement of higher-order")
    print("modes near cutoff, so these boundaries are conservative (early).")
    print("Full-vectorial solvers put the TE1 cutoff of a 220 nm strip near")
    print("450-500 nm width, which is why foundry PDKs standardize there: as")
    print("wide as possible for low sidewall-scattering loss, while the first")
    print("higher-order mode stays cut off or leaky. Treat the EIM numbers as")
    print("a design guide, not signoff.")

    # --- bend loss ---------------------------------------------------------
    header("Bend loss heuristic (220 nm strip, per Vlasov & McNab 2004 fit)")
    radii = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    rows = []
    for r in radii:
        per90 = bend_loss_db_per_90deg(r)
        rows.append([
            f"{r:.0f}",
            f"{per90:.4f}",
            f"{2 * per90:.4f}",
            f"{20 * per90:.3f}",
        ])
    table(
        ["R (um)", "dB / 90 deg", "dB / U-turn", "dB / 20 bends"],
        rows, [7, 12, 12, 14],
    )
    print("\nAbove R ~ 5 um the pure-bend loss is negligible against the")
    print("~1-3 dB/cm propagation loss of a typical strip; below R ~ 2 um it")
    print("dominates. This is why rings and dense routing sit at R = 3-10 um.")

    maybe_plot(widths, data)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
