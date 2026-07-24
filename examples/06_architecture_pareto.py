#!/usr/bin/env python3
"""SiPhon example 06 — architecture Pareto sweep: link margin vs pJ/bit.

Sweeps a CPO-style link (microring, remote laser — the preset_cpo_optical_io
component stack) over modulation order, symbol rate, lane count, and laser
sharing, then ranks every configuration on the two axes an architect actually
trades: unallocated link margin (dB, incl. shot + RIN penalties) and
electrical energy per transported bit (pJ/bit). Prints an aligned table and
the Pareto-efficient subset (no other config has both more margin and lower
pJ/bit), and saves a scatter to examples/out/. Illustrative numbers, not any
vendor's data; architecture-level, not signoff.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

LEVELS = [2, 4]
BAUDS_GBD = [26.5625, 53.125, 106.25]
N_LANES = [4, 8, 16]
LASER_SHARED_BY = [1, 4]
LASER_OVERHEAD_MW = 20.0      # per-device control/TEC, amortized by sharing
TUNING_MW_PER_LANE = 2.0      # ring heater lock
SERDES_PJ = 1.0               # XSR-class die-to-optics interface

# Chart palette (same validated categorical slots as example 03)
SERIES = ["#2a78d6", "#eb6834"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"


def pareto(points: list[tuple]) -> list[tuple]:
    """Subset not dominated on (max margin, min pJ/bit)."""
    return [
        p for p in points
        if not any(
            q[1] >= p[1] and q[2] <= p[2] and (q[1] > p[1] or q[2] < p[2])
            for q in points
        )
    ]


def maybe_plot(points: list[tuple], front: list[tuple]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, lv in enumerate(LEVELS):
        pts = [p for p in points if p[3] == lv]
        ax.scatter([p[2] for p in pts], [p[1] for p in pts], s=22,
                   color=SERIES[i], label=f"PAM{lv}" if lv > 2 else "NRZ", alpha=0.8)
    fr = sorted(front, key=lambda p: p[2])
    ax.plot([p[2] for p in fr], [p[1] for p in fr], color=INK, lw=1,
            ls="--", marker="o", ms=5, mfc="none", label="Pareto front")
    ax.axhline(0.0, color=GRID, lw=0.8)
    ax.set_xlabel("energy per bit (pJ/bit)", color=INK_2)
    ax.set_ylabel("link margin (dB)", color=INK_2)
    ax.set_title("CPO-style architecture sweep: margin vs energy", color=INK)
    leg = ax.legend(frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_2)
    fig.tight_layout()
    path = out_dir / "06_architecture_pareto.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] saved {path}")


def main() -> int:
    try:
        from siphon.link import Signaling, energy_per_bit_pj, preset_cpo_optical_io
    except ImportError as exc:
        print(f"This example needs siphon.link ({exc}); re-run after integration.")
        return 1

    base = preset_cpo_optical_io()
    points = []  # (label, margin_db, pj_per_bit, levels)
    for lv in LEVELS:
        for baud in BAUDS_GBD:
            link = replace(base, signaling=Signaling(rate_gbd=baud, levels=lv))
            for lanes in N_LANES:
                for shared in LASER_SHARED_BY:
                    epb = energy_per_bit_pj(
                        link, n_lanes=lanes, laser_shared_by=shared,
                        laser_overhead_mw_per_device=LASER_OVERHEAD_MW,
                        thermal_tuning_mw_per_lane=TUNING_MW_PER_LANE,
                        serdes_pj_per_bit=SERDES_PJ,
                    )
                    mod = f"PAM{lv}" if lv > 2 else "NRZ "
                    label = (f"{mod} {baud:8.4f} GBd x{lanes:2d} lanes, "
                             f"laser/{shared}")
                    points.append((label, link.margin_db, epb["total"], lv))

    front = set(id(p) for p in pareto(points))
    print("SiPhon 06 — architecture Pareto sweep (CPO-style component stack)")
    print(f"{len(points)} configs; margin includes shot-noise + RIN penalties.\n")
    hdr = f"{'config':<36} {'margin (dB)':>12} {'pJ/bit':>8}  Pareto"
    print(hdr)
    print("-" * len(hdr))
    for p in sorted(points, key=lambda p: p[2]):
        mark = "  *" if id(p) in front else ""
        print(f"{p[0]:<36} {p[1]:>+12.2f} {p[2]:>8.2f}{mark}")
    print("\nPareto-efficient subset (max margin, min pJ/bit):")
    for p in sorted(pareto(points), key=lambda p: p[2]):
        print(f"  {p[0]:<36} {p[1]:>+8.2f} dB  {p[2]:6.2f} pJ/bit")
    print("\nReading: sharing one laser across 4 lanes amortizes only the")
    print("per-device overhead; NRZ banks margin that PAM4 trades away for")
    print("lane speed. Lane count is degenerate here — per-line laser power")
    print("and bandwidth scale together — so it separates only when overhead")
    print("or shared infrastructure breaks the proportionality. Any")
    print("negative-margin row is infeasible as drawn (needs more laser")
    print("power or lower loss, which moves it right on the chart).")

    maybe_plot(points, pareto(points))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
