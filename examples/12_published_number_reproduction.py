#!/usr/bin/env python3
"""Etalon example 12 — reproducing a published industry claim, honestly.

docs/THESIS.md names this as the credibility move: "every published
reproduction (NVIDIA's 3.5x efficiency claim, Meta's laser-failure
statistics, OIF 3.2T budgets) is marketing an incumbent can't match without
open-sourcing its crown jewels." This script takes NVIDIA's specific,
sourced claim and checks it against etalon's own bottom-up energy model —
not by cherry-picking numbers to match, but by running the CPO-vs-pluggable
presets straight and reporting the ratio.

THE CLAIM (docs/RESEARCH.md, sourced to NVIDIA's public GTC/product
announcements): "NVIDIA claims 3.5x power efficiency [...] for
Quantum-X/Spectrum-X Photonics" CPO switches versus pluggable-optics
switches. See docs/RESEARCH.md line ~25 and ~80 for the sourced context.

THE HONEST RESULT: etalon's presets (examples/03_cpo_vs_pluggable.py's
architecture-level, illustrative-not-vendor-data link budgets) give roughly
2.2-2.3x, not 3.5x. This script does not paper over that gap — it explains
what's in NVIDIA's number that isn't in a pure optics-pJ/bit comparison,
and shows the sensitivity of the ratio to assumptions this toolkit makes
explicit (serdes/DSP allocation, thermal tuning power) that a marketing
slide does not have to.

Units: dB/dBm optical, pJ/bit electrical, W as labelled.
"""

from __future__ import annotations

from dataclasses import replace


# NVIDIA's claim, as sourced in docs/RESEARCH.md -----------------------------
CLAIMED_RATIO = 3.5
CLAIM_SOURCE = (
    "docs/RESEARCH.md (sourced to NVIDIA public GTC/Quantum-X/Spectrum-X "
    "Photonics announcements, 2026): 'claiming 3.5x power efficiency ... "
    "vs pluggables'"
)


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


def main() -> int:
    try:
        from etalon.link import energy_per_bit_pj, preset_cpo_optical_io, preset_pluggable_dr4
    except ImportError as exc:
        print(f"This example needs etalon.link ({exc}).")
        return 1

    header("Etalon 12 — reproducing NVIDIA's 3.5x CPO power-efficiency claim")
    print(f"Claim: {CLAIMED_RATIO:g}x")
    print(f"Source: {CLAIM_SOURCE}")

    plug = preset_pluggable_dr4()
    cpo = preset_cpo_optical_io()

    header("Etalon's own energy-per-bit model, run straight (no tuning to match)")
    epb_plug = energy_per_bit_pj(plug, n_lanes=8, serdes_pj_per_bit=4.0)
    epb_cpo = energy_per_bit_pj(
        cpo, n_lanes=8, thermal_tuning_mw_per_lane=2.0, serdes_pj_per_bit=1.0
    )
    rows = [
        [k, f"{epb_plug[k]:.3f}", f"{epb_cpo[k]:.3f}"]
        for k in ("laser", "modulator", "tia", "tuning", "serdes", "total")
    ]
    table(["contributor", "pluggable (pJ/bit)", "CPO (pJ/bit)"], rows, [12, 20, 14])

    ratio = epb_plug["total"] / epb_cpo["total"]
    print(f"\nEtalon's ratio: {epb_plug['total']:.2f} / {epb_cpo['total']:.2f} "
          f"= {ratio:.2f}x")
    print(f"NVIDIA's claim: {CLAIMED_RATIO:g}x")
    below_pct = 100.0 * (1.0 - ratio / CLAIMED_RATIO)
    above_pct = 100.0 * (CLAIMED_RATIO / ratio - 1.0)
    print(f"Gap: etalon's ratio is {below_pct:.0f}% below the claimed ratio "
          f"(equivalently, the claim is {above_pct:.0f}% higher than etalon's number).")

    header("Cross-check against independently published ABSOLUTE numbers")
    print("docs/RESEARCH.md (same line as the 3.5x claim, industry-sourced,")
    print("not NVIDIA-specific) also gives absolute pJ/bit ranges: pluggable")
    print("~15-20 pJ/bit, CPO ~5-8 pJ/bit (800G module-level).")
    print(f"Etalon's numbers here: pluggable {epb_plug['total']:.2f}, "
          f"CPO {epb_cpo['total']:.2f} pJ/bit — BOTH below the published ranges,")
    print("not just their ratio. That is the more important honest finding: this")
    print("energy_per_bit_pj model computes PER-LANE OPTICAL-ENGINE electrical")
    print("energy (laser/modulator/TIA/tuning/serdes-interface), not a full")
    print("MODULE-level number — packaging, power-supply conversion loss, host")
    print("ASIC-side gearbox/retimer chips beyond the XSR interface, and control")
    print("plane overhead are all outside this model's scope. The published")
    print("15-20 / 5-8 pJ/bit figures are module-level and naturally larger.")

    header("Why the gap is real, not a bug — four explicit reasons")
    print("1. LAYER MISMATCH (see above) — etalon models the optical engine's own")
    print("   electrical energy, not the full module; published absolute numbers")
    print("   are module-level and structurally larger on both sides.")
    print("2. SCOPE. NVIDIA's '3.5x' is a NETWORK claim (their own materials frame")
    print("   it as switch/fabric-level), which plausibly folds in copper-trace")
    print("   SerDes elimination on the switch ASIC side, not just optics pJ/bit —")
    print("   this script only compares optical-I/O energy, the piece etalon")
    print("   actually models.")
    print("3. SERDES ALLOCATION. The two presets' 'serdes' contributor (host-side")
    print("   SerDes/DSP energy NOT already inside the modulator/TIA numbers) is a")
    print("   caller-supplied knob, not something etalon derives — this script uses")
    print("   4.0 pJ/bit (pluggable, full retimer DSP) vs 1.0 pJ/bit (CPO, XSR-class")
    print("   die-to-optics), the same assumption examples/03 uses. NVIDIA's own")
    print("   number for their gearbox-free architecture could differ.")
    print("4. THERMAL TUNING. The CPO preset carries an explicit 2.0 mW/lane ring")
    print("   heater tax (etalon.thermal's whole reason for existing) that a")
    print("   marketing pJ/bit number may or may not include explicitly.")

    header("Sensitivity: how close can defensible assumptions get to 3.5x?")
    print(f"{'serdes CPO (pJ/bit)':>22}  {'tuning (mW/lane)':>18}  {'ratio':>7}")
    for serdes_cpo, tuning_mw in ((1.0, 2.0), (0.5, 2.0), (1.0, 0.5), (0.5, 0.2)):
        epb = energy_per_bit_pj(
            cpo, n_lanes=8, thermal_tuning_mw_per_lane=tuning_mw,
            serdes_pj_per_bit=serdes_cpo,
        )
        r = epb_plug["total"] / epb["total"]
        print(f"{serdes_cpo:>22.2f}  {tuning_mw:>18.2f}  {r:>6.2f}x")
    print("\nEven the most favorable defensible corner in this sweep (minimal XSR")
    print("serdes energy, well-isolated/undercut ring thermal tuning) does not")
    print("reach 3.5x on optics-only pJ/bit — supporting reason #2: NVIDIA's number")
    print("is very likely counting network-level savings (copper SerDes removal on")
    print("the switch side) beyond what a pure optical-I/O energy model captures.")

    header("What this reproduction is worth")
    print("Not 'NVIDIA is wrong' — their number is plausibly correct at the scope")
    print("they're measuring. It IS worth something that etalon's independent,")
    print("open, bottom-up physics lands in the same regime (a real few-x")
    print("improvement, not 10x and not 1.1x) using nothing but public component")
    print("assumptions — and that the gap has a specific, statable, falsifiable")
    print("explanation instead of a shrug. That is the credibility artifact")
    print("docs/THESIS.md is describing: reproducible, sourced, and honest about")
    print("where it does and doesn't match the marketing.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
