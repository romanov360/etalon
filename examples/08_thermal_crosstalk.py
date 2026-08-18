"""Etalon 08 — ring-to-ring thermal crosstalk on a DWDM ring bank.

etalon.wdm.optimize_ring_assignment picks the channel assignment that
minimizes total heater power, treating every ring as thermally isolated.
This script asks the next question: once that assignment is heating
several rings on the same bus, how much does each ring's heater detune
its NEIGHBORS, and what does the honest (coupled) power budget look like
once the bank has to fight its own crosstalk to stay locked?
"""

from __future__ import annotations

import numpy as np

from etalon import thermal, wdm

N_RINGS = 8
FSR_NM = 3.2  # a dense DWDM-scale FSR (contrast with example 02's CWDM case)
PITCH_UM = 30.0  # ring-to-ring spacing on the bus
DECAY_UM = 25.0  # thermal healing length: bulk (non-undercut) SOI, mid measured range


def header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    header(f"Etalon 08 — thermal crosstalk on an {N_RINGS}-ring DWDM bank")

    rng = np.random.default_rng(2026)
    offsets_nm = rng.uniform(-FSR_NM / 2, FSR_NM / 2, N_RINGS)
    print(f"As-fabricated resonance offsets (nm, uniform over the {FSR_NM:.1f} nm FSR):")
    print("  " + ", ".join(f"{o:+.3f}" for o in offsets_nm))

    assignment = wdm.optimize_ring_assignment(offsets_nm, FSR_NM)
    print(f"\noptimize_ring_assignment: rotation={assignment.rotation}, "
          f"naive total {assignment.naive_total_mw:.3f} mW -> "
          f"optimized total {assignment.total_mw:.3f} mW "
          f"({100.0 * (1 - assignment.total_mw / assignment.naive_total_mw):.0f}% saved) — "
          "all THERMALLY ISOLATED so far.")

    header(f"Adding crosstalk: {N_RINGS} rings at {PITCH_UM:.0f} um pitch, "
           f"{DECAY_UM:.0f} um healing length")
    layout = thermal.RingLayout.uniform(N_RINGS, PITCH_UM)
    k = thermal.coupling_matrix(layout, DECAY_UM)
    print(f"Nearest-neighbor coupling fraction: {k[0, 1]:.4f} "
          f"(a neighbor's heater raises this ring's temperature by "
          f"{100 * k[0, 1]:.1f}% of what the same power would do to itself)")

    target_nm = np.abs(np.array(assignment.per_ring_mw)) * wdm.TUNING_EFFICIENCY_NM_PER_MW
    screen = thermal.worst_case_neighbor_shift_nm(assignment.per_ring_mw, layout, DECAY_UM)
    print("\nScreening bound (etalon.wdm's assignment used AS-IS, crosstalk-blind):")
    print(f"  {'ring':>4}  {'own target (nm)':>16}  {'neighbor shift (nm)':>20}  {'% of target':>11}")
    for i, (t, s) in enumerate(zip(target_nm, screen)):
        pct = f"{100 * s / t:6.1f}%" if t > 1e-9 else "    n/a"
        print(f"  {i:>4}  {t:>16.4f}  {s:>+20.4f}  {pct:>11}")
    worst_pct = 100 * np.max(np.abs(screen) / np.maximum(target_nm, 1e-9))
    print(f"\nWorst-case unwanted detune from neighbor heat: {worst_pct:.0f}% of a ring's "
          "own target shift —")
    print("large enough that a crosstalk-blind lock will not actually land on-channel.")

    header("Self-consistent coupled solve (etalon.thermal.solve_coupled_powers)")
    try:
        coupled = thermal.solve_coupled_powers(target_nm, layout, DECAY_UM)
        print(coupled.report())
        print(f"\nCrosstalk tax vs. the thermally-isolated optimum: "
              f"{100.0 * (coupled.total_mw / assignment.total_mw - 1.0):+.1f}% total power.")
    except ValueError as exc:
        print(f"solve_coupled_powers: {exc}")
        print("\nThis is the headline result, not a demo bug: rings 2 and 4 sit next to")
        print("heavily-heated neighbors and need a near-zero target shift of their own.")
        print("At this pitch/healing-length combination, neighbor heat alone overshoots")
        print("their target, and a resistive heater cannot pull heat back out — the")
        print("crosstalk-blind assignment from optimize_ring_assignment is physically")
        print("UNLOCKABLE as chosen, not just power-inefficient.")

    header("Sweep: healing length vs. whether the bank can lock at all")
    print(f"{'decay_um':>10}  {'coupled mW':>12}  {'vs isolated':>14}")
    for decay in (2.0, 5.0, 8.0, 12.0, 18.0, 25.0):
        try:
            r = thermal.solve_coupled_powers(
                target_nm, thermal.RingLayout.uniform(N_RINGS, PITCH_UM), decay
            )
            delta = 100.0 * (r.total_mw / assignment.total_mw - 1.0)
            print(f"{decay:>10.0f}  {r.total_mw:>12.3f}  {delta:>+13.1f}%")
        except ValueError:
            print(f"{decay:>10.0f}  {'--':>12}  {'unlockable':>14}")
    print("\nShorter healing length (better thermal isolation, e.g. undercut trenches,")
    print("or simply more ring pitch on the layout) both lowers the power tax AND")
    print("determines whether the crosstalk-blind assignment can lock at all — below")
    print("some pitch/decay ratio it categorically cannot, no matter how much power")
    print("is spent. optimize_ring_assignment cannot see this failure mode; it")
    print("assumes thermal isolation. The two functions are complementary: assign")
    print("first to minimize the isolated bound, then run the coupled solve to find")
    print("out whether — and at what cost — that assignment survives contact with")
    print("its own neighbors' heat.")
    print()


if __name__ == "__main__":
    main()
