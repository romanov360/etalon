#!/usr/bin/env python3
"""SiPhon 10 — whole-bank WDM yield under correlated fab variation + thermal crosstalk.

Example 08 asked: for ONE as-fabricated ring bank, does the crosstalk-blind
assignment even lock? This script asks the manufacturing question: across a
population of dies with correlated (die-level + per-ring) fabrication
offset, what fraction of ring banks lock at all, and of those that lock,
what fraction land within a heater-power budget? siphon.montecarlo.run_bank
is the piece that makes this possible — wdm.optimize_ring_assignment and
thermal.solve_coupled_powers are both whole-bank joint solves, so a trial
can't be decomposed ring-by-ring the way siphon.montecarlo.run_module
decomposes lane failures.
"""

from __future__ import annotations

import numpy as np

from siphon import montecarlo as mc
from siphon import thermal, wdm

N_RINGS = 8
FSR_NM = 3.2
PITCH_UM = 30.0
DECAY_UM = 12.0          # tighter thermal isolation than example 08's 25 um
POWER_BUDGET_MW = 6.0    # per-ring heater power budget
N_TRIALS = 4_000


def header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def bank_margin_mw(offset_nm) -> np.ndarray:
    """One Monte Carlo trial: correlated offsets -> assignment -> coupled
    solve -> per-ring heater-power margin against POWER_BUDGET_MW.

    Raises ValueError (via solve_coupled_powers) when the crosstalk-coupled
    lock is unreachable at this pitch/decay — run_bank turns that into a
    whole-trial (all-ring) failure, exactly the bank-scrap event a real
    wafer lot would show.
    """
    offsets = np.asarray(offset_nm)
    layout = thermal.RingLayout.uniform(N_RINGS, PITCH_UM)
    assignment = wdm.optimize_ring_assignment(offsets, FSR_NM)
    target_nm = np.abs(np.array(assignment.per_ring_mw)) * wdm.TUNING_EFFICIENCY_NM_PER_MW
    result = thermal.solve_coupled_powers(target_nm, layout, DECAY_UM)
    return POWER_BUDGET_MW - np.array(result.heater_mw)


def main() -> int:
    header(f"SiPhon 10 — {N_RINGS}-ring WDM bank yield: fab variation x thermal crosstalk")
    print(f"FSR {FSR_NM:.1f} nm, {PITCH_UM:.0f} um pitch, {DECAY_UM:.0f} um healing length, "
          f"{POWER_BUDGET_MW:.1f} mW/ring budget, {N_TRIALS} trials")

    params = {
        # Die-level common offset (lithography/CD shifts the whole bank's
        # comb together) plus per-ring differential (local CD/thickness
        # noise) -- same common+differential split montecarlo.CommonDifferential
        # uses for lanes, here correlating WDM CHANNELS on one bus instead.
        "offset_nm": mc.BankParam(
            "offset_nm", mean=0.0, sigma_common=0.6, sigma_diff=0.4,
        ),
    }
    res = mc.run_bank(
        bank_margin_mw, params, n_rings=N_RINGS, n_trials=N_TRIALS, seed=2026,
        metric_name="heater margin (mW)",
    )
    print(res.report(threshold=0.0))

    header("Reading the result")
    lockable_frac = 1.0 - res.n_failed_trials / res.n_trials
    print(f"Trials where the crosstalk-coupled lock exists at all: "
          f"{100.0 * lockable_frac:.1f}% "
          f"({res.n_failed_trials} of {res.n_trials} scrapped outright).")
    print("This is thermal crosstalk overshooting some ring's target — a failure")
    print("mode with no per-lane analogue in montecarlo.run_module: it is a property")
    print("of the WHOLE bank's joint draw, not attributable to one ring.")
    print(f"\nOf banks that DO lock, fraction meeting the {POWER_BUDGET_MW:.1f} mW/ring "
          f"budget on every ring: "
          f"{100.0 * res.bank_yield_above(0.0) / max(lockable_frac, 1e-9):.1f}%")
    print(f"Overall bank yield (locks AND meets budget): "
          f"{100.0 * res.bank_yield_above(0.0):.1f}%")

    header("Sensitivity: differential (ring-to-ring) spread vs. thermal isolation")
    print(f"{'sigma_diff':>11}  {'decay_um':>9}  {'bank yield':>11}  {'lockable %':>11}")
    for sigma_diff, decay in ((0.4, 12.0), (0.2, 12.0), (0.4, 6.0), (0.2, 6.0)):
        def metric(offset_nm, _decay=decay):
            offsets = np.asarray(offset_nm)
            layout = thermal.RingLayout.uniform(N_RINGS, PITCH_UM)
            assignment = wdm.optimize_ring_assignment(offsets, FSR_NM)
            target_nm = (np.abs(np.array(assignment.per_ring_mw))
                         * wdm.TUNING_EFFICIENCY_NM_PER_MW)
            result = thermal.solve_coupled_powers(target_nm, layout, _decay)
            return POWER_BUDGET_MW - np.array(result.heater_mw)

        sweep_params = {
            "offset_nm": mc.BankParam("offset_nm", mean=0.0,
                                       sigma_common=0.6, sigma_diff=sigma_diff),
        }
        r = mc.run_bank(metric, sweep_params, n_rings=N_RINGS, n_trials=1_500, seed=2026)
        lockable = 100.0 * (1.0 - r.n_failed_trials / r.n_trials)
        print(f"{sigma_diff:>11.2f}  {decay:>9.0f}  "
              f"{100.0 * r.bank_yield_above(0.0):>10.1f}%  {lockable:>10.1f}%")

    print("\nsigma_common (not swept above) moves bank yield too, but roughly an order")
    print("of magnitude less than sigma_diff over a comparable range — because")
    print("optimize_ring_assignment's barrel shift absorbs MOST of the common-mode")
    print("offset (that is exactly its job), just not all of it, since the barrel")
    print("shift is one shared rotation, quantized to FSR/N steps. sigma_diff, the")
    print("ring-to-ring spread NO single rotation can absorb, and decay_um, thermal")
    print("isolation, dominate. A process team chasing yield by tightening litho")
    print("overlay (common-mode) alone will see real but disappointing returns; the")
    print("bigger payoff is reducing ring-to-ring (not die-to-die) variation and")
    print("improving thermal isolation — undercut trenches or wider ring pitch.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
