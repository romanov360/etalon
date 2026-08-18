#!/usr/bin/env python3
"""Etalon example 11 — FFE rescues margin the unequalized ISI penalty eats.

Example 07 showed etalon.isi's ISI penalty growing fast once the symbol
rate becomes a sizable fraction of a demux passband's linewidth, and its
own docstring names the gap: that penalty assumes NO receiver
equalization, so it is only an upper bound on what an FFE-equipped
receiver actually sees. This script pushes the same 4-channel ring demux
to 106.25 GBd NRZ (a real 106G-class lane rate) where the unequalized eye
is open but paying a real ISI penalty, then asks etalon.equalize what a
zero-forcing FFE recovers — and books the honest cost: FFE cancels ISI
but always pays for it in noise enhancement. Both numbers, not one.

Scope reminder (see etalon.equalize): zero-forcing (not MMSE) FFE, no
DFE, no adaptive convergence — architecture-level budgeting, not signoff.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from etalon.waveguide import Waveguide

CENTER_UM = 1.55
N_CH = 4
SPACING_GHZ = 200.0
FSR_TARGET_UM = 0.0096
KAPPA_POWER = 0.10
LOSS_DB_PER_CM = 2.0
WIDTH_UM = 0.45
HEIGHT_UM = 0.22
ISI_CHANNEL = 1
BAUD_GBD = 106.25  # 106G-class NRZ lane rate; a real ISI penalty, eye still open
SPS = 16


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


def build_demux(RingAddDrop, Circuit, channels_um: list[float]):
    wg = Waveguide(width_um=WIDTH_UM, height_um=HEIGHT_UM)
    circuit = Circuit()
    designs = []
    for i, ch in enumerate(channels_um):
        neff0 = wg.neff(ch, "TE")
        ng = wg.group_index(ch, "TE")
        circ_fsr = ch**2 / (ng * FSR_TARGET_UM)
        m = max(1, round(neff0 * circ_fsr / ch))
        circ = m * ch / neff0
        circuit.add(
            f"ring{i}",
            RingAddDrop(
                circumference_um=circ, neff0=neff0, ng=ng,
                kappa1_power=KAPPA_POWER, kappa2_power=KAPPA_POWER,
                loss_db_per_cm=LOSS_DB_PER_CM, wl0_um=ch,
            ),
        )
        designs.append({"ch": ch, "m": m, "circ": circ, "ng": ng})
    circuit.expose("in", ("ring0", "in"))
    for i in range(N_CH - 1):
        circuit.connect((f"ring{i}", "through"), (f"ring{i + 1}", "in"))
    circuit.expose("thru", (f"ring{N_CH - 1}", "through"))
    for i in range(N_CH):
        circuit.expose(f"drop{i}", (f"ring{i}", "drop"))
        circuit.expose(f"add{i}", (f"ring{i}", "add"))
    return circuit, designs


def main() -> int:
    try:
        from etalon import equalize, isi, wdm
        from etalon.circuit import Circuit
        from etalon.components import RingAddDrop
        from etalon.link import LossElement, preset_cpo_optical_io
    except ImportError as exc:
        print(f"This example needs several etalon modules, not all built yet ({exc}).")
        return 1

    header("Etalon 11 — FFE vs. an unequalized ISI penalty (etalon.equalize)")

    plan = wdm.ChannelPlan.dwdm(CENTER_UM, SPACING_GHZ, N_CH)
    channels = list(plan.centers_um)
    circuit, designs = build_demux(RingAddDrop, Circuit, channels)
    ch_um = channels[ISI_CHANNEL]
    d = designs[ISI_CHANNEL]
    print(f"Same 4-channel DWDM ring demux as example 07: ring {ISI_CHANNEL} "
          f"L = {d['circ']:.2f} um, resonance at {ch_um * 1e3:.2f} nm.")

    span_um = 0.010
    wl = np.linspace(ch_um - span_um, ch_um + span_um, 200_001)
    t = circuit.transmission(wl, "in", f"drop{ISI_CHANNEL}")

    header(f"Unequalized eye at {BAUD_GBD:g} GBd NRZ (etalon.isi)")
    unequalized = isi.filter_isi_penalty_db(
        wl, t, center_wl_um=ch_um, rate_gbd=BAUD_GBD, levels=2, samples_per_symbol=SPS,
    )
    pen_str = "CLOSED (inf dB)" if math.isinf(unequalized.penalty_db) else f"{unequalized.penalty_db:.3f} dB"
    print(f"IL {unequalized.insertion_loss_db:.3f} dB, ISI penalty: {pen_str}")
    print("Real margin, eaten by a passband that's narrow relative to this baud —")
    print("etalon.isi's own docstring says this number is only an upper bound")
    print("'with an FFE-based receiver'; let's compute what that receiver actually")
    print("needs to recover it.")

    header("Closed-form zero-forcing FFE taps (etalon.equalize)")
    rows = []
    results = {}
    for n_post in (2, 4, 6, 8):
        r = equalize.solve_ffe_taps(
            wl, t, center_wl_um=ch_um, rate_gbd=BAUD_GBD, n_pre=1, n_post=n_post,
            samples_per_symbol=SPS,
        )
        results[n_post] = r
        resid = "-inf" if math.isinf(r.residual_isi_db) else f"{r.residual_isi_db:.2f}"
        rows.append([f"1 / {n_post}", resid, f"{r.noise_enhancement_db:+.3f}"])
    table(["taps (pre/post)", "residual ISI (dB)", "noise enhancement (dB)"],
          rows, [17, 18, 24])
    print("\nMore postcursor taps drive the residual ISI down (the tap span finally")
    print("reaching the ring's photon-lifetime tail), but the noise enhancement")
    print("keeps climbing — the FFE is not a free lunch, it trades ISI for noise")
    print("gain, and the two numbers must both be booked.")

    best = results[8]
    print(f"\nBest case here (1 pre / 8 post): {best.report()}")

    header("Booking BOTH numbers into the link budget")
    base = preset_cpo_optical_io()
    base = replace(base, signaling=replace(base.signaling, rate_gbd=BAUD_GBD))
    # Unequalized: IL as a path loss element, the full ISI penalty in
    # penalties_db (as example 07 does).
    unequalized_link = replace(
        base,
        path=base.path + [LossElement("rx demux drop port", unequalized.insertion_loss_db)],
        penalties_db={**(base.penalties_db or {}), "demux_isi": unequalized.penalty_db},
    )
    # Equalized: same IL, but the (now near-zero, by zero-forcing
    # construction) residual ISI in penalties_db, AND the noise
    # enhancement as an ADDITIONAL penalty -- it inflates the noise floor
    # the rin/shot penalties are already computed against, so it is
    # booked as its own line, not folded into the (unrelated) signal-side
    # ISI number.
    equalized = replace(
        base,
        path=base.path + [LossElement("rx demux drop port", unequalized.insertion_loss_db)],
        penalties_db={
            **(base.penalties_db or {}),
            "demux_isi_equalized": 0.0,  # zero-forcing: ~0 dB within the tap span
            "ffe_noise_enhancement": best.noise_enhancement_db,
        },
    )
    print(equalized.report())
    delta = equalized.margin_db - unequalized_link.margin_db
    print(f"\nUnequalized margin: {unequalized_link.margin_db:+.2f} dB "
          f"(full {unequalized.penalty_db:.3f} dB ISI penalty booked).")
    print(f"Equalized margin:   {equalized.margin_db:+.2f} dB "
          f"(~0 dB residual ISI, {best.noise_enhancement_db:+.3f} dB noise "
          f"enhancement instead) — net {delta:+.2f} dB from equalizing.")
    print("\nThis is the honest version of etalon.isi's 'FFE receiver: treat as an")
    print("upper bound' caveat — not just asserting the ISI penalty shrinks, but")
    print("pricing what shrinking it costs.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
