#!/usr/bin/env python3
"""SiPhon example 07 — from demux passband to a COMPUTED link-budget penalty.

The E-O-E bridge: a 4-channel DWDM ring demux is composed with the circuit
solver, the drop-port field response S21(lambda) of one channel is fed to
siphon.isi.filter_isi_penalty_db, and the resulting eye-closure (ISI)
penalty enters the link budget as a COMPUTED penalties_db entry — the kind
of allocation that is otherwise hand-entered (cf. the DR4 preset's
'isi_equalization': 1.0 guess). The script sweeps NRZ baud through the drop
port (table of IL and ISI penalty), then rebuilds preset_cpo_optical_io()'s
waterfall with BOTH numbers booked once each — the drop-port IL as a path
LossElement, the ISI penalty in penalties_db — and prints the margin delta.

Scope reminder (see siphon.isi): passive linear filtering downstream of the
modulator, chirp-free ideal PAM — budgeting-grade eye closure, not TDECQ.
Units: um for wavelength, GBd for symbol rate, dB for loss/penalty.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from siphon.waveguide import Waveguide

# Design targets ---------------------------------------------------------
CENTER_UM = 1.55          # C-band DWDM demux for a CPO-style link
N_CH = 4
SPACING_GHZ = 200.0       # dense grid: where ring demuxes earn their keep
FSR_TARGET_UM = 0.0096    # ~9.6 nm FSR = 2x the 4-channel span (guard band)
KAPPA_POWER = 0.10        # power cross-coupling, both couplers
LOSS_DB_PER_CM = 2.0
WIDTH_UM = 0.45           # C-band single-mode strip
HEIGHT_UM = 0.22
BAUDS_GBD = [8.0, 16.0, 32.0, 53.125]
# samples/symbol per baud: keep the simulated analog band >= ~0.5 THz so the
# slow bauds are not dominated by band-edge truncation of the passband tails.
SPS_BY_BAUD = {8.0: 64, 16.0: 32, 32.0: 16, 53.125: 16}
ISI_CHANNEL = 1           # compute the penalty for this drop port

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
    print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))


def build_demux(RingAddDrop, Circuit, channels_um: list[float]):
    """4 add-drop rings cascaded on one bus, one drop port per channel."""
    wg = Waveguide(width_um=WIDTH_UM, height_um=HEIGHT_UM)
    circuit = Circuit()
    designs = []
    for i, ch in enumerate(channels_um):
        neff0 = wg.neff(ch, "TE")
        ng = wg.group_index(ch, "TE")
        circ_fsr = ch**2 / (ng * FSR_TARGET_UM)   # FSR = wl^2 / (ng * L)
        m = max(1, round(neff0 * circ_fsr / ch))  # resonance: neff * L = m * wl
        circ = m * ch / neff0                     # snap ring onto its channel
        circuit.add(
            f"ring{i}",
            RingAddDrop(
                circumference_um=circ,
                neff0=neff0,
                ng=ng,
                kappa1_power=KAPPA_POWER,
                kappa2_power=KAPPA_POWER,
                loss_db_per_cm=LOSS_DB_PER_CM,
                wl0_um=ch,
            ),
        )
        designs.append({"ch": ch, "m": m, "circ": circ, "ng": ng})
    circuit.expose("in", ("ring0", "in"))
    for i in range(N_CH - 1):
        circuit.connect((f"ring{i}", "through"), (f"ring{i + 1}", "in"))
    circuit.expose("thru", (f"ring{N_CH - 1}", "through"))
    for i in range(N_CH):
        circuit.expose(f"drop{i}", (f"ring{i}", "drop"))
        circuit.expose(f"add{i}", (f"ring{i}", "add"))  # unused, terminated
    return circuit, designs


def maybe_plot(wl, drop_db, ch_um, bauds, penalties) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(10, 3.8), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )
    ax0.set_facecolor(SURFACE)
    ax0.plot((wl - ch_um) * 1e3, drop_db, color=SERIES[0], lw=1.4)
    ax0.set_xlim(-2.0, 2.0)
    ax0.set_ylim(-45, 3)
    ax0.set_xlabel(f"wavelength - {ch_um * 1e3:.2f} nm (nm)", color=INK_2)
    ax0.set_ylabel("drop transmission (dB)", color=INK_2)
    ax0.set_title(f"demux drop{ISI_CHANNEL} passband", color=INK)

    ax1.set_facecolor(SURFACE)
    finite = [(b, p) for b, p in zip(bauds, penalties) if math.isfinite(p)]
    ax1.plot(
        [b for b, _ in finite], [p for _, p in finite],
        color=SERIES[1], lw=1.6, marker="o", ms=4,
    )
    for b, p in zip(bauds, penalties):
        if not math.isfinite(p):
            ax1.annotate(
                "eye closed", (b, ax1.get_ylim()[1]), ha="right",
                color=SERIES[1], fontsize=8.5,
            )
    ax1.set_xlabel("symbol rate (GBd)", color=INK_2)
    ax1.set_ylabel("ISI penalty (dB)", color=INK_2)
    ax1.set_title("computed eye closure, NRZ", color=INK)

    for ax in (ax0, ax1):
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK_2)
    fig.tight_layout()
    path = out_dir / "07_filter_isi.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] saved {path}")


def main() -> int:
    try:
        from siphon import isi, wdm
        from siphon.circuit import Circuit
        from siphon.components import RingAddDrop
        from siphon.link import LossElement, preset_cpo_optical_io
    except ImportError as exc:
        print("This example needs siphon.isi, siphon.circuit, siphon.components,")
        print(f"siphon.wdm and siphon.link, which are not all built yet ({exc}).")
        print("Re-run after integration; example 01 runs on the core modules alone.")
        return 1

    header("SiPhon 07 — demux passband -> computed ISI penalty -> link margin")

    plan = wdm.ChannelPlan.dwdm(CENTER_UM, SPACING_GHZ, N_CH)
    channels = list(plan.centers_um)
    print("Channel plan: 4-ch DWDM, "
          + ", ".join(f"{c * 1e3:.2f}" for c in channels)
          + f" nm ({SPACING_GHZ:.0f} GHz grid)")
    print(f"Demux: add-drop rings on one bus, kappa^2 = {KAPPA_POWER} per "
          f"coupler, {LOSS_DB_PER_CM} dB/cm,")
    print(f"target FSR {FSR_TARGET_UM * 1e3:.1f} nm "
          f"(2x the {plan.span_nm():.1f} nm channel span).")

    circuit, designs = build_demux(RingAddDrop, Circuit, channels)
    ch_um = channels[ISI_CHANNEL]
    d = designs[ISI_CHANNEL]
    print(f"Ring {ISI_CHANNEL}: L = {d['circ']:.2f} um (mode m = {d['m']}), "
          f"resonance snapped to {ch_um * 1e3:.2f} nm.")

    # Composed drop response of the target channel: the signal passes the
    # upstream rings' through ports, then drops — circuit.transmission IS
    # the S21(lambda) the ISI engine consumes.
    span_um = 0.0045  # covers +-3.6 nm needed at 53.125 GBd, sps = 16
    wl = np.linspace(ch_um - span_um, ch_um + span_um, 120_001)
    t = circuit.transmission(wl, "in", f"drop{ISI_CHANNEL}")
    drop_db = 10.0 * np.log10(np.maximum(np.abs(t) ** 2, 1e-15))
    above = np.flatnonzero(drop_db >= drop_db.max() - 3.0103)
    # delta_f[GHz] = c * delta_wl / wl^2, c = 2.99792458e5 um*GHz
    fwhm_ghz = (wl[above[-1]] - wl[above[0]]) / ch_um**2 * 2.99792458e5
    print(f"Composed drop{ISI_CHANNEL} passband: IL "
          f"{-drop_db.max():.2f} dB at peak, FWHM ~ {fwhm_ghz:.0f} GHz.")

    header(f"NRZ baud sweep through drop{ISI_CHANNEL} "
           "(IL and ISI penalty are SEPARATE)")
    rows, penalties = [], []
    result_32 = None
    for baud in BAUDS_GBD:
        r = isi.filter_isi_penalty_db(
            wl, t, center_wl_um=ch_um, rate_gbd=baud, levels=2,
            samples_per_symbol=SPS_BY_BAUD.get(baud, 16),
        )
        penalties.append(r.penalty_db)
        if baud == 32.0:
            result_32 = r
        pen = "eye closed" if math.isinf(r.penalty_db) else f"{r.penalty_db:.3f}"
        rows.append([
            f"{baud:g}", f"{r.insertion_loss_db:.3f}", pen,
            f"{r.sampling_phase_ui:.3f}",
        ])
    table(
        ["baud (GBd)", "IL (dB)", "ISI penalty (dB)", "phase (UI)"],
        rows, [10, 8, 16, 10],
    )
    print(f"\nThe ~{fwhm_ghz:.0f} GHz drop passband is essentially free at "
          "8-16 GBd, but the penalty")
    print("grows fast once the symbol rate becomes a sizable fraction of the")
    print("linewidth — same composed filter, one computed table, no")
    print("hand-entered allocation. (Values within a few thousandths of 0 dB")
    print("mean 'negligible'.)")

    header("Rebuilding preset_cpo_optical_io() with the computed penalty")
    base = preset_cpo_optical_io()
    print(result_32.report())
    print()
    # Book each number exactly once: the preset's path has no demux loss
    # element, so the drop-port IL goes into the PATH; the eye closure
    # beyond that flat loss goes into penalties_db.
    upgraded = replace(
        base,
        path=base.path + [LossElement("rx demux drop port",
                                      result_32.insertion_loss_db)],
        penalties_db={**(base.penalties_db or {}),
                      "demux_isi": result_32.penalty_db},
    )
    print(upgraded.report())
    delta = upgraded.margin_db - base.margin_db
    print(f"\nMargin delta from the computed demux (IL + ISI): {delta:+.3f} dB "
          f"({base.margin_db:+.2f} -> {upgraded.margin_db:+.2f} dB) —")
    print(f"{result_32.insertion_loss_db:.3f} dB flat loss in the path plus "
          f"{result_32.penalty_db:.3f} dB eye closure in penalties_db.")
    print("siphon.isi reports the two separately precisely so each is booked")
    print("exactly once (a preset path may or may not already carry the IL).")

    maybe_plot(wl, drop_db, ch_um, BAUDS_GBD, penalties)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
