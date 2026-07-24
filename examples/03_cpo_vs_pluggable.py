#!/usr/bin/env python3
"""SiPhon example 03 — CPO vs pluggable optics for a 51.2 Tb/s switch.

The flagship comparison: one lane of a 400G-DR4-class pluggable module
(106.25 Gb/s PAM4, MZM, full DSP) against one lane of a co-packaged /
optical-I/O style link (32 Gb/s NRZ, microring, remote laser, XSR serdes).
Prints both link-budget waterfalls, compares margin, breaks down electrical
energy per bit by contributor, and converts the pJ/bit difference into wall
power and electricity cost for a 51.2 Tb/s switch carrying 64 x 800G of
optical engines.

Requires siphon.link. All component numbers are the presets' illustrative
2026-plausible values, not any vendor's data. Units: dB/dBm optical, pJ/bit
electrical, W and $ as labelled.
"""

from __future__ import annotations

from pathlib import Path

# Switch-level framing --------------------------------------------------------
SWITCH_TBPS = 51.2                # front-panel bandwidth per switch ASIC
N_ENGINES = 64                    # 64 x 800G optical engines
ENGINE_GBPS = 800.0
ELECTRICITY_USD_PER_KWH = 0.10    # US industrial-ish datacenter rate
PUE = 1.3                         # facility overhead multiplier
LIFETIME_YEARS = 5.0
USD_PER_W_YEAR = ELECTRICITY_USD_PER_KWH / 1e3 * 8760.0 * PUE  # ~$1.14/W/yr

# Energy-model knobs (see link.energy_per_bit_pj docstring) --------------------
N_LANES = 8                       # lanes per engine model in both cases
PLUG_SERDES_PJ = 4.0              # host serdes + module DSP allocation
CPO_SERDES_PJ = 1.0               # XSR-class die-to-optics interface
CPO_TUNING_MW_PER_LANE = 2.0      # ring heater lock, dense grid (cf. example 02)
CPO_LASER_SHARED_BY = 1           # one ELS line per lane in this preset

# Chart palette (validated categorical slots, light mode)
SERIES = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"

EPB_KEYS = ["laser", "modulator", "tia", "tuning", "serdes", "total"]


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


def epb_total(epb: dict[str, float]) -> float:
    for k, v in epb.items():
        if "total" in k:
            return float(v)
    return float(sum(v for v in epb.values() if isinstance(v, (int, float))))


def optics_watts(pj_per_bit: float, tbps: float = SWITCH_TBPS) -> float:
    """Wall power of the optical I/O: pJ/bit x Tb/s -> W (1 pJ/bit @ 1 Tb/s = 1 W)."""
    return pj_per_bit * tbps


def maybe_plot(epb_plug: dict[str, float], epb_cpo: dict[str, float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)

    keys = [k for k in EPB_KEYS if k != "total" and (k in epb_plug or k in epb_cpo)]
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (label, epb) in enumerate(
        [("pluggable DR4", epb_plug), ("CPO optical I/O", epb_cpo)]
    ):
        vals = [float(epb.get(k, 0.0)) for k in keys]
        bars = ax.bar(
            x + (i - 0.5) * (w + 0.02), vals, width=w,
            color=SERIES[i], label=label, edgecolor=SURFACE, linewidth=1,
        )
        for b, v in zip(bars, vals):
            ax.annotate(
                f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                xytext=(0, 2), textcoords="offset points",
                ha="center", color=INK_2, fontsize=8,
            )
    ax.set_xticks(x, keys)
    ax.set_ylabel("pJ/bit", color=INK_2)
    ax.set_title("Electrical energy per bit by contributor", color=INK)
    leg = ax.legend(frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_2)
    fig.tight_layout()
    path = out_dir / "03_energy_per_bit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] saved {path}")


def main() -> int:
    try:
        from siphon.link import (
            energy_per_bit_pj,
            preset_cpo_optical_io,
            preset_pluggable_dr4,
        )
    except ImportError as exc:
        print("This example needs siphon.link, which is not built in this")
        print(f"checkout yet ({exc}).")
        print("Re-run after integration; example 01 runs on the core modules alone.")
        return 1

    plug = preset_pluggable_dr4()
    cpo = preset_cpo_optical_io()

    header("SiPhon 03 — pluggable DR4 vs co-packaged optical I/O")
    print("Two ways to get bits off a 51.2 Tb/s switch ASIC:")
    print("  A. pluggable DR4-class module — 106.25 Gb/s PAM4, Si MZM, full DSP")
    print("  B. CPO / optical I/O          — 32 Gb/s NRZ, microring, remote laser")
    print("All numbers are the presets' illustrative values (see siphon.link).")

    header("A. Pluggable DR4 — link-budget waterfall (per lane)")
    print(plug.report())

    header("B. CPO optical I/O — link-budget waterfall (per lane)")
    print(cpo.report())

    # --- margin comparison ---------------------------------------------------
    header("Margin comparison")
    rows = [
        ["launched OMA (dBm)", f"{plug.launched_oma_dbm:+.2f}", f"{cpo.launched_oma_dbm:+.2f}"],
        ["path loss (dB)", f"{plug.path_loss_db:.2f}", f"{cpo.path_loss_db:.2f}"],
        ["received OMA (dBm)", f"{plug.received_oma_dbm:+.2f}", f"{cpo.received_oma_dbm:+.2f}"],
        ["required OMA (dBm)", f"{plug.sensitivity_oma_dbm:+.2f}", f"{cpo.sensitivity_oma_dbm:+.2f}"],
        ["MARGIN (dB)", f"{plug.margin_db:+.2f}", f"{cpo.margin_db:+.2f}"],
    ]
    table(["", "pluggable", "CPO"], rows, [22, 10, 10])
    print("\nThe pluggable burns a 12 dBm laser to survive PAM4's SNR appetite")
    print("plus two lossy grating couplers; the CPO lane runs NRZ with a low-RIN")
    print("6.5 dBm remote line and banks several dB more margin — headroom that")
    print("real designs trade back for cheaper couplers or lower laser power.")

    # --- energy per bit -------------------------------------------------------
    header(f"Energy per bit (electrical), {N_LANES} lanes per engine")
    epb_plug = energy_per_bit_pj(plug, n_lanes=N_LANES, serdes_pj_per_bit=PLUG_SERDES_PJ)
    epb_cpo = energy_per_bit_pj(
        cpo,
        n_lanes=N_LANES,
        laser_shared_by=CPO_LASER_SHARED_BY,
        thermal_tuning_mw_per_lane=CPO_TUNING_MW_PER_LANE,
        serdes_pj_per_bit=CPO_SERDES_PJ,
    )
    rows = []
    for k in EPB_KEYS:
        if k not in epb_plug and k not in epb_cpo:
            continue
        a = float(epb_plug.get(k, 0.0))
        b = float(epb_cpo.get(k, 0.0))
        rows.append([k, f"{a:.2f}", f"{b:.2f}", f"{b - a:+.2f}"])
    table(
        ["contributor", "pluggable", "CPO", "delta"],
        rows, [12, 10, 8, 8],
    )
    print("                       (pJ/bit; 1 mW per Gb/s = 1 pJ/bit)")
    t_plug, t_cpo = epb_total(epb_plug), epb_total(epb_cpo)
    print("\nSerdes/DSP dominates the pluggable and is where CPO wins: the XSR")
    print(f"reach cuts it {PLUG_SERDES_PJ:.0f} -> {CPO_SERDES_PJ:.0f} pJ/bit, and the "
          "ring modulator + NRZ TIA shave the")
    print("rest. The laser term barely moves — the CPO line is 5.5 dB weaker but")
    print("each lane carries 3.3x fewer bits — and rings add a small tuning tax")
    print(f"({CPO_TUNING_MW_PER_LANE:.0f} mW/lane of heaters).")

    # --- switch-level framing ---------------------------------------------------
    header(f"Per-switch framing: {SWITCH_TBPS:g} Tb/s "
           f"({N_ENGINES} x {ENGINE_GBPS:g}G optical engines)")
    w_plug = optics_watts(t_plug)
    w_cpo = optics_watts(t_cpo)
    dw = w_plug - w_cpo
    rows = [
        ["energy per bit (pJ/bit)", f"{t_plug:.2f}", f"{t_cpo:.2f}", f"{t_cpo - t_plug:+.2f}"],
        ["optics power per switch (W)", f"{w_plug:.0f}", f"{w_cpo:.0f}", f"{w_cpo - w_plug:+.0f}"],
        [f"per {ENGINE_GBPS:g}G engine (W)",
         f"{w_plug / N_ENGINES:.1f}", f"{w_cpo / N_ENGINES:.1f}",
         f"{(w_cpo - w_plug) / N_ENGINES:+.1f}"],
        ["per 1.6T of front panel (W)",
         f"{t_plug * 1.6:.1f}", f"{t_cpo * 1.6:.1f}", f"{(t_cpo - t_plug) * 1.6:+.1f}"],
        ["electricity $/switch/year",
         f"{w_plug * USD_PER_W_YEAR:,.0f}", f"{w_cpo * USD_PER_W_YEAR:,.0f}",
         f"{-dw * USD_PER_W_YEAR:+,.0f}"],
        [f"electricity $/switch/{LIFETIME_YEARS:.0f}yr",
         f"{w_plug * USD_PER_W_YEAR * LIFETIME_YEARS:,.0f}",
         f"{w_cpo * USD_PER_W_YEAR * LIFETIME_YEARS:,.0f}",
         f"{-dw * USD_PER_W_YEAR * LIFETIME_YEARS:+,.0f}"],
    ]
    table(["", "pluggable", "CPO", "delta"], rows, [28, 10, 9, 9])
    print(f"\nAssumptions: ${ELECTRICITY_USD_PER_KWH}/kWh, PUE {PUE} "
          f"=> ${USD_PER_W_YEAR:.2f} per W-year at the wall.")
    print(f"CPO saves ~{dw:.0f} W of optics per switch — "
          f"${dw * USD_PER_W_YEAR * LIFETIME_YEARS:,.0f} of electricity over "
          f"{LIFETIME_YEARS:.0f} years,")
    print(f"or ${dw * USD_PER_W_YEAR * LIFETIME_YEARS * 1000:,.0f} per 1000 "
          "switches — before counting the smaller heat load, the")
    print("reclaimed faceplate, and the shorter electrical channels that made")
    print("the XSR serdes possible in the first place. The trade: serviceability")
    print("(a failed engine is no longer a hot-swap part) and laser redundancy")
    print("engineering on the external light source.")

    maybe_plot(epb_plug, epb_cpo)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
