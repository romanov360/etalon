#!/usr/bin/env python3
"""SiPhon example 02 — 4-channel ring-bank WDM demux.

Designs a bank of four add-drop microring filters that demultiplex the CWDM4
grid (1271/1291/1311/1331 nm): the ring circumference is chosen from the
target free spectral range via the group index of a 400 x 220 nm SOI strip,
each ring is then snapped onto its channel by picking an integer azimuthal
mode number. The script prints an FSR/Q/insertion-loss table, the full 4 x 4
drop-port crosstalk matrix from the RingAddDrop spectra, and a thermal-tuning
power budget via the siphon.wdm helpers.

Requires siphon.components and siphon.wdm (in addition to the core modules).
Units: um for wavelength/geometry, dB for loss, mW for heater power.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from siphon.constants import linear_to_db
from siphon.waveguide import Waveguide, bend_loss_db_per_90deg

# Design targets ---------------------------------------------------------
WIDTH_UM = 0.40          # O-band single-mode strip
HEIGHT_UM = 0.22
N_CH = 4
CH_SPACING_UM = 0.020    # CWDM grid pitch
FSR_TARGET_UM = N_CH * CH_SPACING_UM  # one resonance per ring inside the band
KAPPA_POWER = 0.05       # power cross-coupling, both couplers
LOSS_DB_PER_CM = 2.0     # straight-waveguide propagation loss

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


def attempt(label: str, fn, *args, **kwargs):
    """Call a concurrently-developed helper; report instead of crashing on drift."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - demo robustness against API drift
        print(f"  [skipped] {label}: {type(exc).__name__}: {exc}")
        return None


def channel_wavelengths_um(plan) -> list[float]:
    """Channel centers from a ChannelPlan (fallback grid if the API drifted)."""
    for name in ("centers_um", "wavelengths_um", "wavelengths", "channels"):
        wl = getattr(plan, name, None)
        if wl is not None:
            return [float(c) for c in np.atleast_1d(wl)]
    return [1.271, 1.291, 1.311, 1.331]  # CWDM4 fallback


def drop_through_db(ring, wl_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop- and through-port power transmission (dB) for input at 'in'."""
    s = ring.s_params(wl_um)
    i_in = ring.ports.index("in")
    i_drop = ring.ports.index("drop")
    i_thru = ring.ports.index("through")
    drop = np.abs(s[:, i_drop, i_in]) ** 2
    thru = np.abs(s[:, i_thru, i_in]) ** 2
    floor = 1e-15
    return linear_to_db(np.maximum(drop, floor)), linear_to_db(np.maximum(thru, floor))


def find_drop_peak(ring, wl_center_um: float, span_um: float) -> tuple[float, float, float]:
    """(peak wavelength, peak drop transmission dB, FWHM um) near wl_center_um."""
    wl = np.linspace(wl_center_um - span_um / 2, wl_center_um + span_um / 2, 6001)
    drop_db, _ = drop_through_db(ring, wl)
    k = int(np.argmax(drop_db))
    # refine on a fine grid around the coarse peak
    wl_f = np.linspace(wl[k] - 2e-4, wl[k] + 2e-4, 4001)
    drop_db_f, _ = drop_through_db(ring, wl_f)
    kf = int(np.argmax(drop_db_f))
    peak_wl, peak_db = float(wl_f[kf]), float(drop_db_f[kf])
    # FWHM from the -3 dB crossings on a window a few linewidths wide
    wl_w = np.linspace(peak_wl - 2e-3, peak_wl + 2e-3, 20001)
    drop_db_w, _ = drop_through_db(ring, wl_w)
    above = drop_db_w >= peak_db - 3.0103
    idx = np.flatnonzero(above)
    fwhm = float(wl_w[idx[-1]] - wl_w[idx[0]]) if idx.size >= 2 else float("nan")
    return peak_wl, peak_db, fwhm


def design_rings(RingAddDrop, channels_um: list[float]):
    """One ring per channel: FSR-sized circumference snapped to an integer mode."""
    wg = Waveguide(width_um=WIDTH_UM, height_um=HEIGHT_UM)
    rings, designs = [], []
    for ch in channels_um:
        neff0 = wg.neff(ch, "TE")
        ng = wg.group_index(ch, "TE")
        circ_fsr = ch**2 / (ng * FSR_TARGET_UM)      # FSR = wl^2 / (ng * L)
        m = max(1, round(neff0 * circ_fsr / ch))     # resonance: neff * L = m * wl
        circ = m * ch / neff0
        ring = RingAddDrop(
            circumference_um=circ,
            neff0=neff0,
            ng=ng,
            kappa1_power=KAPPA_POWER,
            kappa2_power=KAPPA_POWER,
            loss_db_per_cm=LOSS_DB_PER_CM,
            wl0_um=ch,
        )
        rings.append(ring)
        designs.append({"ch": ch, "neff0": neff0, "ng": ng, "m": m, "circ": circ})
    return rings, designs


def maybe_plot(rings, channels_um: list[float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)

    wl = np.linspace(1.255, 1.347, 6001)
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (ring, ch) in enumerate(zip(rings, channels_um)):
        drop_db, _ = drop_through_db(ring, wl)
        ax.plot(wl * 1e3, drop_db, color=SERIES[i], lw=1.6)
        ax.annotate(
            f"ch{i} {ch * 1e3:.0f}", (ch * 1e3, 1.5), ha="center",
            color=INK, fontsize=8.5,
        )
    ax.set_ylim(-55, 6)
    ax.set_xlabel("wavelength (nm)", color=INK_2)
    ax.set_ylabel("drop transmission (dB)", color=INK_2)
    ax.set_title("CWDM4 ring-bank demux — drop-port spectra", color=INK)
    ax.grid(color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_2)
    fig.tight_layout()
    path = out_dir / "02_ring_demux_spectra.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] saved {path}")


def main() -> int:
    try:
        from siphon import wdm
        from siphon.components import RingAddDrop
    except ImportError as exc:
        print("This example needs siphon.components and siphon.wdm, which are")
        print(f"not built in this checkout yet ({exc}).")
        print("Re-run after integration; example 01 runs on the core modules alone.")
        return 1

    header("SiPhon 02 — 4-channel CWDM ring-bank demux")

    plan = attempt("wdm.ChannelPlan.cwdm4()", wdm.ChannelPlan.cwdm4)
    channels = channel_wavelengths_um(plan) if plan is not None else channel_wavelengths_um(None)
    print("Channel plan (CWDM4): " + ", ".join(f"{c * 1e3:.0f} nm" for c in channels))
    print(f"Waveguide: {WIDTH_UM * 1e3:.0f} x {HEIGHT_UM * 1e3:.0f} nm Si strip, "
          f"SiO2 clad; kappa^2 = {KAPPA_POWER} per coupler, "
          f"{LOSS_DB_PER_CM} dB/cm loss")

    # --- geometry from target FSR ------------------------------------------
    header(f"Ring sizing from target FSR = {FSR_TARGET_UM * 1e3:.0f} nm "
           "(one resonance in band per ring)")
    rings, designs = design_rings(RingAddDrop, channels)
    rows = []
    for i, d in enumerate(designs):
        radius = d["circ"] / (2 * np.pi)
        rows.append([
            f"ch{i}", f"{d['ch'] * 1e3:.0f}", f"{d['neff0']:.4f}", f"{d['ng']:.4f}",
            f"{d['m']}", f"{d['circ']:.3f}", f"{radius:.3f}",
        ])
    table(
        ["ring", "ch (nm)", "n_eff", "n_g", "m", "L (um)", "R (um)"],
        rows, [5, 8, 8, 8, 4, 8, 7],
    )
    fsr_nm = FSR_TARGET_UM * 1e3
    r_min = min(d["circ"] for d in designs) / (2 * np.pi)
    print(f"\nAn {fsr_nm:.0f} nm FSR on the 20 nm CWDM grid forces "
          f"R ~ {r_min:.2f} um, where the bend-loss")
    print(f"heuristic already charges {4 * bend_loss_db_per_90deg(r_min):.2f} dB "
          "per round trip — CWDM's wide grid is hostile")
    print("to ring demuxes; ring banks earn their keep on dense (DWDM) grids.")
    n_max = attempt("wdm.ring_bank_channel_count_limit",
                    wdm.ring_bank_channel_count_limit, fsr_nm, CH_SPACING_UM * 1e3)
    if n_max is not None:
        print(f"Capacity check: {n_max} channels fit in one {fsr_nm:.0f} nm FSR "
              "with the default 1-channel")
        print(f"guard band; we run all {N_CH} by spending the guard "
              "(span 60 nm of 80 nm FSR).")
        try:
            wdm.laser_grid_check(plan, fsr_nm)
            print("wdm.laser_grid_check: OK — no channel aliases onto an "
                  "adjacent resonance order.")
        except Exception as exc:  # noqa: BLE001
            print(f"wdm.laser_grid_check: {exc}")

    # --- measured spectra ----------------------------------------------------
    header("Filter metrics from RingAddDrop spectra")
    peaks = []
    rows = []
    for i, (ring, d) in enumerate(zip(rings, designs)):
        peak_wl, peak_db, fwhm = find_drop_peak(ring, d["ch"], span_um=0.004)
        # measured FSR: locate the next resonance one design-FSR to the red
        fsr_design = d["ch"] ** 2 / (d["ng"] * d["circ"])
        nxt_wl, _, _ = find_drop_peak(ring, peak_wl + fsr_design, span_um=0.6 * fsr_design)
        fsr_meas = nxt_wl - peak_wl
        q = peak_wl / fwhm if fwhm and np.isfinite(fwhm) else float("nan")
        peaks.append(peak_wl)
        rows.append([
            f"ch{i}", f"{d['ch'] * 1e3:.1f}", f"{peak_wl * 1e3:.2f}",
            f"{(peak_wl - d['ch']) * 1e6:+.0f}",
            f"{fsr_design * 1e3:.1f}", f"{fsr_meas * 1e3:.1f}",
            f"{q:.0f}", f"{-peak_db:.2f}",
        ])
    table(
        ["ring", "target", "peak", "detune", "FSR des", "FSR meas", "Q", "IL"],
        rows, [5, 7, 8, 7, 8, 9, 7, 6],
    )
    print("        (nm)      (nm)     (pm)     (nm)      (nm)          (dB)")
    print("\n'FSR des' is wl^2/(ng L) at the channel; 'FSR meas' is the spacing to")
    print("the next resonance order to the red, where the local FSR is larger")
    print("(FSR grows as wl^2) — both are consistent.")
    print("Q here is the loaded Q; insertion loss at the drop peak stays low")
    print("because the ring is nearly critically coupled (kappa1 = kappa2 and")
    print("small round-trip loss).")

    # --- crosstalk matrix ----------------------------------------------------
    header("Drop-port crosstalk matrix (ring row, channel column, dB)")
    wl_peaks = np.array(peaks)
    xt = np.zeros((N_CH, N_CH))
    for i, ring in enumerate(rings):
        drop_db, _ = drop_through_db(ring, wl_peaks)
        xt[i, :] = drop_db
    rows = []
    for i in range(N_CH):
        rows.append([f"ring{i}"] + [f"{xt[i, j]:.1f}" for j in range(N_CH)])
    table(
        ["", *(f"ch{j}" for j in range(N_CH))],
        rows, [6, 7, 7, 7, 7],
    )
    print("\nAggregate crosstalk per channel "
          "(exact power sum | wdm worst-case bound):")
    exact_worst, bound_worst = [], []
    for j in range(N_CH):
        others_db = np.array([xt[i, j] for i in range(N_CH) if i != j])
        exact = float(linear_to_db(np.sum(10.0 ** (others_db / 10.0))))
        iso_db = float(-np.max(others_db))  # weakest suppression, as isolation
        bound = attempt("wdm.aggregate_crosstalk_db",
                        wdm.aggregate_crosstalk_db, iso_db, N_CH - 1)
        exact_worst.append(exact)
        bound_worst.append(bound if bound is not None else exact)
        b = f"{bound:6.1f} dB" if bound is not None else "     --"
        print(f"  ch{j} ({channels[j] * 1e3:.0f} nm): {exact:6.1f} dB | {b}")
    print(f"Worst channel: {max(exact_worst):.1f} dB exact "
          f"({max(bound_worst):.1f} dB bound); a 4-lane demux typically "
          "targets < -20 dB.")

    # --- thermal tuning --------------------------------------------------------
    header("Thermal tuning budget (siphon.wdm helpers)")
    fsr_nm = FSR_TARGET_UM * 1e3
    ng_mid = designs[N_CH // 2]["ng"]
    shift = attempt("wdm.resonance_shift_nm_per_k",
                    wdm.resonance_shift_nm_per_k, channels[N_CH // 2], ng_mid)
    p_worst = attempt("wdm.tuning_power_mw", wdm.tuning_power_mw, fsr_nm / 2.0)
    p_exp = attempt("wdm.expected_tuning_power_mw",
                    wdm.expected_tuning_power_mw, fsr_nm)
    eff = getattr(wdm, "TUNING_EFFICIENCY_NM_PER_MW", 0.25)
    print(f"Heater efficiency assumed : {eff} nm/mW (integrated heater, no undercut)")
    if shift is not None:
        print(f"Resonance thermal shift   : {shift * 1e3:.1f} pm/K "
              f"(so ~{fsr_nm / 2 / shift:.0f} K swing for a half-FSR hop)")
    if p_worst is not None:
        print(f"Worst-case lock (FSR/2)   : {p_worst:7.1f} mW/ring")
    if p_exp is not None:
        print(f"Expected lock (fab spread): {p_exp:7.1f} mW/ring, "
              f"{N_CH * p_exp:.0f} mW for the {N_CH}-ring bank")
    print("\nThe 80 nm FSR that made the rings tiny also makes their tuning")
    print("budget brutal — expected power scales with FSR/4. On a dense grid")
    print("(one FSR of a few nm) the same lock costs a few mW per ring, which")
    print("is why ring banks and DWDM go together.")

    maybe_plot(rings, channels)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
