"""Etalon 04 — Monte Carlo yield of a CPO optical-I/O link.

A deterministic link budget answers "does the nominal link close?"; the
question a CPO program actually lives or dies on is "what fraction of
manufactured lanes close, over temperature and process?". This script
declares plausible per-lane variations for the CPO preset and reports
parametric yield and the parameters that dominate the margin spread.
"""

from dataclasses import replace

from etalon import link, montecarlo as mc


def build_link(p: dict[str, float]) -> link.LinkBudget:
    base = link.preset_cpo_optical_io()
    return link.LinkBudget(
        laser=replace(base.laser, power_dbm=p["laser_power_dbm"], rin_db_hz=p["rin_db_hz"]),
        modulator=replace(
            base.modulator,
            insertion_loss_db=p["ring_il_db"],
            extinction_ratio_db=p["ring_er_db"],
        ),
        path=[
            link.LossElement("laser fiber-to-chip coupler", p["coupler_db"]),
            link.LossElement("tx on-chip routing", 0.5),
            link.LossElement("tx chip-to-fiber coupler", p["coupler_db"]),
            link.LossElement("jumper fiber + connectors", 0.3),
            link.LossElement("rx fiber-to-chip coupler", p["coupler_db"]),
            link.LossElement("rx on-chip routing", 0.5),
        ],
        photodiode=replace(base.photodiode, responsivity_a_per_w=p["responsivity"]),
        tia=base.tia,
        signaling=base.signaling,
        penalties_db=dict(base.penalties_db or {}) | {"wavelength_drift": p["drift_penalty_db"]},
    )


PARAMS = [
    # ELS line power: spec window
    mc.Uniform("laser_power_dbm", low=5.5, high=7.5),
    # every fiber-to-chip interface shares a process-dependent coupling loss
    mc.Normal("coupler_db", mean=1.5, sigma=0.3, low=0.8),
    # ring modulator on-state loss and ER at speed
    mc.Normal("ring_il_db", mean=1.5, sigma=0.3, low=0.5),
    mc.Normal("ring_er_db", mean=4.0, sigma=0.5, low=2.5),
    # Ge photodiode responsivity
    mc.Normal("responsivity", mean=1.0, sigma=0.07, low=0.6, high=1.2),
    # laser RIN across the supplier distribution
    mc.Uniform("rin_db_hz", low=-152.0, high=-145.0),
    # residual thermal-lock error of the ring across traffic transients
    mc.Normal("drift_penalty_db", mean=0.5, sigma=0.2, low=0.0),
]


def main() -> None:
    print("=" * 72)
    print("Etalon 04 — Monte Carlo yield, CPO optical-I/O lane (32G NRZ, rings)")
    print("=" * 72)

    nominal = link.preset_cpo_optical_io()
    print(f"\nnominal margin: {nominal.margin_db:+.2f} dB   (deterministic waterfall: 03_cpo_vs_pluggable.py)\n")

    result = mc.run(
        lambda p: build_link(p).margin_db,
        PARAMS,
        n=20_000,
        seed=2026,
        metric_name="link margin (dB)",
    )
    print(result.report(threshold=0.0))

    print()
    for extra in (1.0, 2.0, 3.0):
        print(
            f"  yield with {extra:.0f} dB of extra reserved margin: "
            f"{100.0 * result.yield_above(extra):6.2f} %"
        )
    print(
        "\nReading: the sensitivity line names the variation that dominates lane\n"
        "yield — in a fiber-attach-limited process that is the coupler loss, which\n"
        "is exactly the 50-80%-of-module-cost packaging bottleneck from the\n"
        "research (docs/RESEARCH.md, 'bottlenecks')."
    )


if __name__ == "__main__":
    main()
