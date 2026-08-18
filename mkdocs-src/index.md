# Etalon

An open-source silicon photonics design toolkit for photonic integrated
circuits (PICs) — material dispersion models, waveguide mode solvers,
S-parameter photonic circuit simulation, WDM system math, closed-form
optical link budgets for AI-datacenter interconnects (co-packaged optics
and pluggable optical transceivers), and Monte Carlo corner/yield analysis.
Pure Python (numpy/scipy), Apache-2.0.

*Etalon* — French *étalon*, "measuring standard" — is also the name of the
resonant optical cavity this toolkit's ring components model.

## Install

```bash
git clone https://github.com/romanov360/etalon.git && cd etalon
uv sync
uv run pytest          # 434 tests
```

```bash
pip install etalon   # once published to PyPI
```

## Quick taste

```python
from etalon import link, waveguide

wg = waveguide.Waveguide(width_um=0.5, height_um=0.22)
print(wg.neff(1.55), wg.group_index(1.55))   # 2.49, 4.02 (EIM)

cpo = link.preset_cpo_optical_io()
print(cpo.report())                          # full dB waterfall, TX -> margin
print(cpo.margin_db)                         # +15.0 dB nominal
```

## What's inside

| Module | What it does |
|---|---|
| [`etalon.materials`](modules/materials.md) | Sellmeier refractive-index models (Si, SiO₂, Si₃N₄) with validity ranges, thermo-optic coefficients, material group index |
| [`etalon.waveguide`](modules/waveguide.md) | Exact three-layer slab TE/TM mode solver; effective-index-method strip/rib solver with near-cutoff accuracy warning; n_eff, group index, chromatic dispersion; bend-loss heuristic |
| [`etalon.fdmode`](modules/fdmode.md) | Semi-vectorial finite-difference mode solver on the cross-section — the in-repo check on the EIM's own approximation |
| [`etalon.components`](modules/components.md) | S-parameter models: straight, directional coupler, Y-branch, phase shifters, grating coupler, all-pass & add-drop rings (Bogaerts 2012), analytic MZI |
| [`etalon.circuit`](modules/circuit.md) | Netlist-level S-matrix circuit solver (subnetwork reduction with proper feedback handling — rings built from couplers work) |
| [`etalon.link`](modules/link.md) | IM-DD link-budget engine: Q-from-BER (NRZ/PAM4), thermal-limited sensitivity, ER/RIN/shot/crosstalk penalties, waterfall reports, energy-per-bit breakdowns, CPO & pluggable presets |
| [`etalon.wdm`](modules/wdm.md) | Channel plans (CWDM4/LR4/DWDM grids), ring FSR/channel-count limits, thermal-tuning power, barrel-shift channel-assignment optimizer, aggregate crosstalk |
| [`etalon.thermal`](modules/thermal.md) | Ring-to-ring thermal crosstalk: spatial heat-coupling kernel from ring pitch, self-consistent coupled heater-power solve, and a screening bound that flags when a thermally-isolated channel assignment is physically unlockable once neighbor heat is accounted for |
| [`etalon.montecarlo`](modules/montecarlo.md) | Monte Carlo corner analysis: truncated-normal/uniform parameters, correlated (common+differential) variation, per-lane and all-lanes-pass module yield, sensitivity ranking, and whole-bank (jointly-coupled) yield for metrics that can't be decomposed lane-by-lane |
| [`etalon.isi`](modules/isi.md) | E-O-E bridge: computed PAM eye-closure (ISI) penalty of a passive filter response S21(λ) — exhaustive de Bruijn time-domain eye, feeds `LinkBudget.penalties_db` |
| [`etalon.equalize`](modules/equalize.md) | Closed-form zero-forcing FFE tap solver: symbol-spaced pulse response of a passive filter, Toeplitz tap solve, and the noise-enhancement penalty equalization always costs |
| [`etalon.extract`](modules/extract.md) | Parameter extraction: least-squares fitting of component/circuit models to measured transmission spectra (the calibration on-ramp) |
| [`etalon.touchstone`](modules/touchstone.md) | Touchstone (.sNp) file I/O: read/write measured or foundry-exported S-parameter data — the real-data on-ramp into `etalon.extract`/`etalon.circuit` |
| [`etalon.reliability`](modules/reliability.md) | Laser reliability arithmetic: Arrhenius acceleration, FIT/MTTF, lognormal wear-out, N-laser module survival, wall-plug-efficiency thermal derating |

## Scope and honesty

Everything here is closed-form or semi-analytic physics — fast, transparent,
and dependency-free, accurate to the level a system architect needs for
budgeting and corner exploration. It is **not** a substitute for
full-vectorial mode solvers or FDTD for device signoff, and the presets are
published-data-plausible illustrations, not measured silicon.

Every module has gone through at least one independent adversarial review
before being considered done. See the [Changelog](changelog.md) for what
that process has actually caught.

## Links

- [GitHub repository](https://github.com/romanov360/etalon)
- [docs/RESEARCH.md](https://github.com/romanov360/etalon/blob/master/docs/RESEARCH.md) — the industry deep-research report behind this project
- [docs/THESIS.md](https://github.com/romanov360/etalon/blob/master/docs/THESIS.md) — the startup thesis this repo is Product 0 of
