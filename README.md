# Etalon

[![tests](https://github.com/romanov360/etalon/actions/workflows/tests.yml/badge.svg)](https://github.com/romanov360/etalon/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/romanov360/etalon/branch/master/graph/badge.svg)](https://codecov.io/gh/romanov360/etalon)

**An open-source silicon photonics design toolkit for photonic integrated circuits
(PICs)** — material dispersion models, waveguide mode solvers, S-parameter photonic
circuit simulation, WDM system math, closed-form optical link budgets for AI-datacenter
interconnects (co-packaged optics and pluggable optical transceivers), and Monte Carlo
corner/yield analysis. Pure Python (numpy/scipy), Apache-2.0.

(*Etalon* — French *étalon*, "measuring standard" — is also the name of the resonant
optical cavity this toolkit's ring components model. The repo predates the name; both
now point at the same thing.)

If you searched for: *photonic circuit simulator Python*, *co-packaged optics link
budget*, *silicon photonics mode solver*, *ring resonator / MZI S-parameters*,
*optical transceiver margin calculator*, *photonic EDA*, or *PIC yield Monte Carlo* —
this is that.

Etalon is the open-core engine ("Product 0") of the startup thesis developed in this repo:
the loudest unsolved bottleneck in the AI-datacenter photonics buildout is predicting,
testing, and yielding co-packaged optics — see [docs/THESIS.md](docs/THESIS.md) for the
plan and [docs/RESEARCH.md](docs/RESEARCH.md) for the 21,000-word industry deep-research
report behind it (July 2026; 28-agent research run, raw transcripts in
[docs/research/](docs/research/)).

## Install & test

From source (current):

```bash
git clone https://github.com/romanov360/etalon.git && cd etalon
uv sync
uv run pytest          # 434 tests
```

```bash
pip install etalon      # Python 3.10+
```

## What's inside

| Module | What it does |
|---|---|
| `etalon.materials` | Sellmeier refractive-index models (Si, SiO₂, Si₃N₄) with validity ranges, thermo-optic coefficients, material group index |
| `etalon.waveguide` | Exact three-layer slab TE/TM mode solver; effective-index-method strip/rib solver with near-cutoff accuracy warning; n_eff, group index, chromatic dispersion; bend-loss heuristic |
| `etalon.fdmode` | Semi-vectorial finite-difference mode solver on the cross-section — the in-repo check on the EIM's own approximation |
| `etalon.components` | S-parameter models: straight, directional coupler, Y-branch, phase shifters, grating coupler, all-pass & add-drop rings (Bogaerts 2012), analytic MZI |
| `etalon.circuit` | Netlist-level S-matrix circuit solver (subnetwork reduction with proper feedback handling — rings built from couplers work) |
| `etalon.link` | IM-DD link-budget engine: Q-from-BER (NRZ/PAM4), thermal-limited sensitivity, ER/RIN/shot/crosstalk/PDL penalties, waterfall reports, energy-per-bit breakdowns, CPO & pluggable presets |
| `etalon.wdm` | Channel plans (CWDM4/LR4/DWDM grids), ring FSR/channel-count limits, thermal-tuning power, barrel-shift channel-assignment optimizer, aggregate crosstalk |
| `etalon.thermal` | Ring-to-ring thermal crosstalk: spatial heat-coupling kernel from ring pitch, self-consistent coupled heater-power solve, and a screening bound that flags when a thermally-isolated channel assignment is physically unlockable once neighbor heat is accounted for |
| `etalon.montecarlo` | Monte Carlo corner analysis: truncated-normal/uniform parameters, correlated (common+differential) variation, per-lane and all-lanes-pass module yield, sensitivity ranking, and whole-bank (jointly-coupled) yield for metrics — like thermal-crosstalk-coupled ring tuning — that can't be decomposed lane-by-lane |
| `etalon.isi` | E-O-E bridge: computed PAM eye-closure (ISI) penalty of a passive filter response S21(λ) — exhaustive de Bruijn time-domain eye, feeds `LinkBudget.penalties_db` |
| `etalon.equalize` | Closed-form zero-forcing FFE tap solver: symbol-spaced pulse response of a passive filter, Toeplitz tap solve, and the noise-enhancement penalty equalization always costs — the priced version of `etalon.isi`'s "treat as an upper bound with an FFE receiver" caveat |
| `etalon.extract` | Parameter extraction: least-squares fitting of component/circuit models to measured transmission spectra (the calibration on-ramp) |
| `etalon.touchstone` | Touchstone (.sNp) file I/O: read/write measured or foundry-exported S-parameter data (MA/DB/RI, all frequency units) into the same `ports` + `s_params(wl)` protocol as any Etalon component — the real-data on-ramp into `etalon.extract`/`etalon.circuit` |
| `etalon.reliability` | Laser reliability arithmetic: Arrhenius acceleration, FIT/MTTF, lognormal wear-out, N-laser module survival, wall-plug-efficiency thermal derating |

## Quick taste

```python
from etalon import link, waveguide

wg = waveguide.Waveguide(width_um=0.5, height_um=0.22)
print(wg.neff(1.55), wg.group_index(1.55))   # 2.49, 4.02 (EIM)

cpo = link.preset_cpo_optical_io()
print(cpo.report())                          # full dB waterfall, TX -> margin
print(cpo.margin_db)                         # +15.0 dB nominal
```

## Examples

```bash
uv run python examples/01_waveguide_explorer.py    # mode solving, single-mode limits
uv run python examples/02_ring_wdm_filter.py       # 4-channel ring demux design
uv run python examples/03_cpo_vs_pluggable.py      # flagship: two link waterfalls + pJ/bit
uv run python examples/04_monte_carlo_yield.py     # lane yield + dominant-variation ranking
uv run python examples/05_parameter_extraction.py  # fit a ring model to noisy "measured" spectra
uv run python examples/06_architecture_pareto.py   # margin vs pJ/bit Pareto sweep
uv run python examples/07_filter_isi.py            # demux passband -> computed ISI penalty
uv run python examples/08_thermal_crosstalk.py     # ring-bank thermal crosstalk vs. tuning power
uv run python examples/09_touchstone_roundtrip.py  # write/read a .s2p file, fit a model to it
uv run python examples/10_wdm_bank_yield.py        # whole-bank yield: fab variation x thermal crosstalk
uv run python examples/11_ffe_equalization.py      # zero-forcing FFE vs. an unequalized ISI penalty
uv run python examples/12_published_number_reproduction.py  # NVIDIA's 3.5x claim, checked honestly
uv run python examples/13_pdk_import.py            # real SiEPIC foundry data -> Touchstone -> jointly-fit model
```

## How Etalon relates to other photonics tools

| Tool | What it is | Relationship |
|---|---|---|
| [gdsfactory](https://github.com/gdsfactory/gdsfactory) | Layout/PDK framework for PICs | Complementary — gdsfactory draws the chip; Etalon budgets and yields the *system* (link margins, corners). No layout in Etalon. |
| [SAX](https://github.com/flaport/sax) | JAX-based S-parameter circuit solver | Overlapping on circuit solving; Etalon adds the physics models, link-budget/WDM/yield layer, with zero JAX/GPU dependency. |
| Ansys Lumerical (INTERCONNECT/FDTD/MODE), Synopsys OptoCompiler | Commercial photonic EDA | Etalon is the open, scriptable alternative for *system-level budgeting and corner exploration* — not a replacement for full-vectorial device signoff. |
| VPIphotonics / Keysight ADS optical links | Commercial E-O-E link simulation | Etalon covers the closed-form IM-DD subset (waterfalls, sensitivities, penalties, pJ/bit) openly and transparently. |
| [MEEP](https://github.com/NanoComp/meep) / [Tidy3D](https://www.flexcompute.com/tidy3d/) | FDTD electromagnetic solvers | Upstream of Etalon: use them to extract device S-parameters; use Etalon to compose circuits and links from them. |

## Scope and honesty

Everything here is closed-form or semi-analytic physics — fast, transparent, and
dependency-free, accurate to the level a system architect needs for budgeting and corner
exploration. It is **not** a substitute for full-vectorial mode solvers or FDTD for device
signoff, and the presets are published-data-plausible illustrations, not measured silicon.
The point of the open engine is exactly that gap: the proprietary layer the thesis builds
toward is calibration against measured wafer/test data.

Of the five bottlenecks [docs/RESEARCH.md](docs/RESEARCH.md) names as the industry's
unclaimed middle (lasers, packaging/fiber-attach, test/known-good-die, ring thermal
tuning, yield/variability), Etalon now directly models three: thermal tuning
(`etalon.thermal`), yield/variability (`etalon.montecarlo`, including whole-bank yield),
and the test/known-good-die on-ramp (`etalon.touchstone` for real measured data,
`etalon.montecarlo.run_bank` for the max-of-N bank-scrap statistic). Lasers and
packaging/fiber-attach are hardware problems outside a software toolkit's scope and
stay that way on purpose.

## Repo map

```
src/etalon/          the toolkit
tests/               434 tests, physics anchored to analytic/known values
examples/            thirteen runnable demos
examples/data/       vendored real foundry-process S-parameter data
                      (MIT-licensed; see PROVENANCE.md for source/citation)
docs/RESEARCH.md     industry deep-research report (July 2026)
docs/THESIS.md       ranked startup theses + recommended play
docs/research/       full raw evidence: agent transcripts, judge verdicts, sources
CHANGELOG.md         every module's adversarial-review record: what was
                      checked, what was found, what got fixed
mkdocs-src/          API reference site source (mkdocs + mkdocstrings,
                      generated from the module docstrings below)
```

The API reference builds locally with `uv run mkdocs serve` (live-reloading
dev server) or `uv run mkdocs build` (static site in `site/`). Not yet
deployed to a public URL.

Every module here has gone through at least one independent adversarial
review before being considered done — a fresh reviewer re-derives the
physics from first principles and tries to break the implementation, not
just reads it and nods along. Across seven review rounds since the initial
47-agent pass, this has caught real bugs: a ring phase-convention error, an
RIN-noise level miscalculation, a bulk-group-delay eye-search bug, a
Touchstone option-line/comment parsing gap, and a silent-garbage FFE
cursor-lock failure, among others — all fixed and pinned as regression
tests. See [CHANGELOG.md](CHANGELOG.md) for the detailed record of each pass.

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
code style, and what "done" means in this repo (short version: every new
physical model needs an analytic-anchor test and a stated scope/honesty-limits
docstring section).

## License & citation

Apache-2.0 (see [LICENSE](LICENSE)). To cite Etalon, use [CITATION.cff](CITATION.cff).
