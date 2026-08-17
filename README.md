# SiPhon

**An open-source silicon photonics design toolkit for photonic integrated circuits
(PICs)** — material dispersion models, waveguide mode solvers, S-parameter photonic
circuit simulation, WDM system math, closed-form optical link budgets for AI-datacenter
interconnects (co-packaged optics and pluggable optical transceivers), and Monte Carlo
corner/yield analysis. Pure Python (numpy/scipy), Apache-2.0.

If you searched for: *photonic circuit simulator Python*, *co-packaged optics link
budget*, *silicon photonics mode solver*, *ring resonator / MZI S-parameters*,
*optical transceiver margin calculator*, *photonic EDA*, or *PIC yield Monte Carlo* —
this is that.

SiPhon is the open-core engine ("Product 0") of the startup thesis developed in this repo:
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
uv run pytest          # 400 tests
```

The distribution name is `siphon-photonics` (the bare name `siphon` on PyPI belongs to
Unidata's meteorology client); the import name stays `siphon`:

```bash
pip install siphon-photonics   # once published to PyPI
```

## What's inside

| Module | What it does |
|---|---|
| `siphon.materials` | Sellmeier refractive-index models (Si, SiO₂, Si₃N₄) with validity ranges, thermo-optic coefficients, material group index |
| `siphon.waveguide` | Exact three-layer slab TE/TM mode solver; effective-index-method strip/rib solver with near-cutoff accuracy warning; n_eff, group index, chromatic dispersion; bend-loss heuristic |
| `siphon.fdmode` | Semi-vectorial finite-difference mode solver on the cross-section — the in-repo check on the EIM's own approximation |
| `siphon.components` | S-parameter models: straight, directional coupler, Y-branch, phase shifters, grating coupler, all-pass & add-drop rings (Bogaerts 2012), analytic MZI |
| `siphon.circuit` | Netlist-level S-matrix circuit solver (subnetwork reduction with proper feedback handling — rings built from couplers work) |
| `siphon.link` | IM-DD link-budget engine: Q-from-BER (NRZ/PAM4), thermal-limited sensitivity, ER/RIN/shot/crosstalk penalties, waterfall reports, energy-per-bit breakdowns, CPO & pluggable presets |
| `siphon.wdm` | Channel plans (CWDM4/LR4/DWDM grids), ring FSR/channel-count limits, thermal-tuning power, barrel-shift channel-assignment optimizer, aggregate crosstalk |
| `siphon.thermal` | Ring-to-ring thermal crosstalk: spatial heat-coupling kernel from ring pitch, self-consistent coupled heater-power solve, and a screening bound that flags when a thermally-isolated channel assignment is physically unlockable once neighbor heat is accounted for |
| `siphon.montecarlo` | Monte Carlo corner analysis: truncated-normal/uniform parameters, correlated (common+differential) variation, per-lane and all-lanes-pass module yield, sensitivity ranking |
| `siphon.isi` | E-O-E bridge: computed PAM eye-closure (ISI) penalty of a passive filter response S21(λ) — exhaustive de Bruijn time-domain eye, feeds `LinkBudget.penalties_db` |
| `siphon.extract` | Parameter extraction: least-squares fitting of component/circuit models to measured transmission spectra (the calibration on-ramp) |
| `siphon.touchstone` | Touchstone (.sNp) file I/O: read/write measured or foundry-exported S-parameter data (MA/DB/RI, all frequency units) into the same `ports` + `s_params(wl)` protocol as any SiPhon component — the real-data on-ramp into `siphon.extract`/`siphon.circuit` |
| `siphon.reliability` | Laser reliability arithmetic: Arrhenius acceleration, FIT/MTTF, lognormal wear-out, N-laser module survival, wall-plug-efficiency thermal derating |

## Quick taste

```python
from siphon import link, waveguide

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
```

## How SiPhon relates to other photonics tools

| Tool | What it is | Relationship |
|---|---|---|
| [gdsfactory](https://github.com/gdsfactory/gdsfactory) | Layout/PDK framework for PICs | Complementary — gdsfactory draws the chip; SiPhon budgets and yields the *system* (link margins, corners). No layout in SiPhon. |
| [SAX](https://github.com/flaport/sax) | JAX-based S-parameter circuit solver | Overlapping on circuit solving; SiPhon adds the physics models, link-budget/WDM/yield layer, with zero JAX/GPU dependency. |
| Ansys Lumerical (INTERCONNECT/FDTD/MODE), Synopsys OptoCompiler | Commercial photonic EDA | SiPhon is the open, scriptable alternative for *system-level budgeting and corner exploration* — not a replacement for full-vectorial device signoff. |
| VPIphotonics / Keysight ADS optical links | Commercial E-O-E link simulation | SiPhon covers the closed-form IM-DD subset (waterfalls, sensitivities, penalties, pJ/bit) openly and transparently. |
| [MEEP](https://github.com/NanoComp/meep) / [Tidy3D](https://www.flexcompute.com/tidy3d/) | FDTD electromagnetic solvers | Upstream of SiPhon: use them to extract device S-parameters; use SiPhon to compose circuits and links from them. |

## Scope and honesty

Everything here is closed-form or semi-analytic physics — fast, transparent, and
dependency-free, accurate to the level a system architect needs for budgeting and corner
exploration. It is **not** a substitute for full-vectorial mode solvers or FDTD for device
signoff, and the presets are published-data-plausible illustrations, not measured silicon.
The point of the open engine is exactly that gap: the proprietary layer the thesis builds
toward is calibration against measured wafer/test data.

## Repo map

```
src/siphon/          the toolkit
tests/               400 tests, physics anchored to analytic/known values
examples/            nine runnable demos
docs/RESEARCH.md     industry deep-research report (July 2026)
docs/THESIS.md       ranked startup theses + recommended play
docs/research/       full raw evidence: agent transcripts, judge verdicts, sources
```

The codebase survived a 47-agent adversarial review (5 lenses, every finding
attacked by a 3-refuter panel): 8 distinct confirmed defects — including a ring
phase-convention conjugation, an extinction-ratio double-count, and a
laser-sharing energy-conservation bug — were fixed and pinned with regression
tests (`tests/test_review_regressions.py`); transcripts in
`docs/research/raw/workflow-adversarial-review/`. A second, independent
validation pass (2026-07-24) confirmed the core algebra and surfaced four more
findings — notably RIN noise evaluated at OMA instead of the true top level
(~0.56 dB hidden optimism at low ER) and a batched rewrite of the circuit
solve — all rolled in and pinned (`tests/test_validation_regressions.py`,
notes in `docs/research/validation-notes-2026-07-24.md`). The roadmap
extensions that followed (extraction, module yield, ring assignment,
reliability, FD solver, shot noise) went through their own 7-reviewer
adversarial pass: 14 findings, all resolved — including an exact joint
RIN+shot noise solve replacing the optimistic independent-dB composition
(`tests/test_extension_review_regressions.py`). The E-O-E bridge
(`siphon.isi`, 2026-07-25) and ring-to-ring thermal crosstalk
(`siphon.thermal`, 2026-08-17) each went through an independent
adversarial re-derivation of their core algebra from physical first
principles; thermal crosstalk's review found no defects but tightened
the exponential-kernel near-field honesty caveat and added edge-case
regression tests (n=1, coincident rings, realistic 8-ring bank scale).
The Touchstone file reader (`siphon.touchstone`, 2026-08-17) went
through an adversarial review that hand-verified the notoriously
error-prone 2-port column order (S11/S21/S12/S22, not row-major) by
inspecting raw written bytes; it found two real parsing gaps — a
disagreeing duplicate option line silently misconverting prior rows,
and rejection of legal inline data-line comments — both fixed and
pinned, plus added coverage for descending-frequency files and
`Circuit.connect()`-cascaded measured components.

## License & citation

Apache-2.0 (see [LICENSE](LICENSE)). To cite SiPhon, use [CITATION.cff](CITATION.cff).
