# SiPhon

**An open silicon photonics design toolkit** — material dispersion models, waveguide mode
solvers, S-parameter circuit simulation, WDM system math, closed-form optical link budgets
for AI-datacenter interconnects, and Monte Carlo corner/yield analysis.

SiPhon is the open-core engine ("Product 0") of the startup thesis developed in this repo:
the loudest unsolved bottleneck in the AI-datacenter photonics buildout is predicting,
testing, and yielding co-packaged optics — see [docs/THESIS.md](docs/THESIS.md) for the
plan and [docs/RESEARCH.md](docs/RESEARCH.md) for the 21,000-word industry deep-research
report behind it (July 2026; 28-agent research run, raw transcripts in
[docs/research/](docs/research/)).

## Install & test

```bash
uv sync
uv run pytest          # 156 tests
```

## What's inside

| Module | What it does |
|---|---|
| `siphon.materials` | Sellmeier refractive-index models (Si, SiO₂, Si₃N₄) with validity ranges, thermo-optic coefficients, material group index |
| `siphon.waveguide` | Exact three-layer slab TE/TM mode solver; effective-index-method strip/rib solver; n_eff, group index, chromatic dispersion; bend-loss heuristic |
| `siphon.components` | S-parameter models: straight, directional coupler, Y-branch, phase shifters, grating coupler, all-pass & add-drop rings (Bogaerts 2012), analytic MZI |
| `siphon.circuit` | Netlist-level S-matrix circuit solver (subnetwork reduction with proper feedback handling — rings built from couplers work) |
| `siphon.link` | IM-DD link-budget engine: Q-from-BER (NRZ/PAM4), thermal-limited sensitivity, ER/RIN/crosstalk penalties, waterfall reports, energy-per-bit breakdowns, CPO & pluggable presets |
| `siphon.wdm` | Channel plans (CWDM4/LR4/DWDM grids), ring FSR/channel-count limits, thermal-tuning power, aggregate crosstalk |
| `siphon.montecarlo` | Monte Carlo corner analysis: truncated-normal/uniform parameters, parametric yield, sensitivity ranking, ASCII distribution reports |

## Quick taste

```python
from siphon import link, waveguide

wg = waveguide.Waveguide(width_um=0.5, height_um=0.22)
print(wg.neff(1.55), wg.group_index(1.55))   # 2.49, 4.02 (EIM)

cpo = link.preset_cpo_optical_io()
print(cpo.report())                          # full dB waterfall, TX -> margin
print(cpo.margin_db)                         # +15.1 dB nominal
```

## Examples

```bash
uv run python examples/01_waveguide_explorer.py    # mode solving, single-mode limits
uv run python examples/02_ring_wdm_filter.py       # 4-channel ring demux design
uv run python examples/03_cpo_vs_pluggable.py      # flagship: two link waterfalls + pJ/bit
uv run python examples/04_monte_carlo_yield.py     # lane yield + dominant-variation ranking
```

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
tests/               156 tests, physics anchored to analytic/known values
examples/            four runnable demos
docs/RESEARCH.md     industry deep-research report (July 2026)
docs/THESIS.md       ranked startup theses + recommended play
docs/research/       full raw evidence: agent transcripts, judge verdicts, sources
```

The codebase survived a 47-agent adversarial review (5 lenses, every finding
attacked by a 3-refuter panel): 8 distinct confirmed defects — including a ring
phase-convention conjugation, an extinction-ratio double-count, and a
laser-sharing energy-conservation bug — were fixed and pinned with regression
tests (`tests/test_review_regressions.py`); transcripts in
`docs/research/raw/workflow-adversarial-review/`.
