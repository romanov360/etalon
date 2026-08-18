# Changelog

Every module in this repo has gone through at least one independent adversarial
review before being considered done — a fresh reviewer re-derives the physics
from first principles and tries to break the implementation, rather than just
reading it and nodding along. This file is the detailed record of those
passes: what was checked, what was found, what got fixed. `README.md` states
the headline (every module reviewed); this is the evidence.

## 2026-08-17 — Etalon rename

Renamed the package from `siphon` (PyPI: `siphon-photonics`) to `etalon`
(PyPI: `etalon`, fully available). The repo has always been named `etalon`;
the code and PyPI listing now match it. *Étalon* (French, "measuring
standard") is also the name of the resonant optical cavity the ring
components in this toolkit model.

## 2026-08-17 — `equalize`: closed-form zero-forcing FFE tap solver

Closes the gap `isi`'s own docstring names: its ISI penalty assumes an
unequalized receiver and calls itself "only an upper bound... with an
FFE-based receiver." `equalize` solves zero-forcing FFE taps for the same
passive S21(λ) input and reports both the residual ISI *and* the noise
enhancement it costs — an FFE is never a free lunch, and reporting only the
ISI side would silently claim one.

Went through two adversarial rounds, more than any other module this cycle:

- **Round 1** found two real silent-garbage bugs. A bulk group delay or a
  channel with a dominant postcursor tap could lock the "cursor" onto
  near-zero energy and report a plausible-looking but meaningless result
  (one case: +63 dB noise enhancement, zero error raised). A fixed
  zero-padding guard against FFT wraparound gave an artificially optimistic
  residual-ISI reading on a high-Q-ring stress case. Both fixed: cursor
  location is now a global peak search with an explicit reach check;
  padding now adaptively doubles until the response provably decays rather
  than trusting a guessed constant.
- **Round 2** (follow-up verification) confirmed both fixes hold under
  harder pressure — fractional delays, forced cap-trips, combined
  delay-plus-long-memory cases — and found one narrow residual edge
  (near-tied multipath peaks can flip the cursor choice discontinuously;
  documented, doesn't occur on realistic single-ring channels) plus a test
  that looked like it covered the padding fix but didn't. Both addressed.

`examples/11_ffe_equalization.py`: on a real demux at 106.25 GBd, the FFE
recovers +0.65 dB net margin — real but modest, because the noise-enhancement
cost (3.48 dB) nearly cancels the ISI benefit (4.12 dB) it buys back.

## 2026-08-17 — `montecarlo.run_bank` / `BankParam`: whole-bank yield

`run_module` calls its metric once per lane, independently — fine when each
lane's outcome depends only on its own draws. A WDM ring bank breaks that
assumption: `thermal.solve_coupled_powers` solves a linear system across
every ring on the bus at once, so one ring's achievable margin depends on
every other ring's draw in the same trial. `run_bank` gives the metric the
whole bank's draws per trial instead of one lane's.

Adversarial review independently re-verified the common+differential
sampling statistics, all-NaN-row failure semantics, and yield-statistic
NaN-handling by large-sample simulation. Found and fixed a validation gap
(`BankParam`'s name field wasn't key-matched against its dict key the way
`Normal`/`Uniform`'s is — a copy-paste footgun) and an overstated claim in
the accompanying example's narrative (`sigma_common`'s effect on bank yield
is real, just ~15x smaller than `sigma_diff`'s — the claim said "barely
moves," which the reviewer's own sweep showed was an overstatement).

`examples/10_wdm_bank_yield.py`: the barrel-shift assignment optimizer
absorbs common-mode fabrication drift almost for free (that's its job), but
not differential (ring-to-ring) spread — a fab team chasing yield via litho
overlay control alone will plateau; ring-to-ring uniformity and thermal
isolation are what actually move the number.

## 2026-08-17 — `touchstone`: Touchstone (.sNp) file I/O

Real-measured-data on-ramp: reads/writes the industry-standard S-parameter
file format (VNAs, foundry PDKs, ATE) into the same `ports` + `s_params(wl)`
protocol every component model already speaks, so a measured file drops
straight into `extract.fit_transmission` or a `circuit.Circuit`.

Adversarial review hand-verified the notoriously error-prone 2-port column
order (file order is S11, S21, S12, S22 — not row-major, unlike N≥3 ports)
by inspecting raw written bytes rather than trusting round-trip tests alone.
Found two real parsing gaps: a second, *disagreeing* option line would
silently misconvert earlier data rows (now raises — but an *agreeing*
duplicate is still tolerated, per spec), and legal inline trailing comments
on data lines (`<data> ! note`) were rejected instead of stripped, a real
compatibility gap against actual VNA/PDK exports. Both fixed and pinned;
added coverage for descending-frequency files and `Circuit.connect()`-cascaded
measured components.

## 2026-08-17 — `thermal`: ring-to-ring thermal crosstalk

A ring's heater warms the shared substrate, and that heat reaches
neighboring rings too — `wdm.optimize_ring_assignment` prices each ring's
own tuning power but treats every ring as thermally isolated. `thermal`
adds a spatial heat-coupling kernel and a closed-form self-consistent
coupled-power solve.

Adversarial review re-derived the coupled linear system from physical first
principles and found no defects in the core algebra; tightened the
exponential-kernel docstring's near-field honesty caveat (it's a
tail-matched approximation of a more rigorous Bessel-K0 form, not a derived
near-field solution) and added edge-case regression tests (n=1, coincident
rings, a realistic 8-ring bank at scale).

`examples/08_thermal_crosstalk.py`: at realistic pitch/thermal-isolation
values, a crosstalk-blind ring assignment can be physically **unlockable**,
not just power-inefficient — neighbor heat alone can overshoot a ring's
target, and a resistive heater cannot pull heat back out. This is a failure
mode `wdm` alone cannot see.

## 2026-07-25 — `isi`: E-O-E bridge (eye-closure / ISI penalty)

Bridges the frequency-domain circuit solver to the link budget: takes a
composed passive filter response S21(λ) and computes the worst-case PAM eye
closure it inflicts, as a dB penalty ready for `LinkBudget.penalties_db`.
Exhaustive de Bruijn time-domain eye search — deterministic, no
random-pattern luck.

Adversarial review converged on one high-severity finding: bulk group delay
≥ 1 UI mislabeled every sample and reported a fake closed eye for a
physically penalty-free delay (a 2 mm routing waveguide broke it). Fixed by
estimating the delay from magnitude-weighted adjacent-bin phase differences
and removing it as pure linear phase before the eye search, plus scanning
integer symbol offsets since a ring's stored energy can park the eye optimum
whole symbols away from any correlation-based alignment.

## 2026-07-24 — Roadmap extensions + adversarial review

Six modules built in one pass from the validation pass's suggested roadmap:
`extract` (parameter extraction / calibration on-ramp), `montecarlo`'s
correlated common+differential variation and module-level (max-of-N) yield,
`wdm.optimize_ring_assignment` (barrel-shift channel assignment), `reliability`
(Arrhenius/FIT/MTTF laser reliability), `fdmode` (semi-vectorial
finite-difference mode solver, the in-repo check on the EIM's own bias), and
an EIM near-cutoff accuracy warning. Plus a shot-noise gap-fill in the link
budget and a margin-vs-pJ/bit Pareto sweep harness.

A 7-reviewer adversarial pass across all six found 14 findings, all
resolved. The substantive one: composing RIN and shot noise as independent
dB penalties is optimistic (both are signal-dependent) by up to ~1 dB at
high baud / low extinction ratio — fixed by solving the joint
thermal+RIN+shot quadratic exactly and booking the difference as a
`noise_interaction_db` waterfall row.

## 2026-07-24 — Independent validation pass

An external validation pass re-derived the core numerics from first
principles and cross-checked them against the repo's own solvers. Confirmed
the RIN penalty algebra, the heat-only tuning model, the EIM near-cutoff
caveat, and the circuit solver's correctness (bit-identical against an
independent batched re-implementation). Found four issues:

- RIN noise was evaluated at OMA instead of the true top level; with finite
  extinction ratio the true top level is higher, and the penalty roughly
  doubles at low ER (~0.56 dB hidden optimism in the DR4 preset).
- `Circuit.s_params` solved the interconnection problem in a per-wavelength
  Python loop; rewritten as a stacked batch operation (~4.7x faster,
  bit-identical output).
- `materials.DN_DT_SI` was exported but unused; `wdm` duplicated the literal
  value instead of referencing it.
- Degenerate (near-zero) index contrast in `slab_neffs` silently returned an
  empty mode list and leaked a numpy warning instead of raising.

All four fixed and pinned in `tests/test_validation_regressions.py`.

## 2026-07-23/24 — Initial adversarial review

The original engine (materials, waveguide, components, circuit, link, wdm)
went through a 47-agent adversarial review: 5 lenses, every finding attacked
by a 3-refuter panel before being accepted. 8 distinct confirmed defects,
including a ring phase-convention conjugation, an extinction-ratio
double-count, and a laser-sharing energy-conservation bug. All fixed and
pinned in `tests/test_review_regressions.py`.

## 2026-07-23 — Opening

Initial public commit: waveguide/circuit/link/WDM engine, Monte Carlo corner
analysis, the CPO-vs-pluggable flagship example, and the industry research +
startup thesis (`docs/RESEARCH.md`, `docs/THESIS.md`) this repo is built
around.
