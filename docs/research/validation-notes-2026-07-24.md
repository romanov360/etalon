# External validation notes — 2026-07-24

Independent validation pass on the `siphon` package at commit dd25350, received
by Tristan on 2026-07-24 and rolled into the codebase the same day. This file
preserves the findings as received, with the disposition of each appended.
Regression pins: `tests/test_validation_regressions.py`.

## What the pass validated (and held up)

The reviewer re-derived the core numerics from first principles and
cross-checked numerically against the repo's own solvers (uv, CPython 3.13,
156/156 tests passing at baseline):

- **RIN penalty algebra is exact** — solving the implicit top-eye equation
  numerically reproduces `rin_penalty_db` to 1e-9 dB, including the (M-1) PAM
  scaling and the unreachable-BER guard.
- **ER is charged exactly once** — the OMA-domain composition is internally
  consistent; the explanatory docstring is correct.
- **Heat-only tuning model is right** — FSR/2 (red-shift-only) vs FSR/4
  (bidirectional) verified at 32.0 / 16.0 mW for FSR 16 nm, 0.25 nm/mW.
- **The documented EIM caveat is quantitatively accurate** — TE1 of a 220 nm
  strip first guided at 316 nm width under EIM vs the 450-500 nm
  full-vectorial cutoff; the "conservative (early)" warning in examples/01 is
  exactly right.
- **Circuit solver output is trustworthy** — an independent batched
  re-implementation of the subnetwork-growth math produced bit-identical
  results (max |diff| = 0) on the composed ring-add-drop across 5000
  wavelengths.

## Findings

### F1 (model accuracy) — RIN noise evaluated at OMA, not the true top level

`rin_penalty_db` modeled the top-level RIN with P_top ~ OMA. With finite
extinction ratio the true top level is P_top = OMA * ER/(ER-1) > OMA; at
ER = 5 dB the ratio is 1.46 and the penalty roughly doubles:

- DR4 preset (ER 5 dB, PAM4 53.125 GBd, RIN -140): 0.397 dB -> 0.961 dB
- CPO preset (ER 4 dB, NRZ 32 GBd, RIN -150): 0.003 dB -> 0.007 dB

~0.56 dB of hidden optimism inside the DR4 preset's nominal margin.

**Disposition: CONFIRMED and FIXED.** Reproduced exactly (0.397 -> 0.961 dB);
independently re-verified here by solving the implicit top-eye equation
(0.960937 dB). `rin_penalty_db` gained an `extinction_ratio_db` parameter
(default infinity = old behavior); `LinkBudget` passes the modulator's ER.
Preset margins moved: DR4 +9.83 -> +9.26 dB, CPO +15.06 -> +15.05 dB.

### F2 (performance) — per-wavelength loop in Circuit.s_params

The interconnection problem was solved per wavelength in a Python loop, with a
full-SVD `np.linalg.cond` guard per wavelength dominating the cost. Everything
was already shaped for stacked numpy operations. Measured by the reviewer:
219 ms -> 46 ms (guard kept, 4.7x) -> 10 ms (guard rethought, 22x) at 5000
wavelengths.

**Disposition: CONFIRMED and FIXED (4.7x tier).** Reproduced ~4x locally with
bit-identical output (max |diff| = 0.0). The solve, cond guard, and matmuls
are now stacked over the wavelength axis; the raise-on-singular contract is
unchanged and now reports the *first* singular wavelength index. The 22x tier
(replacing the cond guard with a residual check) was deliberately not taken —
premature until dense sweeps are a measured bottleneck.

### F3 (maintenance) — DN_DT_SI exported but unused; wdm duplicated the literal

`materials.DN_DT_SI` had no computational consumers; `wdm.resonance_shift_nm_per_k`
hardcoded `1.86e-4 * 0.85` while its docstring cross-referenced the constant.

**Disposition: CONFIRMED and FIXED.** The default is now
`materials.DN_DT_SI * STRIP_TE_CONFINEMENT`, with the 0.85 confinement factor
named as a documented `wdm` module constant.

### F4 (hygiene) — degenerate index contrast in slab_neffs

Index contrast below the 2e-9 bracketing epsilon inverted the bracket:
silently returned `[]` (formally dropping the always-guided fundamental of a
symmetric slab) and leaked a numpy `RuntimeWarning` from a negative sqrt.
Not a physical regime (real platforms sit >= 1e-3 above it).

**Disposition: CONFIRMED and FIXED.** Explicit `if lo >= hi: return []` guard
before probing; docstring notes the degenerate regime. No warnings leak
(pinned with `warnings.simplefilter("error")`).

## Scope extensions suggested (roadmap, not defects)

Recorded verbatim as candidate work, ranked by the reviewer's
value-per-effort against docs/THESIS.md. UPDATE (later 2026-07-24): all of
these except E4 were implemented in a 7-agent build workflow
(docs/research/raw/workflow-validation-extensions/) — E1 -> siphon.extract +
examples/05, E2 -> montecarlo.CommonDifferential/run_module, E3 ->
wdm.optimize_ring_assignment, E5 -> siphon.reliability, E6 -> siphon.fdmode,
plus the shot-noise gap-fill (link.shot_penalty_db, presets moved to DR4
+9.00 / CPO +14.95 dB), the near-cutoff EimAccuracyWarning, and the
margin-vs-pJ/bit Pareto harness (examples/06). E4 (E-O-E co-sim bridge) was
deliberately deferred: highest risk of pseudo-accurate output for the effort.

1. **E1 — Parameter extraction** (thesis-critical): fit component/circuit
   models to measured spectra via least squares; the on-ramp to the
   calibration layer and measured Monte Carlo distributions. ~1 week.
2. **E2 — Correlated variation + module-level yield**: common+differential
   parameter decomposition; P(all N lanes pass) vs P(one lane passes) — the
   known-good-die problem. Days.
3. **E3 — Ring-bank channel-assignment optimizer**: barrel-shift assignment
   absorbs common-mode offset; tuning power drops several-fold vs the FSR/2
   bound. Days, given E2.
4. **E4 — E-O-E co-sim seed**: map composed S(lambda) to baseband H(f) and
   compute an ISI penalty into `penalties_db` instead of the hand-entered
   placeholder. 1-2 weeks.
5. **E5 — Laser reliability/thermal module**: Arrhenius FIT (E_a ~ 0.97 eV
   gives ~53x acceleration for 25->60 C junction rise), lognormal wear-out,
   N-laser aggregation, wpe(T). Days to a week.
6. **E6 — Semi-vectorial finite-difference mode solver**: ~200-300 lines,
   self-validates the EIM bias (316 nm vs 450-500 nm TE1 cutoff) in-repo.
   1-2 weeks.

Plus gap-fills: shot-noise term in receiver sensitivity (dark current is
carried but unused), near-cutoff EIM diagnostic warning, and a
margin-vs-pJ/bit Pareto sweep harness stitching E2/E5 together.
