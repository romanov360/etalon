export const meta = {
  name: 'validation-extensions-build',
  description: 'Build 7 SiPhon extensions from the 2026-07-24 validation roadmap with disjoint file ownership',
  phases: [{ title: 'Build', detail: 'one agent per module, disjoint files' }],
}

const COMMON = `You are building an extension to SiPhon, an open-source silicon photonics toolkit at /Users/tr/prog/silicon-photonics (Python 3.13, uv, numpy/scipy ONLY — no new dependencies). Layout: src/siphon/, tests/, examples/.

HOUSE RULES (read 2-3 existing modules first to absorb style):
- Frozen dataclasses where natural; every public function/class gets a docstring stating the physics, the formula, units, conventions, and validity limits; ValueError on bad input; module docstring summarizing scope and honesty limits ("architecture-level, not signoff" tone).
- Units: wavelengths um, power mW/dBm as named, dB for losses/penalties, nm where suffixed _nm.
- Wavelength-dependent S-params convention (if relevant): s_params(wl)->(n_wl,n,n) complex, propagation phase exp(-1j*2*pi*neff*L/wl).
- Tests: physics anchored to analytic/known values or independent re-derivations (bisection etc.), not to the code's own output. Seeded rngs.
- OWNERSHIP: you may ONLY create/modify the files listed under YOUR FILES. Do NOT touch src/siphon/__init__.py, README.md, pyproject.toml, docs/, or any other src/tests/examples file. Integration is done centrally afterwards.
- Verify with: uv run pytest <your test files> -q  (must pass). Also run uv run pytest tests/ -q and report any failures you did NOT cause in notes (do not fix them).
- Your final message is machine-read: return the structured output only.`

const SCHEMA = {
  type: 'object',
  required: ['status', 'files', 'tests_passed', 'n_tests', 'api', 'notes'],
  properties: {
    status: { enum: ['ok', 'partial', 'failed'] },
    files: { type: 'array', items: { type: 'string' } },
    tests_passed: { type: 'boolean' },
    n_tests: { type: 'integer' },
    api: { type: 'string', description: 'public API added, signatures + one-liners' },
    notes: { type: 'string', description: 'caveats, design decisions, expected-failure tests you did not cause' },
  },
}

const TASKS = [
  {
    key: 'shot-noise+pareto',
    prompt: `${COMMON}

TASK: Add shot noise (and dark current) to the receiver sensitivity in src/siphon/link.py, plus a margin-vs-pJ/bit Pareto sweep example.

YOUR FILES: src/siphon/link.py (modify), src/siphon/constants.py (modify: add ELEMENTARY_CHARGE_C = 1.602176634e-19), tests/test_shot_noise.py (new), examples/06_architecture_pareto.py (new).

Read link.py fully first — note how rin_penalty_db was recently generalized to evaluate noise at P_top = OMA*ER/(ER-1); shot noise follows the same top-eye convention.

EXACT SPEC — shot_penalty_db. The link is OMA-domain, thermal-limited baseline u0 = 2 q (M-1) i_n / R (watts), i_n = S_i*sqrt(f_n), f_n = NOISE_BANDWIDTH_FRACTION*baud. Adding shot noise evaluated self-consistently at the sensitivity point: with k = ER/(ER-1) (=1 at infinite ER), q_e elementary charge, I_d dark current (A), the top-eye requirement
    u R / (2 (M-1)) = q * sqrt( i_n^2 + 2 q_e (R k u + I_d) f_n )
is quadratic in u = required OMA (W):  a u^2 - b u - c = 0 with
    a = (R / (2 q (M-1)))^2,  b = 2 q_e R k f_n,  c = i_n^2 + 2 q_e I_d f_n
    => u = (b + sqrt(b^2 + 4 a c)) / (2 a)
    shot_penalty_db = 10 log10(u / u0)  >= 0.
Implement as free function shot_penalty_db(pd: Photodiode, tia: Tia, sig: Signaling, extinction_ratio_db: float = math.inf) -> float (raise ValueError for ER <= 0 dB), plus LinkBudget.shot_penalty_db property passing the modulator's ER. Add it into sensitivity_oma_dbm (thermal + RIN + shot + explicit penalties) and as a row in report() right after the RIN row. Update the LinkBudget/module docstrings: shot noise now modeled at the top level; dark current now used. RIN and shot compose as independent dB add-ons (standard, mildly pessimistic).

KNOWN FALLOUT you must NOT fix (report in notes): tests/test_review_regressions.py::test_required_oma_composition_charges_er_exactly_once and ::test_margin_from_first_principles pin the OLD composition and will fail; they are updated centrally. Do not edit any existing test file.

tests/test_shot_noise.py must include: (1) an independent bisection solve of the implicit equation matching the closed form to 1e-9 dB at DR4-like numbers; (2) penalty -> 0 as RIN... as responsivity noise dominates (e.g. huge i_n); (3) monotonic increase with dark current and with baud; (4) finite-ER penalty > infinite-ER penalty; (5) both presets still have positive margin, and report() contains the shot row; (6) ER<=0 raises.

examples/06_architecture_pareto.py: ~60-line sweep over (levels in {2,4}, baud in a few values, n_lanes in {4,8,16}, laser_shared_by in {1,4}) built on preset-like LinkBudget variants + energy_per_bit_pj; print an aligned table of (config, margin_db, pJ/bit) and the Pareto-efficient subset (max margin, min pJ/bit); save a scatter to examples/out/ following the matplotlib style of examples/03. It must run: uv run python examples/06_architecture_pareto.py. Report the new preset margins (DR4, CPO) in notes.`,
  },
  {
    key: 'extract',
    prompt: `${COMMON}

TASK: E1 — parameter extraction: fit SiPhon models to measured transmission spectra. This is the on-ramp to calibrating models against wafer data (docs/THESIS.md), so API clarity matters most.

YOUR FILES: src/siphon/extract.py (new), tests/test_extract.py (new), examples/05_parameter_extraction.py (new).

Read src/siphon/components.py, src/siphon/circuit.py, and one example first.

SPEC: module siphon.extract built on scipy.optimize.least_squares.
- FitResult frozen dataclass: params (dict[str,float]), cost, residual_rms_db (or matching domain), success (bool), nfev, message; a report() string.
- fit_transmission(build, params0: dict[str,float], wl_um, measured, *, inport: str, outport: str, bounds: dict[str, tuple[float,float]] | None = None, domain: "power_db" | "field_mag" = "power_db", **fixed) -> FitResult.
  build(**{**fixed, **params}) returns either a component model (has .ports/.s_params) or a Circuit; extract the named inport->outport transmission (components: port index lookup; Circuit: .transmission). Residual in the chosen domain. Multiple observations: allow measured to be a dict {(inport,outport): array} variant OR keep single-path and add fit_transmission_multi — your call, keep it simple and documented.
- fit_ring_add_drop(wl_um, through_db, drop_db, x0: dict | None = None) -> FitResult: convenience fitting (kappa1_power, kappa2_power, loss_db_per_cm, neff0, ng) of components.RingAddDrop to through+drop power spectra jointly, with sensible default bounds (kappas in (1e-4, 0.9), loss >= 0). DOCUMENT identifiability honestly: neff0 is only identifiable modulo the resonance-order ambiguity (shifts of wl/L), ng requires >1 FSR in the data; recommend spanning several FSRs.
- Raise ValueError on shape mismatches, unknown ports, params0/bounds key mismatches.

tests/test_extract.py: synthesize RingAddDrop spectra with known params over ~3 FSRs, add seeded Gaussian noise (e.g. 0.05 dB rms), fit from perturbed x0, assert recovery: kappas/loss within a few %, ng within 1%, and the refit residual_rms close to the injected noise floor. Also: a generic fit_transmission test on a simpler model (e.g. DirectionalCoupler coupling), and error-path tests.

examples/05_parameter_extraction.py: the publishable demo — synthesize noisy through/drop spectra ("pretend wafer probe trace"), fit, print true vs recovered table, save an overlay plot (data points vs fitted curve, residuals) to examples/out/ in the style of the other examples. Must run cleanly.`,
  },
  {
    key: 'correlated-mc',
    prompt: `${COMMON}

TASK: E2 — correlated variation and module-level yield in src/siphon/montecarlo.py. A CPO module is max-of-N: one bad lane among N scraps the module; within-die variation is mostly common-mode with a small differential spread.

YOUR FILES: src/siphon/montecarlo.py (modify), tests/test_module_yield.py (new).

Read montecarlo.py fully first and MATCH its existing conventions exactly (Normal/Uniform API, NaN = failed corner counted as yield loss, nan-aware stats, seeded rng, report() style).

SPEC:
- CommonDifferential frozen dataclass: mean, sigma_common, sigma_diff, optional low/high truncation of the TOTAL value (resample like the existing truncated Normal does). One shared common draw per module trial + one independent differential draw per lane: value[trial, lane] = mean + common[trial] + diff[trial, lane].
- run_module(metric, params: dict[str, CommonDifferential | Normal | Uniform], n_lanes: int, n_modules: int, seed=...) -> ModuleYieldResult. Plain Normal/Uniform params are drawn fully independently per lane (sigma_common=0 equivalent); CommonDifferential shares the common part across the lanes of a module. metric(**scalar_params) -> float is evaluated per lane (vectorize with a loop or np.vectorize; document cost).
- ModuleYieldResult: samples shape (n_modules, n_lanes); lane_yield_above/below(spec) and module_yield_above/below(spec) = P(ALL lanes pass); NaN lane => that lane fails and the module fails; n_failed count; nan-aware mean/std/percentile of the lane metric; report() string showing lane vs module yield side by side (that gap IS the known-good-die problem — say so in the docstring).
- Keep every existing API working unchanged.

tests/test_module_yield.py, all seeded, with ANALYTIC anchors:
(1) sigma_common=0, metric=identity, threshold at mean: module_yield ~= lane_yield^n_lanes (binomial tolerance);
(2) sigma_diff=0: module_yield == lane_yield exactly (all lanes identical);
(3) truncation bounds respected; (4) NaN from metric fails lane AND module and is counted; (5) reproducibility with same seed; (6) mixing CommonDifferential with plain Normal works; (7) lane_yield >= module_yield always.`,
  },
  {
    key: 'ring-assignment',
    prompt: `${COMMON}

TASK: E3 — ring-bank channel-assignment (barrel-shift) tuning-power optimizer in src/siphon/wdm.py. A ring bank can cyclically rotate its channel-to-ring assignment; the optimal rotation absorbs common-mode fabrication offset nearly for free, leaving differential spread + FSR/(2N) quantization.

YOUR FILES: src/siphon/wdm.py (modify), tests/test_ring_assignment.py (new).

Read wdm.py fully first (esp. tuning_power_mw, expected_tuning_power_mw and their red-shift-only convention: heaters only red-shift, so a blue-shift request costs the long way around, i.e. detune taken mod FSR).

MODEL: N rings serve N channels equally spaced by delta = fsr_nm / N within one FSR; ring i nominally resonates at channel i and its as-built resonance sits offset_nm[i] red (+) or blue (-) of that nominal. Under barrel shift r (ring i serves channel (i+r) mod N, all rotations are valid WDM plans), the required red-shift heating for ring i is
    h_i(r) = (r * delta - offset_nm[i]) mod fsr_nm      [red-shift-only]
    h_i(r) = min(x, fsr_nm - x), x = same mod           [bidirectional=True]
Total power P(r) = sum_i h_i(r) / efficiency_nm_per_mw; enumerate all N rotations, return the best.

SPEC: optimize_ring_assignment(offsets_nm: array-like, fsr_nm: float, efficiency_nm_per_mw: float = TUNING_EFFICIENCY_NM_PER_MW, bidirectional: bool = False) -> RingAssignment frozen dataclass with: rotation (int), per_ring_mw (tuple), total_mw, mean_mw_per_ring, and for comparison naive_total_mw (rotation 0). Validate fsr_nm > 0, efficiency > 0, len(offsets) >= 1, offsets finite. Docstring: cross-reference expected_tuning_power_mw as the unassigned pessimistic bound; note the optimizer's assumption of identical rings on one bus and that r is shared by the whole bank (it is a WDM-plan relabeling, not per-ring).

tests/test_ring_assignment.py with ANALYTIC anchors:
(1) zero offsets -> rotation 0, zero power;
(2) pure common-mode offset delta_c on all rings: best total <= N * delta (quantization bound) and << the naive (r=0) cost when delta_c ~ fsr/2 — verify the several-fold drop;
(3) exact tiny case N=2 computed by hand;
(4) optimal total never exceeds naive total, for random seeded offsets;
(5) bidirectional <= red-shift-only for the same inputs;
(6) invariance: adding exactly delta to every offset shifts the best rotation by 1 (mod N) with identical total power.`,
  },
  {
    key: 'reliability',
    prompt: `${COMMON}

TASK: E5 — laser reliability and thermal derating module: src/siphon/reliability.py (new). Lasers are the dominant CPO field-failure mechanism; this module turns quotable facts into reproducible arithmetic.

YOUR FILES: src/siphon/reliability.py (new), tests/test_reliability.py (new).

SPEC (all closed-form, no new deps):
- BOLTZMANN_EV_PER_K = 8.617333262e-5.
- arrhenius_acceleration(t_use_c: float, t_stress_c: float, activation_energy_ev: float) -> float: AF = exp( (Ea/k) * (1/T_use - 1/T_stress) ), temperatures converted to kelvin; AF > 1 when t_stress > t_use. VERIFIED ANCHOR: Ea = 0.97 eV, 25 -> 60 C gives 52.8x (pin to ~1%).
- fit_from_mttf(mttf_hours) and mttf_from_fit(fit): FIT = 1e9 / MTTF_hours (exponential/random-failure regime).
- module_fit(fit_per_device: float, n_devices: int) -> float: series system, sum. And module_mttf_hours convenience.
- survival_probability(t_hours, fit) = exp(-fit * t_hours * 1e-9).
- Lognormal wear-out: wearout_fraction(t_hours, t50_hours, sigma) = Phi(ln(t/t50)/sigma) using math.erf; document that random (FIT) and wear-out (lognormal) are distinct regimes and compose as competing risks: total survival = survival_random * (1 - wearout_fraction).
- module_survival(t_hours, n_lasers, fit_per_laser, t50_hours=None, sigma=None) -> float: all-lasers-alive probability, competing risks per laser, independent lasers.
- Thermal: wpe_derated(wpe_ref: float, t_ref_c: float, t_c: float, slope_per_k: float) -> float: first-order linear derating wpe_ref * (1 - slope_per_k*(t_c - t_ref_c)), clipped to (0, wpe_ref* if t < t_ref allow >ref but cap at 1.0]; raise if result <= 0 (out of the linear model's validity). Docstring: pass the result into link.Laser(wpe=...) — this module deliberately does NOT solve the self-heating loop; junction temperature is an input.
- Docstrings cite the standard forms (Arrhenius/JEDEC JESD85-style FIT algebra, lognormal wear-out for laser diodes).

tests/test_reliability.py: pin 52.8x anchor; AF monotonic in Ea and in stress T; fit/mttf round-trip; module_fit scales linearly and module MTTF = device MTTF / n; wearout_fraction(t50) == 0.5 exactly; competing-risks survival <= each factor; module_survival == per-laser^n for identical independent lasers; wpe_derated recovers wpe_ref at t_ref, decreases with T, raises when derated to <= 0.`,
  },
  {
    key: 'eim-diagnostic',
    prompt: `${COMMON}

TASK: Near-cutoff EIM diagnostic in src/siphon/waveguide.py. The EIM caveat ("conservative near cutoff") was validated quantitatively (TE1 first guided at 316 nm width under EIM vs 450-500 nm full-vectorial); move that prose caveat into the API as a warning.

YOUR FILES: src/siphon/waveguide.py (modify), tests/test_eim_diagnostic.py (new).

Read waveguide.py fully first, including how the EIM composes the vertical slab solve with the horizontal swapped-polarization solve, and note the lru_cache on neff.

SPEC:
- class EimAccuracyWarning(UserWarning) at module level, exported.
- Module constant NEAR_CUTOFF_NEFF_MARGIN = 5e-3 (documented: when the horizontally-solved n_eff sits within this of the side/cladding effective index the mode is marginally guided, exactly where EIM is least trustworthy and biased early/conservative).
- In the EIM path (where the horizontal slab problem is solved), if solved n_eff - n_side < NEAR_CUTOFF_NEFF_MARGIN, issue warnings.warn(EimAccuracyWarning) with a message naming the geometry, mode, margin, and advising a full-vectorial check. NOTE the lru_cache: the warning fires on first computation only — document that in the docstring, do not remove the cache.
- ZERO numerical behavior change: every returned value must be bit-identical to before (warning only). Do not alter slab_neffs.

tests/test_eim_diagnostic.py:
(1) standard well-guided geometry (0.5 x 0.22 strip, TE, 1.55) computes with NO EimAccuracyWarning (use warnings.simplefilter('error', EimAccuracyWarning));
(2) find a marginally guided case that triggers it (e.g. narrow width just above modal cutoff for TE0, or TE1 near its EIM cutoff width ~0.32-0.35 um; probe your candidate first with a quick uv run python -c check) and assert pytest.warns(EimAccuracyWarning);
(3) returned n_eff for the standard geometry equals the pre-change value (pin the number to ~1e-9 by computing it BEFORE your edit with uv run python -c "..." and hard-coding it);
(4) warning message mentions 'vectorial'.`,
  },
  {
    key: 'fd-mode-solver',
    prompt: `${COMMON}

TASK: E6 — semi-vectorial finite-difference mode solver, src/siphon/fdmode.py (new). Purpose: SiPhon self-validates its own EIM approximation instead of pointing users off-library. This is the hardest task; if you cannot meet the validation gates, ship what passes and set status='partial' with honest notes rather than shipping junk.

YOUR FILES: src/siphon/fdmode.py (new), tests/test_fdmode.py (new).

Read src/siphon/waveguide.py and src/siphon/materials.py first (reuse materials for n(lambda); mirror the Waveguide geometry parameters).

SPEC:
- solve_modes(width_um, height_um, wl_um, *, slab_um=0.0, core='si', cladding='sio2', box='sio2', polarization='TE', n_modes=2, dx_um=0.02, pad_um=1.5) -> list[FdMode], FdMode frozen-ish dataclass: neff (float), polarization, field (2D ndarray, dominant E component, normalized to max |E| = 1), x_um, y_um (1D coords).
- Method: semi-vectorial finite differences on the cross-section (Stern-type scheme): for quasi-TE (dominant Ex) use the standard semi-vectorial discretization with index-averaged coefficients at vertical interfaces (continuity of Ex handled via 1/n^2 averaging across x-interfaces); quasi-TM analogous across y-interfaces. Uniform grid, Dirichlet (field=0) at the padded domain edge. Assemble scipy.sparse matrix, solve with scipy.sparse.linalg.eigs (shift-invert around k0^2*n_core^2, sigma slightly below max) for the few largest-neff modes; keep only neff > max(n_cladding_eff...) i.e. guided: n_eff > max(n_clad, n_box) and n_eff < n_core.
- Geometry: strip (or rib via slab_um) centered horizontally: box below, cladding above/sides, core rectangle; rib adds the residual slab layer of core material.
- Document: semi-vectorial (no polarization coupling), budgeting-to-design grade, expected ~1e-3 n_eff accuracy for well-guided modes at dx=0.02; NOT full-vectorial signoff.

VALIDATION GATES (all must be tests):
(1) SLAB LIMIT: width_um = 6.0, height 0.22, TE: fundamental neff must match waveguide.slab_neffs(n_si, n_sio2, n_sio2, 0.22, 1.55, 'TE')[0] within 2e-3 (wide strip -> 1D slab).
(2) CONVERGENCE: 0.5 x 0.22 strip TE0 at dx 0.03 vs 0.015: |delta neff| < 2e-3.
(3) PHYSICAL ORDERING: TE0 neff in (n_sio2(1.55), n_si(1.55)) and TE0 > TE1 when both guided (e.g. width 0.8).
(4) EIM COMPARISON (the punchline — make it a test with loose bounds and a docstring): for 0.5 x 0.22 TE0, fd neff differs from waveguide.Waveguide(0.5, 0.22).neff(1.55) by less than 0.15 but more than 1e-4 (EIM known biased); assert fd < eim (EIM overestimates neff for this geometry) ONLY if you verify that numerically first — if the sign differs, assert the |difference| bounds only and note it.
(5) symmetry: field of TE0 is even in x about the center (correlation of field with its x-mirror > 0.99).
Keep runtime sane: default test grid coarse enough that the whole test file runs < 60 s.`,
  },
]

phase('Build')
log(`launching ${TASKS.length} build agents (disjoint file ownership)`)
const results = await parallel(
  TASKS.map(t => () =>
    agent(t.prompt, { label: `build:${t.key}`, phase: 'Build', schema: SCHEMA })
  )
)
const out = {}
TASKS.forEach((t, i) => { out[t.key] = results[i] })
const failed = TASKS.filter((t, i) => !results[i] || results[i].status === 'failed').map(t => t.key)
log(failed.length ? `done; failed/missing: ${failed.join(', ')}` : 'done; all agents reported')
return out