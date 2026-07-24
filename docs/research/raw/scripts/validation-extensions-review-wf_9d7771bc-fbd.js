export const meta = {
  name: 'validation-extensions-review',
  description: 'Adversarial review of the 7 new SiPhon extension modules',
  phases: [{ title: 'Refute', detail: 'one adversarial physicist per module' }],
}

const COMMON = `You are an adversarial reviewer of newly added code in SiPhon, a silicon photonics toolkit at /Users/tr/prog/silicon-photonics. Your job is to REFUTE: find real physics errors, wrong formulas, broken conventions, API traps, or tests that pin wrong behavior. Assume the author was competent but rushed; hunt for the subtle failure, not style nits.

METHOD:
- Read the target code AND its tests fully. Re-derive every formula from first principles.
- Check convention consistency with the rest of the package (read neighboring modules): OMA-domain budgets, exp(-1j*2*pi*neff*L/wl) phase, red-shift-only heaters, NaN = failed corner, units (um/nm/dBm/mW/GBd).
- Run independent numeric cross-checks with uv run python (scratch scripts ONLY under /tmp — you are STRICTLY READ-ONLY in the repo; do NOT edit, create, or delete ANY file under /Users/tr/prog/silicon-photonics, not even experimentally; a tampering audit runs after you).
- A finding is only reportable if you numerically reproduced the failure or can cite the exact line contradicting a derivation. Classify: severity 'high' (wrong numbers in plausible use), 'medium' (wrong in edge cases / misleading API), 'low' (doc/consistency).
- Empty findings list is a fine outcome; do not manufacture findings.`

const SCHEMA = {
  type: 'object',
  required: ['module', 'findings', 'checks_run'],
  properties: {
    module: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['severity', 'file', 'claim', 'evidence'],
        properties: {
          severity: { enum: ['high', 'medium', 'low'] },
          file: { type: 'string' },
          line: { type: 'integer' },
          claim: { type: 'string' },
          evidence: { type: 'string', description: 'numeric repro or derivation, concrete' },
          suggested_fix: { type: 'string' },
        },
      },
    },
    checks_run: { type: 'string', description: 'what you re-derived and cross-checked, compact' },
  },
}

const TARGETS = [
  { key: 'shot-noise', files: 'src/siphon/link.py (shot_penalty_db + LinkBudget integration + report), src/siphon/constants.py, tests/test_shot_noise.py, examples/06_architecture_pareto.py', hints: 'Verify the quadratic solve u=(b+sqrt(b^2+4ac))/(2a) against the implicit top-eye equation independently; check k=ER/(ER-1) top-level convention matches rin_penalty_db; check penalty>=0 always, dark-current units nA->A, f_n=0.75*baud, and that sensitivity_oma_dbm/report() compose thermal+RIN+shot+penalties consistently. Check the Pareto example arithmetic (pJ/bit, Pareto front extraction).' },
  { key: 'extract', files: 'src/siphon/extract.py, tests/test_extract.py, examples/05_parameter_extraction.py', hints: 'Check residual domain math (power_db vs field_mag), port resolution for both component models and Circuit, bounds/params0 key validation, the neff0 coarse pre-scan (resonance-order ambiguity), and whether FitResult.residual_rms_db is honest for the dict-measured multi-path case. Try to construct a fit that silently returns garbage with success=True.' },
  { key: 'correlated-mc', files: 'src/siphon/montecarlo.py (CommonDifferential, run_module, ModuleYieldResult; existing API must be unchanged), tests/test_module_yield.py', hints: 'Verify variance structure: Var=sigma_c^2+sigma_d^2, within-module covariance sigma_c^2; truncation resampling bias; NaN scraps module; module_yield == lane_yield^N under independence (test statistically); check existing Normal/Uniform/run() behavior is untouched (diff against git HEAD).' },
  { key: 'ring-assignment', files: 'src/siphon/wdm.py (optimize_ring_assignment, RingAssignment; existing functions must be unchanged), tests/test_ring_assignment.py', hints: 'Re-derive h_i(r)=(r*delta - offset_i) mod fsr for red-shift-only heaters (sign convention: positive offset = as-built resonance sits red of nominal — check against tuning_power_mw/expected_tuning_power_mw conventions and resonance_shift docstrings). Verify the claimed tie structure (totals differ by integer FSR multiples) and that bidirectional uses min(x, fsr-x). Brute-force check small N in /tmp.' },
  { key: 'reliability', files: 'src/siphon/reliability.py, tests/test_reliability.py', hints: 'Verify Arrhenius AF direction (use vs stress), the 52.8x anchor independently, FIT=1e9/MTTF, series-system sum, lognormal CDF via erf (Phi(ln(t/t50)/sigma)), competing-risks composition, wpe_derated clipping/validity. Check Celsius->kelvin conversions and that module_survival is per-laser^n for identical lasers.' },
  { key: 'eim-diagnostic', files: 'src/siphon/waveguide.py (EimAccuracyWarning, NEAR_CUTOFF_NEFF_MARGIN, neff path), tests/test_eim_diagnostic.py', hints: 'The hard requirement was ZERO numerical change — verify against git HEAD (git diff HEAD -- src/siphon/waveguide.py) that only warning logic was added, and confirm returned values match HEAD for several geometries (compute HEAD values via git stash or git show HEAD:... to a /tmp file). Check the margin comparison uses the correct n_side (rib vs strip: side slab vs cladding index), warning fires for both TE and TM paths, and lru_cache interaction is as documented.' },
  { key: 'fdmode', files: 'src/siphon/fdmode.py, tests/test_fdmode.py', hints: 'The riskiest module. Verify the Stern semi-vectorial discretization: which interfaces get 1/n^2 vs n^2 averaging for TE (dominant Ex) vs TM; check eigenvalue -> neff conversion (neff = sqrt(lambda)/k0 or similar), Dirichlet boundary placement, guided-mode filtering bounds, and the mesh-snapping logic (core edges on cell boundaries). Independent check: 1D limit vs slab_neffs for BOTH TE and TM; a physically expected degeneracy or ordering; convergence direction. Also check field indexing/orientation claims ([j,i] at (x_i, y_j)) against a computed field.' },
]

phase('Refute')
log(`launching ${TARGETS.length} adversarial reviewers (read-only)`)
const results = await parallel(
  TARGETS.map(t => () =>
    agent(
      `${COMMON}\n\nTARGET MODULE: ${t.key}\nFILES IN SCOPE: ${t.files}\nSPECIFIC ATTACK SURFACE: ${t.hints}\n\nReturn structured findings only.`,
      { label: `refute:${t.key}`, phase: 'Refute', schema: SCHEMA }
    )
  )
)
const out = {}
TARGETS.forEach((t, i) => { out[t.key] = results[i] })
const nFindings = results.filter(Boolean).reduce((n, r) => n + r.findings.length, 0)
log(`review done: ${nFindings} findings across ${TARGETS.length} modules`)
return out