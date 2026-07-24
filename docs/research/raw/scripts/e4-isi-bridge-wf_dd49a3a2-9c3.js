export const meta = {
  name: 'e4-isi-bridge',
  description: 'Build E4: circuit-to-link ISI bridge (siphon.isi) with time-domain eye reference, then adversarial review',
  phases: [
    { title: 'Build', detail: 'one agent builds siphon.isi + tests + example 07' },
    { title: 'Refute', detail: '3 adversarial reviewers, read-only' },
  ],
}

const HOUSE = `SiPhon toolkit at /Users/tr/prog/silicon-photonics (Python 3.13, uv, numpy/scipy ONLY). Read 2-3 existing modules first for style: frozen dataclasses, physics-stating docstrings with formulas/units/validity, ValueError on bad input, honest "architecture-level, not signoff" tone. Wavelengths um; S21 field convention exp(-1j*2*pi*neff*L/wl); f[Hz] = 2.99792458e14 / wl[um]. Verify with uv run pytest <your tests> -q; run the full suite and report (not fix) failures you did not cause.`

const BUILD_SCHEMA = {
  type: 'object',
  required: ['status', 'files', 'tests_passed', 'n_tests', 'api', 'notes'],
  properties: {
    status: { enum: ['ok', 'partial', 'failed'] },
    files: { type: 'array', items: { type: 'string' } },
    tests_passed: { type: 'boolean' },
    n_tests: { type: 'integer' },
    api: { type: 'string' },
    notes: { type: 'string' },
  },
}

const REVIEW_SCHEMA = {
  type: 'object',
  required: ['findings', 'checks_run'],
  properties: {
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
          evidence: { type: 'string' },
          suggested_fix: { type: 'string' },
        },
      },
    },
    checks_run: { type: 'string' },
  },
}

const BUILD_PROMPT = `${HOUSE}

TASK: E4 — the E-O-E bridge: compute a PAM eye-closure (ISI) penalty from a composed optical filter response S21(lambda), so a demux passband, grating-coupler ripple, or filter cascade produces a COMPUTED entry for LinkBudget.penalties_db instead of a hand-entered placeholder.

YOUR FILES (only these): src/siphon/isi.py (new), tests/test_isi.py (new), examples/07_filter_isi.py (new). Do NOT touch __init__.py, README, or any other file — integration is central.

SCOPE DISCIPLINE (this is the module's contract; put it in the docstring):
- PASSIVE LINEAR FIELD FILTERING ONLY, downstream of the modulator. The circuit solver's S(lambda) is CW steady-state, so a DRIVEN ring modulator is out of scope (time-varying cavity; its modulation response is not its static linewidth) — say so explicitly.
- Chirp-free intensity modulation: E_in(t) = sqrt(P(t)), P(t) the ideal rectangular PAM-M waveform (levels equally spaced between P0 and P1, outer ER = P1/P0). No TX rise time, no RX equalization: with an FFE-based receiver treat the result as an upper bound. Budgeting-grade eye closure, NOT TDECQ compliance.

METHOD (exact; the time-domain engine IS the product — exhaustive, deterministic, no closed-form approximation):
1. Baseband mapping: carrier fc = C/center_wl_um, C = 2.99792458e14 um*Hz. The periodic simulation has N = M^memory_symbols symbols x samples_per_symbol samples, duration N*T (T = 1/rate_gbd ns): FFT bin k maps to optical frequency fc + f_k (numpy fftfreq layout), wavelength C/(fc+f_k). Interpolate the user-supplied complex s21(wl_um) onto those wavelengths (linear on real & imag parts; wl_um must be monotonic). If the needed band [min wl, max wl] exceeds the supplied grid, raise ValueError STATING the required span in nm — do not extrapolate.
2. Pattern: a de Bruijn sequence B(M, memory_symbols) — every possible symbol subsequence of length memory_symbols appears exactly once, and the sequence is CYCLIC, so circular FFT filtering is exact (no edge transients) and the eye search is exhaustive over all patterns the filter memory can see. Implement the standard de Bruijn construction (Lyndon-word concatenation or the recursive db() algorithm, ~15 lines, no new deps). Cap M^memory_symbols <= 65536 (ValueError above). Validate the sequence property in tests.
3. Propagate the FIELD: e = ifft(fft(sqrt(P_waveform)) * H_baseband); detect I(t) = |e|^2 (square-law, exact within the chirp-free model — do NOT filter the intensity, cross-terms matter).
4. Normalize out flat loss: divide I(t) by |H(fc)|^2 and report insertion_loss_db = -20*log10(|H(fc)|) SEPARATELY, with a docstring warning that this IL is already in the link path budget (couplers/filters) — adding both the penalty AND the IL from here would double-count; the penalty is eye closure BEYOND flat loss.
5. Eye: for each of the samples_per_symbol sampling phases, take one sample per symbol, group by transmitted level, sub-eye openings = min(samples of level j+1) - max(samples of level j); worst sub-eye = min over j. Pick the sampling phase maximizing the worst sub-eye (clock recovery absorbs group delay). penalty_db = 10*log10(ideal_spacing / worst_opening), ideal_spacing = (P1-P0)/(M-1) under the same normalization. If worst_opening <= 0 return penalty_db = math.inf (eye closed), never a complex/NaN.

API:
- IsiResult frozen dataclass: penalty_db, insertion_loss_db, sampling_phase_ui (float in [0,1)), eye_openings (tuple, per sub-eye at the chosen phase, normalized), levels, rate_gbd, memory_symbols; plus a compact report() string.
- filter_isi_penalty_db(wl_um, s21, center_wl_um, rate_gbd, levels=4, extinction_ratio_db=math.inf, samples_per_symbol=16, memory_symbols=6) -> IsiResult. Validate: monotonic wl, matching shapes, center inside the grid, levels power of two >= 2 (reuse the convention of link._check_levels but implement locally or import), ER > 0 dB, sps >= 4, memory_symbols >= 2.
- Docstring shows the intended flow: t = circuit.transmission(wl, 'in', 'drop'); r = filter_isi_penalty_db(wl, t, ...); budget.penalties_db['demux_isi'] = r.penalty_db.

VALIDATION GATES (all as tests, physics anchored):
(1) flat filter (constant complex s21, e.g. 0.7*exp(1j*0.3)): penalty == 0 to 1e-9 dB, IL == -20log10(0.7) to 1e-9.
(2) pure delay exp(-2j*pi*f*tau), tau = 0.3 UI: penalty ~ 0 (< 1e-6 dB) — the sampling-phase search absorbs it.
(3) amplitude scale invariance: s21 -> 0.1*s21 changes IL by exactly +20 dB and penalty by < 1e-9 dB.
(4) single-pole low-pass H = 1/(1 + 1j*f/f3dB) on NRZ: penalty strictly increases as f3dB drops through [2, 1, 0.7, 0.5] x baud, and -> 0 as f3dB -> 50x baud.
(5) de Bruijn property: all M^memory cyclic windows of length memory are distinct (test for M=2,memory=8 and M=4,memory=5).
(6) exhaustiveness beats random: a seeded random 8192-symbol simulation through the same filter never shows a WORSE eye than the de Bruijn worst case (penalty_random <= penalty_debruijn + 1e-9).
(7) memory adequacy: for a ring add-drop drop port (components.RingAddDrop, e.g. kappa 0.05/0.05, 60 um) at 32 GBd, memory_symbols=6 vs 7 agree within 0.15 dB; and penalty(32 GBd) > penalty(8 GBd) >= 0.
(8) finite ER (4 dB) with a flat filter: still 0 penalty; with the low-pass: penalty finite and >= the inf-ER penalty is NOT required (do not assert a sign you have not verified numerically — check first, assert what holds).
(9) insufficient wl span raises ValueError mentioning the required span.

examples/07_filter_isi.py: build the 4-channel ring demux from example 02's style (or a single add-drop), sweep baud [8, 16, 32, 53.125] GBd NRZ through the drop port, print a table (baud, IL dB, ISI penalty dB), then rebuild preset_cpo_optical_io()'s waterfall with penalties_db['demux_isi'] set from the computed value and print the margin delta. Matplotlib optional exactly like examples 03/06 (skip gracefully). Must run: uv run python examples/07_filter_isi.py.

Return structured output only.`

phase('Build')
log('building siphon.isi (E4 bridge)')
const build = await agent(BUILD_PROMPT, { label: 'build:isi', phase: 'Build', schema: BUILD_SCHEMA })
if (!build || build.status === 'failed') {
  log('build failed; skipping review')
  return { build }
}

const LENSES = [
  { key: 'physics', hint: 'Re-derive the physics: field vs intensity filtering (cross-terms), sqrt(P) modulation, |H(fc)|^2 normalization and the IL double-count trap, baseband frequency mapping sign/layout vs numpy fftfreq and the S21 phase convention exp(-1j*2*pi*neff*L/wl) (does a positive group delay map to the correct baseband delay sign? verify numerically with a Straight of known length), eye construction, penalty definition. Build an independent brute-force check in /tmp (e.g. direct time-domain convolution, or an analytically known filter) and compare numbers.' },
  { key: 'numerics', hint: 'Attack the numerics: de Bruijn construction correctness (cyclic exhaustiveness), interpolation onto FFT bins (off-by-one, fftfreq ordering, nyquist bin), circular vs linear convolution assumptions, sampling-phase grid resolution, worst_opening <= 0 handling, memory_symbols truncation for high-Q filters (does gate 7 tolerance hide a real error for kappa=0.01 rings? test one), float precision of gates.' },
  { key: 'api-honesty', hint: 'Attack the API and the claims: does the docstring scope (no modulator dynamics, chirp-free, no equalization, budgeting-not-TDECQ) match what the code does? Is the IL/penalty split unambiguous and is the double-count warning correct against how preset paths actually charge coupler/filter loss? Is example 07 arithmetic right (recompute its margin delta by hand)? Do the tests pin behavior or just run code? Any input that silently returns garbage instead of raising?' },
]

phase('Refute')
log('3 adversarial reviewers on siphon.isi')
const reviews = await parallel(
  LENSES.map(l => () =>
    agent(
      `You are an adversarial reviewer of newly added code in SiPhon at /Users/tr/prog/silicon-photonics. Target: src/siphon/isi.py, tests/test_isi.py, examples/07_filter_isi.py (read neighboring modules for conventions). You are STRICTLY READ-ONLY in the repo — scratch scripts under /tmp only; a tampering audit runs after you. A finding is reportable only with a numeric reproduction or an exact-line contradiction of a derivation; empty findings is a fine outcome. Severity: high = wrong numbers in plausible use, medium = edge-case wrong or misleading API, low = doc/consistency.\n\nYOUR LENS: ${l.hint}\n\nReturn structured findings only.`,
      { label: `refute:${l.key}`, phase: 'Refute', schema: REVIEW_SCHEMA }
    )
  )
)
return { build, reviews: { physics: reviews[0], numerics: reviews[1], 'api-honesty': reviews[2] } }