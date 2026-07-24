export const meta = {
  name: 'siphon-adversarial-review',
  description: 'Review SiPhon physics, numerics, API consistency and tests; adversarially verify findings',
  phases: [
    { title: 'Review', detail: '5 lenses in parallel' },
    { title: 'Verify', detail: '3 refuters per finding, majority vote' },
  ],
}

const FINDINGS = {
  type: 'object',
  properties: {
    findings: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' },
      file: { type: 'string' },
      line: { type: 'number' },
      severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
      claim: { type: 'string', description: 'the precise defect' },
      evidence: { type: 'string', description: 'formula/reference/computed counter-example' },
      suggested_fix: { type: 'string' },
    }, required: ['title', 'file', 'severity', 'claim', 'evidence', 'suggested_fix'] } },
  },
  required: ['findings'],
}

const VERDICT = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean', description: 'true if the finding is wrong, overstated, or not worth fixing' },
    reasoning: { type: 'string' },
  },
  required: ['refuted', 'reasoning'],
}

const BASE = `Repo: /Users/tr/prog/silicon-photonics — "SiPhon", a silicon photonics toolkit (Python 3.13, numpy/scipy, src/siphon/, tests/, examples/). Run code freely with: cd /Users/tr/prog/silicon-photonics && uv run python ... . The full test suite (uv run pytest -q) currently passes 144 tests. Report ONLY defects that matter: wrong physics, wrong math, numerical fragility a user would plausibly hit, silently misleading APIs, or tests that would pass with broken physics. Do NOT report style, docstring nits, hypothetical micro-optimizations, or missing features. For every claim, compute a concrete counter-example or cite the standard formula you checked against. Quality over quantity — an empty findings list is a valid result.`

phase('Review')
const LENSES = [
  { key: 'physics-passive', prompt: `${BASE}\nLENS: passive-optics physics. Audit src/siphon/materials.py, src/siphon/waveguide.py, src/siphon/components.py against the literature: Sellmeier coefficients and validity ranges; slab dispersion equations (TE/TM phase terms, asymmetric cases, mode counting); effective-index-method polarization bookkeeping; linearized-dispersion phase in Straight/rings/MZI; Bogaerts 2012 ring formulas including signs, conjugates, critical coupling, add-drop cross terms; unitarity/reciprocity of every lossless S-matrix (verify numerically at multiple wavelengths); FSR/Q/finesse helpers; group index and dispersion finite differences.` },
  { key: 'physics-link', prompt: `${BASE}\nLENS: link/system physics. Audit src/siphon/link.py and src/siphon/wdm.py: Q-from-BER for PAM-M (SER/BER Gray mapping, erfcinv usage — check numerically against known anchors: BER 1e-12 NRZ -> Q~7.03; KP4 2.4e-4), thermal-limited OMA sensitivity formula and the (M-1) PAM factor, noise bandwidth assumption, ER penalty and the avg-power->OMA conversion consistency (is ER accounted twice or correctly split between launch OMA and penalty?), RIN penalty formula, crosstalk penalty, energy-per-bit accounting (laser sharing, tuning power vs data rate), dBm/mW conversions, DWDM frequency-grid math, tuning-power expectation (FSR/4 argument), preset plausibility (launch power vs damage/nonlinearity, sensitivity vs published receivers).` },
  { key: 'numerics', prompt: `${BASE}\nLENS: numerical robustness. Audit src/siphon/circuit.py (S_ee + S_ei P (I-S_ii P)^-1 S_ie assembly: permutation construction, port bookkeeping order, conditioning check threshold, behavior at exact resonance of lossless loops), src/siphon/waveguide.py (brentq brackets: can f(lo) and f(hi) share a sign for real geometries and silently drop modes? lru_cache on frozen dataclass + float rounding to 9 decimals; finite-difference step sizes vs material-model range edges), src/siphon/montecarlo.py (truncated-normal resampling loop, NaN propagation, ddof choices), and any float-comparison hazards. Actually execute stress cases (extreme coupling values, zero-loss rings on resonance, very thick/thin slabs, wavelengths near model range edges).` },
  { key: 'consistency', prompt: `${BASE}\nLENS: cross-module consistency and honest APIs. Check: units are um/dB/dBm/GHz consistently and docstrings match code; S-matrix convention (b=Sa, port ordering, no-reflection claims) is identical across components and circuit; sign conventions exp(-i beta L) vs ring exp(+i phi) are documented and cannot silently combine wrongly when a ring model is used inside Circuit next to Straights (build such a circuit and check the spectrum is still physical); __init__ exports match modules; README/examples claims match actual behavior (run examples 01-04 and verify printed numbers match what the README/docs promise, e.g. '144 tests', margins, neff values); pyproject metadata sane.` },
  { key: 'test-adequacy', prompt: `${BASE}\nLENS: test adequacy. Read every file in tests/. Find tests that would still pass if the physics were wrong (tautologies, tolerances so loose they admit sign errors, snapshot-of-current-output tests), important untested behaviors (e.g. TM slab against an independent anchor, circuit solver against a 3+ component analytic case, PAM4 sensitivity anchor, ring critical coupling depth), and any test that encodes a wrong expectation. For each proposed gap, state the exact assertion that would catch a realistic bug. Mutation-test mentally: pick 5 plausible single-character physics mutations (sign flip, squared term dropped) and check whether the suite would catch each; report the ones it would miss as findings.` },
]

const reviewed = await pipeline(
  LENSES,
  l => agent(l.prompt, { label: `review:${l.key}`, phase: 'Review', schema: FINDINGS, effort: 'high' }),
  (rev, lens) => {
    if (!rev || !rev.findings.length) return []
    return parallel(rev.findings.map(f => () =>
      parallel([1, 2, 3].map(i =>
        () => agent(
          `${BASE}\nYou are refuter #${i}. A reviewer claims the following defect in SiPhon. Independently verify by reading the code and RUNNING it. Refute it if: the claim is factually wrong, the behavior is actually correct physics, the docstring already covers it as a documented limitation AND it would not mislead a competent user, or it is trivia not worth changing. Default to refuted=true if you cannot reproduce the problem concretely.\n\nFINDING:\n${JSON.stringify(f, null, 2)}`,
          { label: `verify:${(f.title || '').slice(0, 40)}`, phase: 'Verify', schema: VERDICT }
        )
      )).then(votes => {
        const v = votes.filter(Boolean)
        const upheld = v.filter(x => !x.refuted).length
        return { ...f, lens: lens.key, upheld, of: v.length, confirmed: upheld >= 2, refuter_notes: v.map(x => x.reasoning.slice(0, 300)) }
      })
    ))
  }
)

const all = reviewed.filter(Boolean).flat().filter(Boolean)
const confirmed = all.filter(f => f.confirmed)
log(`${all.length} raw findings, ${confirmed.length} confirmed after refutation panel`)
const order = { critical: 0, major: 1, minor: 2 }
confirmed.sort((a, b) => order[a.severity] - order[b.severity])
return { confirmed, rejected: all.filter(f => !f.confirmed).map(f => ({ title: f.title, severity: f.severity, upheld: f.upheld, of: f.of })) }