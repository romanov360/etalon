export const meta = {
  name: 'siphon-build-fanout',
  description: 'Parallel build of SiPhon component, circuit, link, WDM modules plus tests and examples',
  phases: [{ title: 'Build', detail: '6 agents, disjoint file ownership' }],
}

const RESULT = {
  type: 'object',
  properties: {
    files_written: { type: 'array', items: { type: 'string' } },
    api_summary: { type: 'string', description: 'public API you implemented: signatures + one-line docs' },
    tests_passing: { type: 'boolean' },
    notes: { type: 'string', description: 'known limitations, conventions chosen, anything the integrator must know' },
  },
  required: ['files_written', 'api_summary', 'tests_passing', 'notes'],
}

const SHARED = `You are building one module of "SiPhon", a silicon photonics design toolkit at /Users/tr/prog/silicon-photonics (Python 3.13, numpy+scipy only in library code, src layout: src/siphon/). FIRST read the existing core modules to match style and reuse them: src/siphon/constants.py, src/siphon/materials.py, src/siphon/waveguide.py, and pyproject.toml.

HARD RULES:
- You own ONLY the files listed in your task. Do not create or modify any other file (no __init__.py edits, no pyproject edits — the integrator wires those).
- Units: wavelengths and geometry in um; power in dBm/mW as named; loss in dB. Docstrings state units.
- S-parameter convention (where applicable): a model exposes attribute \`ports: tuple[str, ...]\` and method \`s_params(wl: np.ndarray) -> np.ndarray\` of shape (len(wl), n_ports, n_ports), complex, with b_i = sum_j S[k, i, j] a_j at wavelength wl[k]. Port order in the matrix == order of \`ports\`. Passive models must be reciprocal (S == S.T per wavelength) and lossless ones unitary.
- Write focused pytest tests in your assigned test file; every physics formula needs at least one test against an analytic/known value (energy conservation, unitarity, known resonance positions, closed-form limits).
- Run your tests with: cd /Users/tr/prog/silicon-photonics && uv run pytest <your test file> -q   (the package is installed editable; import as \`from siphon import ...\`). Iterate until green.
- Match the docstring style of the existing modules: physics references and validity caveats, no chatty comments.
- Your final message is consumed by an orchestrator: return structured output only.`

phase('Build')
const TASKS = [
  {
    key: 'components',
    prompt: `${SHARED}

TASK: implement src/siphon/components.py and tests/test_components.py.

Provide S-parameter models (all following the convention above), each a small class:
- Straight(length_um, neff0, ng, loss_db_per_cm=0.0, wl0_um=1.55): 2 ports ("in","out"). Linearized dispersion: neff(wl) = neff0 + (wl - wl0)*(neff0 - ng)/wl0 (derived from ng = neff - wl*dneff/dwl). Phase exp(-1j*2*pi*neff(wl)*L/wl); amplitude 10**(-loss_db_total/20). Include classmethod from_waveguide(wg: siphon.waveguide.Waveguide, length_um, loss_db_per_cm, wl0_um, mode="TE") that evaluates neff0/ng from the mode solver once at wl0.
- DirectionalCoupler(coupling=0.5, loss_db=0.0): 4 ports ("in0","in1","out0","out1"), wavelength-flat; through amplitude t=sqrt(1-coupling), cross 1j*sqrt(coupling), times loss amplitude. in0->out0 through, in0->out1 cross. No back-reflection.
- YBranch(excess_loss_db=0.0): 3 ports ("in","out0","out1"), 50/50 split; note in docstring that an ideal lossless 1x2 cannot be unitary as a 3-port (radiated power) so only test power conservation <= 1.
- PhaseShifter(phase_rad=0.0, loss_db=0.0): 2 ports. Also ThermalPhaseShifter(power_mw, p_pi_mw, loss_db=0.0) with phase = pi*power/p_pi.
- GratingCoupler(peak_il_db=4.0, bw_1db_nm=35.0, center_um=1.55): 2 ports, Gaussian power response: IL_db(wl) = peak + (wl-center)^2 * (1 dB point at bw/2 from center).
- RingAddDrop(circumference_um, neff0, ng, kappa1_power, kappa2_power, loss_db_per_cm, wl0_um): 4 ports ("in","through","add","drop"), analytic transfer functions per Bogaerts et al., Laser Photon. Rev. 6, 47 (2012): with self-coupling t_i=sqrt(1-kappa_i), round-trip amplitude a and phase phi(wl) using the same linearized neff as Straight: through = (t1 - conj(t2)*a*exp(1j*phi))/(1 - t1*t2*a*exp(1j*phi)) etc. Fill the full reciprocal 4x4 including add->drop paths; document that in<->add direct coupling is 0.
- RingAllPass(circumference_um, neff0, ng, kappa_power, loss_db_per_cm, wl0_um): 2 ports, through = (t - a*exp(1j*phi))/(1 - t*a*exp(1j*phi)).
- MZI(dl_um, neff0, ng, coupling_in=0.5, coupling_out=0.5, loss_db_per_cm=0.0, wl0_um=1.55): 4 ports, composed ANALYTICALLY (matrix product of DC * diag(arm phases) * DC), not via the circuit solver.
Helper functions: ring_fsr_um(wl_um, ng, circumference_um)=wl^2/(ng*L); loaded_q(wl_um, fwhm_um); finesse(fsr, fwhm).

Required tests (min): unitarity of lossless DC and MZI; MZI extinction at balanced/imbalanced known wavelengths and FSR check; ring resonance wavelengths at phi=2*pi*m; all-pass critical coupling (a == t) gives deep extinction; add-drop energy conservation when lossless; FSR formula agreement with spectral peak spacing; Straight phase matches hand-computed value; GratingCoupler 1-dB bandwidth.`,
  },
  {
    key: 'circuit',
    prompt: `${SHARED}

TASK: implement src/siphon/circuit.py and tests/test_circuit.py.

A netlist-level frequency-domain circuit solver over S-parameter models:
- class Circuit: methods add(name: str, model) -> None; connect(a: tuple[str,str], b: tuple[str,str]) -> None (each port connected at most once; error otherwise); expose(external_name: str, port: tuple[str,str]) -> None. Unconnected+unexposed ports at solve time -> error listing them.
- Circuit.s_params(wl: np.ndarray) -> np.ndarray shape (len(wl), n_ext, n_ext) with external port order = order of expose() calls; also .external_ports tuple.
- Algorithm: per wavelength assemble block-diagonal S of all instances; split ports into external (e) and internal (i, the connected pairs); P = permutation matrix swapping each connected pair (a_int = P b_int). Then S_ext = S_ee + S_ei @ P @ inv(I - S_ii @ P) @ S_ie. Use numpy.linalg.solve rather than explicit inverse. Vectorize over wavelength if convenient (loop is fine, keep it clear).
- Convenience: transmission(wl, "inport", "outport") -> complex array; transmission_db(...) -> float array (power dB).
- Detect singular (I - S_ii P) (e.g. lossless resonant loop exactly on resonance) and raise a clear error naming the wavelength index.

For tests you may import from siphon.components IF it exists; if it does not exist yet, define tiny local stub models in the test file implementing the ports/s_params protocol (a fixed 2-port phase/attenuation element and an ideal 4-port coupler) so your tests are self-contained. Required tests: (1) two cascaded phase elements == product; (2) MZI assembled from 2 couplers + 2 arms matches the analytic 2x2 expression computed in the test; (3) feedback loop: all-pass ring built from one 4-port coupler with cross-out connected to cross-in through a phase/loss element matches the closed-form all-pass response; (4) reciprocity of assembled S; (5) error paths (dangling port, double connect).`,
  },
  {
    key: 'link',
    prompt: `${SHARED}

TASK: implement src/siphon/link.py and tests/test_link.py.

An optical link-budget engine for datacom (the kind of math a co-packaged-optics system architect runs), all closed-form and documented:
- @dataclass Laser: power_dbm (per wavelength line, fiber/chip-coupled as documented field launch_reference: str), wpe (wall-plug efficiency 0..1), rin_db_hz (e.g. -145).
- @dataclass Modulator: insertion_loss_db, extinction_ratio_db, modulation_loss_db computed property: for OOK/PAM the average-power penalty; energy_fj_per_bit.
- @dataclass Photodiode: responsivity_a_per_w, dark_current_na=10.
- @dataclass Tia: input_noise_pa_per_sqrt_hz, bandwidth_ghz, energy_fj_per_bit.
- Signaling: @dataclass Signaling(rate_gbd, levels=4 (PAM4) or 2 (NRZ), target_ber=2.4e-4 pre-FEC KP4). Q factor from BER: for PAM-M, symbol error ~ 2(1-1/M) Q(q) -> per-target-BER q via scipy.special.erfcinv; document the standard approximation BER = SER/log2(M) with Gray coding.
- receiver_sensitivity_oma_dbm(pd, tia, sig): thermal-noise-limited closed form: required OMA_outer = 2*q*(M-1)*i_n_rms/R, i_n_rms = input_noise * sqrt(0.75 * baud GHz -> Hz noise bandwidth as 0.75*rate) ; document assumptions (no shot noise, no ISI; add rin_penalty separately).
- Penalty helpers (each returns dB): er_penalty_db(er_db) = 10*log10((10**(er/10)+1)/(10**(er/10)-1)); rin_penalty_db(rin_db_hz, rate_gbd, q, levels) using standard sigma_rin formula, raise ValueError if RIN alone makes the target unreachable; crosstalk_penalty_db(agg_crosstalk_db) = -10*log10(1 - 10**(xt/10)) style power penalty.
- @dataclass LossElement(name: str, loss_db: float).
- class LinkBudget(laser, modulator, path: list[LossElement], photodiode, tia, signaling, penalties_db: dict[str, float] | None): properties/methods: launched_oma_dbm (laser power - mod IL - modulation loss, ER-limited OMA), received_oma_dbm (minus sum of path losses), sensitivity_oma_dbm (incl. explicit penalties + rin + er penalties, documented composition), margin_db, report() -> str (aligned multi-line waterfall table: each contribution, running power, then margin; this is the flagship output, make it beautiful plain text).
- energy_per_bit_pj(link, n_lanes, laser_shared_by=1, thermal_tuning_mw_per_lane=0, serdes_pj_per_bit=...) -> dict breakdown {laser, modulator, tia, tuning, serdes, total} in pJ/bit; laser electrical = optical/wpe.
- Factory presets with 2026-realistic numbers and per-field justifying comments: preset_pluggable_dr4() (100G/lane PAM4, EML-ish, 3.5 dB GC losses etc.) and preset_cpo_optical_io() (external laser via fiber, lower per-lane rate, ring-based, much lower serdes energy). Numbers should be plausible, clearly marked illustrative.

Required tests: Q-from-BER round trip (BER 2.4e-4 -> q ~ 3.4-3.6; 1e-12 -> ~7.03); ER penalty at 3 dB ER ~ 4.77 dB, at inf ER -> 0; monotonicity (more path loss -> less margin, 1 dB in = 1 dB out); sensitivity worsens with rate and with PAM4 vs NRZ by 20*log10(3) thermal-limited ~ 9.54 dB; energy breakdown sums; both presets produce finite positive-margin links and report() contains every element name.`,
  },
  {
    key: 'wdm',
    prompt: `${SHARED}

TASK: implement src/siphon/wdm.py and tests/test_wdm.py.

WDM system helpers for ring-based and CWDM links:
- ChannelPlan: dataclass with center wavelengths list; classmethods cwdm4() (1271/1291/1311/1331 nm), lr4(), dwdm(center_um, spacing_ghz, n_channels) computing wavelength grid from frequency grid via c=299792.458 um*GHz... (be careful with units: f_GHz = c_um_ghz/wl_um with c = 2.99792458e5 um*GHz? No: c = 2.99792458e8 m/s = 2.99792458e14 um/s; f in GHz = 2.99792458e5 / wl_um. Use that.). Methods: spacing_nm(), as_frequencies_thz().
- ring_bank_channel_count_limit(fsr_nm, channel_spacing_nm, guard_channels=1): max channels in one FSR.
- thermal_tuning: tuning_efficiency_nm_per_mw default 0.25 (typical Si microring heater, document); tuning_power_mw(detune_nm, efficiency) and expected_tuning_power_mw(fsr_nm, efficiency): mean detune = FSR/4 under uniform fabrication offset with wrap-around (document derivation: uniform over FSR, nearest resonance within FSR/2, mean FSR/4).
- resonance_shift_nm_per_k(wl_um, ng, dneff_dT=1.86e-4*0.85): dlambda/dT = wl * (dneff/dT)/ng, document confinement-factor scaling.
- aggregate_crosstalk_db(per_channel_isolation_db, n_aggressors): 10*log10(n * 10**(-iso/10)) as worst-case incoherent sum, returned as negative dB (isolation positive input).
- laser_grid_check(plan, ring_fsr_nm): warn/raise if channel span exceeds FSR.

Required tests: CWDM4 wavelengths exact; DWDM 100 GHz spacing at 1550 -> ~0.8 nm; frequency/wavelength round trip; tuning power at FSR/4 detune with 0.25 nm/mW; crosstalk aggregation (-30 dB iso, 3 aggressors -> ~ -25.2 dB); channel count limit arithmetic.`,
  },
  {
    key: 'core-tests',
    prompt: `${SHARED}

TASK: write tests/test_materials.py and tests/test_waveguide.py for the EXISTING core modules (do not modify library code; if you believe you found a real bug, document it in your notes output instead of fixing).

Coverage targets:
- materials: n_si(1.55)≈3.4776, n_sio2(1.55)≈1.4440, n_sin(1.55)≈1.9963 (tolerance 1e-3); vectorized input returns array; out-of-range raises ValueError; unknown material KeyError; group_index_material('si',1.55)≈3.60 (tol 0.05); dn/dT constants exist and are positive.
- waveguide.slab_neffs: 220nm Si/SiO2 slab at 1.55 TE -> single mode ≈2.848 (tol 2e-3); TM0 lower than TE0; thick slab (1.5um) multimode with strictly decreasing neffs all in (1.444, 3.4776); no-mode cases return [] (10nm slab TM; core index below cladding); asymmetric slab (air top) has fewer/shifted modes than symmetric.
- Waveguide (EIM): 500x220 strip TE neff(1.55)≈2.49 (tol 0.02, EIM value), TM ≈1.85 (tol 0.03); group_index ≈4.0 (tol 0.15); neff decreases with narrower width; rib (slab_um=0.09) neff > strip neff (same core); wl sweep monotonic decreasing neff over 1.5-1.6; invalid geometry raises; 100x100nm guide raises no-mode ValueError... verify first with a quick uv run python check which tiny geometry actually fails, use one that does.
- bend_loss_db_per_90deg: positive, decreasing in radius, ≈0.086 dB at 1um within factor 2; nonpositive radius raises.
Where I wrote ≈ values above, VERIFY the actual current outputs first by running python, then set test tolerances tight around correct physics (do not just snapshot wrong values — sanity-check against the physics expectations given).`,
  },
  {
    key: 'examples',
    prompt: `${SHARED}

TASK: write three runnable example scripts. They are the product demo, so clear narrative printing matters (aligned tables, section headers). No matplotlib requirement: try-import matplotlib and save PNGs to examples/out/ if available, else skip plotting silently. The components/circuit/link/wdm modules are being written CONCURRENTLY by other agents to the exact contracts below — code against these contracts, and where a call might drift, keep usage minimal and mainstream:
- siphon.components: Straight(length_um, neff0, ng, loss_db_per_cm=0, wl0_um=1.55) ports ("in","out"); DirectionalCoupler(coupling, loss_db=0) ports ("in0","in1","out0","out1"); RingAddDrop(circumference_um, neff0, ng, kappa1_power, kappa2_power, loss_db_per_cm, wl0_um) ports ("in","through","add","drop"); MZI(dl_um, neff0, ng, ...) 4 ports; all expose s_params(wl)->(n_wl,n,n) and .ports.
- siphon.circuit: Circuit() with .add(name, model), .connect((n1,p1),(n2,p2)), .expose(ext_name,(n,p)), .transmission_db(wl, ext_in, ext_out).
- siphon.link: preset_pluggable_dr4() and preset_cpo_optical_io() -> LinkBudget with .margin_db, .report(); energy_per_bit_pj(link, n_lanes=8, ...) -> dict.
- siphon.wdm: ChannelPlan.cwdm4(), tuning_power_mw / expected_tuning_power_mw, aggregate_crosstalk_db.
- siphon.waveguide.Waveguide / siphon.materials exist already — read them.

Files (only these + optionally examples/out/.gitkeep):
1. examples/01_waveguide_explorer.py — sweep strip width 0.35-0.7um at 220nm: neff/ng tables TE/TM, single-mode boundary discussion, bend-loss table; pure core modules, must run TODAY (verify by running it).
2. examples/02_ring_wdm_filter.py — design a 4-channel ring-bank demux: pick circumference from target FSR via ng, print FSR/Q/crosstalk table using RingAddDrop spectra, thermal tuning power via wdm helpers.
3. examples/03_cpo_vs_pluggable.py — the flagship: both link presets, print both waterfall report()s, margin comparison, energy_per_bit_pj breakdown table and $/W framing per 1.6T switch (51.2T -> 64x800G o.e.).
Scripts 2-3 will only run after integration; make them robust (wrap the import of not-yet-existing modules with a clear "module not built yet" message ONLY at the __main__ entry, structure code so the orchestrator can fix minor API drift easily). Return in notes which scripts you verified running.`,
  },
]

const results = await parallel(TASKS.map(t => () =>
  agent(t.prompt, { label: `build:${t.key}`, phase: 'Build', schema: RESULT })
))
return results.map((r, i) => ({ task: TASKS[i].key, ...(r || { failed: true }) }))