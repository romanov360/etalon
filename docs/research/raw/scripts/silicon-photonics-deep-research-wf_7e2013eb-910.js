export const meta = {
  name: 'silicon-photonics-deep-research',
  description: 'Deep research sweep of the silicon photonics industry, synthesize startup theses, adversarially judge them',
  phases: [
    { title: 'Sweep', detail: '9 parallel researchers, one per industry dimension' },
    { title: 'Synthesize', detail: 'merge all research into 5-7 startup theses' },
    { title: 'Judge', detail: '3-lens adversarial panel per thesis' },
  ],
}

const RESEARCH_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: '3-5 paragraph synthesis of this dimension' },
    key_facts: { type: 'array', items: { type: 'string' }, description: 'concrete facts with numbers/dates where possible' },
    players: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, what: { type: 'string' }, funding_or_scale: { type: 'string' } }, required: ['name', 'what'] } },
    opportunities: { type: 'array', items: { type: 'string' }, description: 'gaps a new startup could exploit' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'key_facts', 'players', 'opportunities', 'sources'],
}

const COMMON = `You are a researcher on silicon photonics (as of mid-2026). Use ToolSearch with query "select:WebSearch,WebFetch" to load web tools, then do AT LEAST 6-10 distinct web searches and fetch the most substantive pages. Prioritize 2024-2026 news, funding announcements, foundry/PDK docs, and technical papers. Be concrete: numbers, dates, dollar amounts, named products. Your final output is raw structured data via the StructuredOutput tool, not prose for a human.`

phase('Sweep')
const DIMENSIONS = [
  { key: 'market', prompt: `${COMMON}\nDimension: overall silicon photonics market. TAM/CAGR by segment (datacom transceivers, co-packaged optics, telecom, sensing, computing), what is actually driving growth (AI cluster scale-out/scale-up bandwidth), unit economics of pluggable optics vs CPO, key inflection points expected 2026-2030.` },
  { key: 'optical-io', prompt: `${COMMON}\nDimension: optical I/O and co-packaged optics for AI datacenters. Cover Ayar Labs, Celestial AI, Lightmatter (Passage), Broadcom Bailly/CPO, Nvidia Quantum-X/Spectrum-X photonics, Marvell, TeraPoP/OIF standards, UCIe+optics, linear pluggable optics (LPO) vs CPO debate. Who is winning, what remains unsolved (laser integration, serviceability, thermals), design-win status at hyperscalers.` },
  { key: 'computing', prompt: `${COMMON}\nDimension: photonic computing (analog optical AI accelerators, optical switching for ML fabrics). Lightmatter, Lightelligence, Q.ANT, Salience Labs, optical circuit switches (Google Apollo/Palomar), silicon photonic quantum computing (PsiQuantum, Xanadu). What actually works commercially vs hype, realistic timelines.` },
  { key: 'foundry', prompt: `${COMMON}\nDimension: silicon photonics foundry & manufacturing ecosystem. GlobalFoundries Fotonix, TSMC COUPE, Intel silicon photonics, imec, Tower PH18, AIM Photonics, LioniX, Ligentec (SiN), heterogeneous III-V integration (laser-on-Si approaches: Intel, OpenLight, Nexus/Quintessent quantum dot lasers). MPW access, costs, PDK maturity, what a fabless photonics startup's path to silicon looks like in 2026.` },
  { key: 'eda-tools', prompt: `${COMMON}\nDimension: photonic design automation / EDA tooling. Ansys Lumerical, Synopsys OptoCompiler, Cadence, Luceda IPKISS, VPIphotonics, and the open-source ecosystem: gdsfactory (and GDSFactory+ / gdsfactory.com company), Tidy3D/Flexcompute, MEEP, SAX, Femwell, Palace. Also AI/ML-driven inverse design. Where are the tooling gaps and pain points photonic designers complain about? How big is this market and who pays?` },
  { key: 'sensing', prompt: `${COMMON}\nDimension: silicon photonics beyond datacom: FMCW LiDAR (Aeva, Voyant, Scantinel), biosensing/diagnostics (Genalyte, photonic biosensor chips), optical gyroscopes (Anello), spectroscopy/wearables (Rockley legacy, Apple watch rumors), optical atomic clocks, microwave photonics, free-space optical comms (Taara). Which sensing verticals are real businesses vs perpetual science projects.` },
  { key: 'funding', prompt: `${COMMON}\nDimension: startup funding and exits in silicon photonics 2023-2026. Recent rounds (amounts, valuations, investors) for photonics startups; acquisitions (e.g. by Nvidia, Marvell, Cisco, Nokia/Infinera); which theses VCs are funding now; any photonics startups that died and why. What stage/check sizes look like for a new entrant in 2026.` },
  { key: 'bottlenecks', prompt: `${COMMON}\nDimension: hard technical bottlenecks in silicon photonics productization. Laser integration and reliability, fiber attach / packaging cost (often >50% of module cost), wafer-level and known-good-die testing of photonics, thermal tuning power of rings, yield/process variation and the need for design-for-manufacturing, photonic packaging OSATs (PHIX, ficonTEC, Tyndall). For each bottleneck: who is attacking it and how open the field is.` },
  { key: 'ai-infra-angle', prompt: `${COMMON}\nDimension: intersection of AI boom and photonics as a business opportunity for a SOFTWARE-first startup. Think: photonic EDA + AI (copilots for photonic design, inverse design as a service, PDK-aware generative design), digital twins of optical links, test/measurement automation software, link-level simulation for CPO system architects at hyperscalers. Evidence of demand, who would pay, existing attempts.` },
]

const sweep = await parallel(DIMENSIONS.map(d => () =>
  agent(d.prompt, { label: `research:${d.key}`, phase: 'Sweep', schema: RESEARCH_SCHEMA })
))
const research = sweep.map((r, i) => ({ key: DIMENSIONS[i].key, ...(r || {}) })).filter(r => r.summary)
log(`Sweep complete: ${research.length}/9 dimensions returned data`)

phase('Synthesize')
const THESES_SCHEMA = {
  type: 'object',
  properties: {
    landscape_summary: { type: 'string', description: '5-8 paragraph executive summary of the whole space' },
    theses: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' },
      one_liner: { type: 'string' },
      wedge: { type: 'string', description: 'the specific initial product and customer' },
      why_now: { type: 'string' },
      market: { type: 'string', description: 'TAM/SAM reasoning with numbers' },
      competition: { type: 'string' },
      moat: { type: 'string' },
      capital_intensity: { type: 'string', description: 'fabless/software vs deep-tech capex profile' },
      software_mvp: { type: 'string', description: 'what software could be built THIS WEEK to start' },
      risks: { type: 'array', items: { type: 'string' } },
    }, required: ['name', 'one_liner', 'wedge', 'why_now', 'market', 'competition', 'moat', 'capital_intensity', 'software_mvp', 'risks'] } },
  },
  required: ['landscape_summary', 'theses'],
}

const synthesis = await agent(
  `You are a deep-tech venture strategist. Below is structured research on the silicon photonics industry (mid-2026), one block per dimension. Synthesize it into (a) an executive landscape summary and (b) 5-7 DISTINCT startup theses spanning the risk spectrum — at least two must be software-first/low-capex (buildable by a small team starting with code, no fab required), and at least one should be a bold deep-tech play. Be brutally honest about competition and capital needs. Research:\n\n${JSON.stringify(research, null, 2).slice(0, 180000)}`,
  { label: 'synthesize-theses', phase: 'Synthesize', schema: THESES_SCHEMA, effort: 'high' }
)
if (!synthesis) throw new Error('synthesis failed')
log(`Synthesized ${synthesis.theses.length} startup theses`)

phase('Judge')
const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    score: { type: 'number', description: '1-10' },
    rationale: { type: 'string' },
    fatal_flaws: { type: 'array', items: { type: 'string' } },
    improvements: { type: 'array', items: { type: 'string' } },
  },
  required: ['score', 'rationale', 'fatal_flaws', 'improvements'],
}
const LENSES = [
  { key: 'market', prompt: 'MARKET skeptic: is the customer and budget real? Would anyone pay in the next 18 months? Attack TAM inflation.' },
  { key: 'moat', prompt: 'COMPETITIVE skeptic: why won\'t Synopsys/Ansys/Broadcom/an incumbent or a well-funded existing startup crush this? Attack the moat.' },
  { key: 'execution', prompt: 'EXECUTION skeptic: can a tiny team starting with software today actually reach a sellable product? Attack capital needs, hiring, sales cycle, technical risk.' },
]

const judged = await pipeline(
  synthesis.theses,
  (thesis) => parallel(LENSES.map(l => () =>
    agent(`You are an adversarial deep-tech investor. ${l.prompt}\nDefault to skepticism; score 1-10 where 7+ means you would actually take a meeting. Thesis:\n${JSON.stringify(thesis, null, 2)}`,
      { label: `judge:${l.key}:${thesis.name.slice(0, 30)}`, phase: 'Judge', schema: VERDICT_SCHEMA })
  )).then(votes => {
    const v = votes.filter(Boolean)
    const avg = v.length ? v.reduce((s, x) => s + x.score, 0) / v.length : 0
    return { thesis, avg_score: Math.round(avg * 10) / 10, verdicts: v }
  })
)

const ranked = judged.filter(Boolean).sort((a, b) => b.avg_score - a.avg_score)
log(`Judging complete. Top thesis: ${ranked[0]?.thesis?.name} (${ranked[0]?.avg_score}/10)`)

return {
  landscape_summary: synthesis.landscape_summary,
  ranked_theses: ranked,
  research_by_dimension: research,
}