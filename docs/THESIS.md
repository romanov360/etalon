# Startup Thesis: Where to Build in Silicon Photonics (July 2026)

*Derived from a 28-agent research + adversarial-judging run. Full evidence base in
[RESEARCH.md](RESEARCH.md); all six theses with complete judge verdicts in
[research/theses_full.md](research/theses_full.md).*

---

## The scoreboard

Six theses were generated from the research and each was attacked by a three-judge
adversarial panel (market skeptic, competitive skeptic, execution skeptic). Scores are
averages; 7+ meant "an investor would take the meeting."

| Rank | Thesis | Score | One-liner | Why it didn't score higher |
|---|---|---|---|---|
| 1 | **Interferon** — CPO digital twin + test/yield data layer | 4.7 | Software that predicts, tests, and certifies co-packaged optics | Keysight closed the VPIphotonics deal (June 2026) and now ships E-O-E link simulation in ADS; cross-customer data pooling is contractually hard; ~20 buyers on the twin side |
| 2 | **Aperture** — neutral, AI-native photonic EDA | 4.3 | The independent cloud/agentic design platform, open-core on gdsfactory | TAM honestly small (~$0.4–1.3B); gdsfactory's creators (DoPlayDo) already commercialize it; Cadence already partnered with Flexcompute |
| 3 | **TrueNorth Photonics** — defense PNT, merchant model | 4.3 | Fabless silicon-photonic gyro/IMU engines; second source to Anello | Real demand (GPS jamming, $20M APFIT award to Anello) but hardware timelines and defense sales cycles vs. a small team |
| 4 | **Helios Fabric** — OCS control plane for the non-Google world | 4.0 | Google's Apollo topology/scheduling stack, productized for merchant OCS buyers | Neoclouds are early; OCS vendors may bundle; Google-grade control-plane talent problem |
| 5 | **Kelvin Light** — athermal multi-wavelength external laser | 4.0 | Kill the TEC: uncooled ELSFP-socket laser module (40–60% ELS power saving) | The right problem (lasers = 60% of link power, 90% of failures) but $15–20M before a prototype; not startable by a software team |
| 6 | **OpenBeam** — merchant OCI-MSA optical engine, non-TSMC | 3.7 | The multi-source UCIe-fronted optical chiplet on GF/Tower | $150–300M to production against Ayar ($3.75B valuation), Marvell/Celestial, NVIDIA in-house — capital-closed lane |

No thesis cleared the meeting bar as written. That is the honest read of deep-tech
venture odds in a space where $15B+ of M&A just consolidated every obvious lane.
But the judges' *improvement* notes converge on a specific, fundable synthesis.

## What the research actually says

Five findings carry the decision (numbers and sources in RESEARCH.md):

1. **One demand engine.** AI-datacenter interconnect dominates: silicon-photonics
   modules $4.2B (2024) → $24.8B (2030); 2026 is the year of 1.6T; CPO is crossing from
   demo to deployment (Broadcom Davisson shipping, NVIDIA Quantum-X on TSMC COUPE) but is
   still ~0.5% of AI-DC optics, heading to ~30–35% of high-speed ports by 2027–2030.
2. **The chip lanes are capital-closed.** Optical I/O chiplets need $500M+ (Ayar,
   Lightmatter); photonic *computing* is strategically dead (Lightmatter pivoted to
   interconnect, Luminous died); consumer biosensing is a graveyard (Rockley, >$500M).
3. **The unglamorous middle is unclaimed.** Lasers (~60% of link power, ~90% of link
   failures per Meta), packaging/fiber attach (50–80% of module cost), **test/known-good-die
   (>100 s per PIC, no standard, one bad engine among 24+ scraps a CPO assembly)**, ring
   thermal tuning, and yield variability — TSMC itself names test, fiber array, and
   packaging as the CPO blockers.
4. **The software layer just became a vacuum.** In ~12 months every independent Western
   photonic design-software vendor was absorbed: Lumerical→Synopsys, RSoft+VPIphotonics→
   Keysight, Luceda→Semitronix (China), Quantifi→Teradyne. No neutral, cloud-native vendor
   remains, while 110+ US photonics companies, four foundry PDK ecosystems, and hyperscaler
   CPO teams all need modern tooling.
5. **Exits are fast and priced.** Teramount $430M (10 months after a $50M round), Nubis
   $270M, DustPhotonics ~$1.3B, Celestial AI $3.25–5.5B. Acquirers demonstrably pay for
   production-ready CPO-adjacent capability; time-to-exit can be <4 years.

## The recommended play

**Build the open, statistically-aware E-O-E link engine now; sell qualification
compression, not seat licenses; earn the test/yield data layer from inside one design
partner's flow.** This merges Interferon's problem (the loudest bottleneck) with the
judges' surviving corrections:

- **Product 0 (this repo).** An open-source photonic circuit + link toolkit — waveguide
  physics, S-matrix circuit solver, modulator/receiver/link-budget models, WDM/thermal-tuning
  math, Monte Carlo corner analysis. This is `etalon/`. It is the credibility artifact the
  top thesis prescribes ("that artifact alone opens doors to CPO architecture teams") and
  the distribution wedge the #2 thesis prescribes (open-core, community-standard). It is
  deliberately *not* the revenue product.
- **Revenue motion 1 (months 0–12): paid qualification compression.** Meta disclosed a
  90-million-hour CPO reliability validation; that budget provably exists today. Sell
  $200K–$3M engagements to CPO challengers and OSATs — link-margin/corner modeling, test-time
  compression, wafer-to-package correlation — with contractual rights to the resulting
  anonymized model improvements written into the first contract.
- **Product 1 (months 6–24): the test/yield analytics layer** (Interferon "Side B" only,
  per all three judges): known-good-optical-engine binning analytics on top of probe
  hardware, priced per wafer against measurable scrap economics (one bad engine among 24+
  kills the assembly). Partner with a probe vendor that *lacks* an analytics story
  (FormFactor or ficonTEC), not Teradyne, which already bought its own.
- **Deliberate non-goals:** no chiplet, no laser, no head-on "digital twin" SKU against
  Keysight/ADS, no fork-against-maintainers of gdsfactory (interoperate instead).
- **Design partner precondition.** Do not raise until one design partner — most plausibly a
  non-Broadcom CPO challenger (Marvell/Celestial ecosystem, Ayar-adjacent OSAT, or a
  Taiwanese test house) — signs with explicit data-access and model-ownership terms. The
  judges were unanimous: without this the models are toys.
- **Capitalization honesty.** $5–8M total, 8–12 people, milestones aimed at either (a) real
  per-wafer ARR when CPO volume arrives 2028–2029, or (b) a $150–400M strategic sale to
  Advantest/FormFactor/Keysight — whoever loses the analytics race. Skip the $25–40M
  Series A the original thesis wanted; the judges showed the math doesn't support it.

**Backup wedge** (kept warm, shares 80% of the codebase): the ITAR/neutrality angle from
Aperture — post-Luceda/Semitronix, defense primes and US-sensitive teams have a
disqualified vendor and real pricing power. The same engine + verification layer sells
there at $150–500K ACVs through SBIR/prime-subcontract channels.

## Why this can win anyway

- The pain is verified from three independent directions (TrendForce, TSMC, Meta).
- The incumbents' weakness is structural, not featural: Keysight/Teradyne own probe and
  simulation *hardware-anchored* franchises and customers do not want their test-hardware
  vendor owning their yield data; Semitronix is geopolitically excluded from US floors.
- The open engine compounds: every published reproduction (NVIDIA's 3.5× efficiency claim,
  Meta's laser-failure statistics, OIF 3.2T budgets) is marketing an incumbent can't match
  without open-sourcing its crown jewels.
- Timing symmetry: the 2026–2027 first-deployment failure data is about to exist, and the
  entity that structures it first owns the category's vocabulary (and the eventual
  OIF/OCP known-good-die standard — fund one engineer to co-chair it).

## Honest expected value

You asked for $1T. No credible plan gets a new photonics company there — the entire
silicon-photonics module market is forecast at ~$25B in 2030, and the judged expected
outcome of the best thesis here is a $150–400M strategic exit on $5–8M raised (a 20–80×
return on capital), with a low-probability tail into a $1B+ standalone if the per-wafer
analytics attach to CPO volume at scale. The $1T-scale prize in this decade belongs to
whoever owns the AI-datacenter compute+network stack; photonics is a load-bearing
component of it, and the play above is positioned exactly where that stack is currently
weakest. That is the truthful version of "as big as it gets" from a two-person start.

## 18-month execution plan

| When | Milestone |
|---|---|
| Week 0 | Etalon core public: waveguide/circuit/link/WDM engine + Monte Carlo corners, CPO-vs-pluggable flagship example (this repo) |
| Month 1–2 | Published reproductions of public CPO data (NVIDIA power claims, Meta laser stats, OIF ELSFP budgets); post where the 59 VC-backed photonics startups' designers live |
| Month 2–4 | 3 paid pilot conversations from inbound; pick the design partner lane (CPO challenger vs. Taiwanese test house); OIF/OCP KGD working-group participation |
| Month 4–9 | First $200K+ qualification-compression engagement with data-rights language; hire 2 (photonic test + E-O-E co-sim — target the VPIphotonics diaspora, a 6–12 month window) |
| Month 9–18 | Per-wafer binning-analytics deployment on partner's floor; seed round ($5–8M) only after the partner signs; backup: ITAR/defense verification revenue via SBIR |

---

*Everything above is grounded in the July 2026 research run recorded under
[docs/research/](research/); judge verdicts that killed or reshaped each claim are in
[research/theses_full.md](research/theses_full.md).*
