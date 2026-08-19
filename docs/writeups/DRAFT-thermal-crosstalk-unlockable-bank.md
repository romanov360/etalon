# Your ring bank might be un-lockable, not just power-hungry

**Status: DRAFT. Not published anywhere. Written for review before any
posting** — see the note at the bottom.

---

If you design ring-resonator WDM banks for silicon photonics, you already
budget heater power for thermal tuning: fabrication puts each ring's
resonance somewhere random within a free spectral range, and you burn
milliwatts per ring to drag it onto its assigned channel. The standard
move to cut that budget is a barrel-shift channel assignment — instead of
"ring *i* serves channel *i*," you let the whole bank rotate its channel
labels so the assignment absorbs as much of the common, die-wide
fabrication offset as possible almost for free. It's a good trick and it
works.

It also has a blind spot that doesn't show up until you go looking for
it: **the assignment it finds can be one that cannot physically lock**, not
just one that costs more power than you'd like.

## The mechanism

A ring's heater doesn't just heat that ring. It heats the substrate
underneath it, and that heat reaches the ring's neighbors on the same
bus. This is a well-documented effect in the ring-resonator literature —
thermal crosstalk between adjacent tunable rings, with measured
characterization back to at least Padmaraju & Bergman's 2014 work and
published compensation schemes for programmable photonic circuits
continuing through recent (2024-era) papers. It's not exotic physics:
it's a substrate conducting heat, the same way any two closely-spaced
resistors on a PCB warm each other.

The barrel-shift optimizer doesn't know this exists. It picks the
rotation that minimizes total heater power *assuming every ring is
thermally isolated* — which is the right assumption for the problem it's
solving, and a bad one for the problem you actually have.

## What happens when you put the two together

Take an 8-ring bank on a realistic dense-WDM grid, minimize its heater
power with the barrel-shift assignment (this part is standard, and it
works — the optimizer saves real power over the naive assignment), then
ask a different question: given the pitch between rings and a plausible
thermal healing length for the substrate stack, can this bank actually
*reach* the resonances the assignment says it should?

At tight pitch and a bulk (non-undercut) substrate, the answer for some
rings is no. Not "no, it costs more than expected" — no, as in the linear
system that says "here is the heater power every ring needs" returns a
negative number for two of the eight rings. A resistive heater cannot
supply negative power. What that negative number is telling you: those
two rings sit next to heavily-heated neighbors, need only a small shift
of their own, and their neighbors' heat alone overshoots them past their
target. There is no heater setting that fixes it, because the ring can't
pull heat back out of the substrate.

The barrel-shift optimizer picked this exact assignment because, ignoring
thermal crosstalk, it was the cheapest one. It's a legitimate local
optimum of the wrong problem.

## How much margin do you actually have

Sweeping the substrate's thermal healing length (worse isolation = longer
healing length = heat reaches further) against the same assignment shows
this isn't a hard on/off cliff, and it isn't even monotonic. At the
shortest healing lengths (tight thermal isolation, e.g. undercut
trenches under the rings), the coupled solve tracks the isolated-power
estimate almost exactly — crosstalk barely matters yet. Push the healing
length out and, before anything breaks, the coupled total actually drops
*below* the isolated estimate: some rings' neighbors are heating in the
same direction they need, so they're effectively lending each other free
heat and the bank gets a little cheaper to lock, not more expensive.
Then, on the same geometry, past roughly 10-12 µm the assignment stops
being able to lock at all — the same neighbor-lending effect that was
briefly a subsidy overshoots the rings that needed only a small shift of
their own. Not "expensive." Not achievable. The failure doesn't arrive
as a rising cost curve you'd catch by watching the power number climb;
it arrives after the power number has been quietly improving.

The two calculations answer different questions and you need both. The
crosstalk-blind optimizer tells you the cheapest assignment *if* rings
don't talk to each other thermally. The coupled solve tells you whether
that assignment survives contact with the fact that they do — and, when
it doesn't, exactly which rings fail and by how much, which is the
information you need to fix the layout (more pitch, better isolation) or
revisit the assignment, not just a warning that something's wrong
somewhere.

## Why this is worth publishing as open code, not just a paper result

The physics here isn't new — ring thermal crosstalk is documented,
measured, and has published compensation schemes. What doesn't seem to
exist anywhere accessible is the *composition*: a channel-assignment
optimizer and a thermal-crosstalk solver that talk to each other, so you
can ask "does my power-optimal assignment actually work" as one
computation instead of two disconnected spreadsheets you have to
cross-reference by hand. That composition is what turns "thermal
crosstalk is a known effect" into "here's whether your specific 8-ring
bank at your specific pitch can lock," which is the question a system
architect actually has.

It's also a small, honest example of the gap this project is built
around: photonics has excellent device-level and full-wave tools, and
comparatively little open tooling at the *system* level — the layer
where "is this architecture even physically achievable" questions live.
This is one instance of that gap, worked all the way through with real
numbers, in code anyone can run and check.

---

*Draft prepared 2026-08-18, based on `examples/08_thermal_crosstalk.py`
in the [Etalon](https://github.com/romanov360/etalon) toolkit
(`etalon.thermal`, adversarially reviewed — see the repo's CHANGELOG for
what that review checked and found). This file is not linked from
anywhere, is not on any public platform, and should not be posted
without explicit review and go-ahead — draft only, per instructions.*
