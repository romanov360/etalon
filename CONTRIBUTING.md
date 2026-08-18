# Contributing to Etalon

Etalon is a young project with one main author, but built to be usable and
extendable by others. This file is what you need to send a good PR.

## Setup

```bash
git clone https://github.com/romanov360/etalon.git && cd etalon
uv sync
uv run pytest          # should show 434+ passing, ~50s
```

No API keys, no external services, no network access needed to build or test.

## Before you open a PR

- **Run the full test suite.** `uv run pytest -v`. CI runs it too, but don't
  rely on CI to find things a local run would have caught faster.
- **Check coverage for new code.** `uv run pytest --cov=etalon --cov-report=term-missing`
  shows uncovered lines per file. New physics should be covered by an
  analytic-anchor test, not just a "doesn't crash" smoke test — see the
  point above.
- **Run the examples that touch your change.** `uv run python examples/NN_*.py`
  — the test suite covers unit-level correctness; the examples are the
  integration smoke test and have caught real bugs unit tests missed (see
  `CHANGELOG.md`).
- **Add tests, not just implementation.** Every module in this repo has
  analytic anchors where possible (a known closed-form result the code must
  reproduce exactly) rather than only "does it run without crashing" checks.
  If you're adding a new physics result, ask: what's the special case with a
  hand-derivable answer?

## Code style

- **No comments explaining *what* the code does** — names and structure
  should make that obvious. Comments exist to record *why*: a non-obvious
  physical constraint, a convention that isn't self-evident from the code, a
  workaround for a specific numerical issue. If you'd delete a comment and
  nothing would be lost, delete it.
- **Every module has a "Scope" and/or "Honesty limits" section** in its
  docstring, stating what the model does and does NOT claim to capture (see
  `src/etalon/isi.py` or `src/etalon/equalize.py` for the pattern). New
  modules should have one too. This project's credibility rests on being
  explicit about where the physics is exact, where it's a reduced-order
  approximation, and where it's architecture-level budgeting rather than
  signoff-grade — say which, every time.
- **Units belong in variable/parameter names**, not just docstrings
  (`width_um`, `rate_gbd`, `power_dbm`). This has already prevented at least
  one class of bug (unit-convention mismatches) that would otherwise be easy
  to introduce silently.
- **Validate inputs at the boundary, raise `ValueError` with enough context
  to fix it.** No bare exceptions — every `raise` should tell the caller what
  was wrong and, where it's not obvious, what to do about it. See any
  existing module for the tone (e.g. `src/etalon/thermal.py`'s
  `solve_coupled_powers`).

## What "done" means here

Every module in this repo went through at least one adversarial review before
being merged — a second pass by a reviewer who tries to break the
implementation rather than just reads it and nods along, re-deriving the
physics from first principles rather than trusting the code's own comments.
See `CHANGELOG.md` for what that process has actually caught: sign errors,
silent-garbage failure modes, hidden double-counting, and at least one case
where the author's own docstring made a false claim ("always ≥ 0") that a
fresh numeric check disproved.

You don't need to run a formal multi-round review to contribute — that's a
project-maintainer step for now — but do hold your own PR to the same bar
before opening it: **actually run the numbers**, don't just read the math and
trust it looks right. If you're adding a new physical model, the most useful
thing you can do is try to break it yourself first (edge cases, degenerate
inputs, a case where you can independently compute the right answer) before
asking someone else to.

## Reporting a bug

Open an issue with:
- The smallest code snippet that reproduces it.
- What you expected vs. what you got (numbers, not just "wrong").
- If it's a physics question (not a code bug) — e.g. "should this really be
  computed this way?" — say so explicitly; those get triaged differently
  from implementation bugs.

## Scope

See the README's "Scope and honesty" section and `docs/THESIS.md` for what
this project is and isn't trying to be. In short: closed-form / semi-analytic
physics, architecture-level budgeting accuracy, zero heavy dependencies
(numpy/scipy only). PRs that would pull in a large new dependency (a GPU
framework, a full FDTD solver, etc.) or that aim at signoff-grade accuracy
are probably out of scope — open an issue to discuss before investing time in
one.
