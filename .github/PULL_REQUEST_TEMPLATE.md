## What this does

<!-- One or two sentences. If it's a new module, what physical effect/capability does it add? -->

## Checklist

- [ ] `uv run pytest` passes locally
- [ ] Ran any `examples/*.py` this change touches (or added a new one, if this is a new capability worth demonstrating)
- [ ] New physics has at least one analytic-anchor test (a known closed-form case the code must reproduce), not just "doesn't crash"
- [ ] New/changed modules state their scope and honesty limits in the docstring (see `CONTRIBUTING.md`)
- [ ] I tried to break my own change before opening this PR (edge cases, degenerate inputs, an independently-computable check) — see `CONTRIBUTING.md`'s "What 'done' means here"

## Numeric verification (for physics changes)

<!--
If this changes or adds a physical model: show the check you ran to convince
yourself it's right — a hand-derived special case, a comparison against a
published value, an independent re-implementation of the key formula. This
is the single most useful thing a reviewer can look at.
-->
