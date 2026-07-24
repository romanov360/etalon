# Research provenance

Everything in this directory is the recorded output of the multi-agent research and
build runs executed on 2026-07-23 for the silicon-photonics startup investigation.

## Top level

- `research_result.json` — full parsed result of the deep-research workflow
  (`wf_7e2013eb-910`): executive landscape summary, 9 research dimensions
  (facts / players / opportunities / sources), 6 startup theses, and every
  judge verdict with scores, fatal flaws, and improvement suggestions.
- `theses_full.md` — human-readable dump of all 6 ranked theses with their
  complete judge panel verdicts.
- `extract_theses.py` — the script used to extract/format the above from the
  raw workflow output.

## `raw/`

- `workflow-deep-research/` — complete transcripts of the deep-research
  workflow (28 agents): `journal.jsonl` records each agent's structured return
  value; `agent-*.jsonl` are the full per-agent transcripts (every tool call,
  web search, and fetched page); `agent-*.meta.json` are per-agent metadata.
- `workflow-build-fanout/` — same layout for the SiPhon build workflow
  (`wf_135a4353-69a`, 6 build agents).
- `workflow-adversarial-review/` — same layout for the post-build adversarial
  review workflow (`wf_485198c6-740`: 5 review lenses, findings verified by
  3-refuter panels).
- `scripts/` — the exact workflow orchestration scripts that were executed.
- `task-outputs/` — raw background-task output files as delivered by the
  harness (the deep-research result `wkpde1vcl.output` is the authoritative
  full copy of what the workflow returned).

The polished write-ups derived from this material live one level up:
`docs/RESEARCH.md` (industry report) and `docs/THESIS.md` (startup thesis).
