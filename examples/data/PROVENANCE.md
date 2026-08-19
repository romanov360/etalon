# Vendored data provenance

## `SiEPIC_ebeam_dc_gap200nm_Lc10um.sparam`

**Source**: [SiEPIC/SiEPIC_EBeam_PDK](https://github.com/SiEPIC/SiEPIC_EBeam_PDK),
file
`klayout/EBeam/CML/EBeam/source_data/ebeam_dc_te1550/dc_gap=200nm_Lc=10um.sparam`,
commit
[`06b17bc`](https://github.com/SiEPIC/SiEPIC_EBeam_PDK/blob/06b17bcb702eed2507c307c6c9cd5b0f2e19f32d/klayout/EBeam/CML/EBeam/source_data/ebeam_dc_te1550/dc_gap%3D200nm_Lc%3D10um.sparam),
fetched 2026-08-18.

**License**: MIT (SiEPIC EBeam PDK's own `LICENSE.md`, Copyright (c) 2016-2020
Lukas Chrostowski and contributors) — permissive, compatible with this
repo's Apache-2.0.

**What it is**: a 4-port, 101-point full S-matrix (Lumerical `.sparam`
lookup-table format: frequency in Hz, linear magnitude, unwrapped phase in
radians) for a directional coupler on the SiEPIC EBeam silicon photonics
process (UBC's open-access e-beam-lithography multi-project-wafer program),
200 nm gap, 10 um coupling length, TE polarization, C/O-band-adjacent
sweep. This is FDTD-simulated data calibrated against the actual foundry
process design rules (not a measured wafer trace, and not from a
commercial/NDA-gated PDK) — the closest thing to real foundry component
data that is genuinely open and redistributable. `examples/13_pdk_import.py`
documents this distinction explicitly.

**Why vendored, not fetched live**: every example in this repo runs in CI
on every push (`.github/workflows/tests.yml`); a live GitHub API/raw-content
fetch would make CI depend on network access and GitHub's rate limits for
no benefit, since the source file is small, stable, and MIT-licensed.

**Reproducing the fetch** (if you want the current upstream version instead
of this pinned commit):

```bash
gh api "repos/SiEPIC/SiEPIC_EBeam_PDK/contents/klayout/EBeam/CML/EBeam/source_data/ebeam_dc_te1550/dc_gap=200nm_Lc=10um.sparam" \
  --jq '.content' | base64 -d > dc_gap200nm_Lc10um.sparam
```
