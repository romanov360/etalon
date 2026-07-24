"""SiPhon — an open silicon photonics design toolkit.

Material dispersion models, waveguide mode solvers (EIM and semi-vectorial
finite-difference), S-parameter component models, a netlist-level circuit
solver, WDM system helpers, a closed-form optical link-budget engine for
AI-datacenter interconnects, parameter extraction from measured spectra,
laser reliability arithmetic, and Monte Carlo corner/yield analysis.
"""

from . import (
    circuit,
    components,
    constants,
    extract,
    fdmode,
    isi,
    link,
    materials,
    montecarlo,
    reliability,
    waveguide,
    wdm,
)

__all__ = [
    "circuit",
    "components",
    "constants",
    "extract",
    "fdmode",
    "isi",
    "link",
    "materials",
    "montecarlo",
    "reliability",
    "waveguide",
    "wdm",
]
__version__ = "0.1.0"
