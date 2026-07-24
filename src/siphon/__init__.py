"""SiPhon — an open silicon photonics design toolkit.

Material dispersion models, waveguide mode solvers, S-parameter component
models, a netlist-level circuit solver, WDM system helpers, and a closed-form
optical link-budget engine for AI-datacenter interconnects.
"""

from . import (
    circuit,
    components,
    constants,
    link,
    materials,
    montecarlo,
    waveguide,
    wdm,
)

__all__ = [
    "circuit",
    "components",
    "constants",
    "link",
    "materials",
    "montecarlo",
    "waveguide",
    "wdm",
]
__version__ = "0.1.0"
