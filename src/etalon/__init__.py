"""Etalon — an open silicon photonics design toolkit.

Material dispersion models, waveguide mode solvers (EIM and semi-vectorial
finite-difference), S-parameter component models, a netlist-level circuit
solver, WDM system helpers, ring-to-ring thermal crosstalk, Touchstone
(.sNp) file I/O for measured/foundry data, closed-form zero-forcing FFE
equalizer taps, a closed-form optical link-budget engine for AI-datacenter
interconnects, parameter extraction from measured spectra, laser
reliability arithmetic, and Monte Carlo corner/yield analysis.
"""

from . import (
    circuit,
    components,
    constants,
    equalize,
    extract,
    fdmode,
    isi,
    link,
    materials,
    montecarlo,
    reliability,
    thermal,
    touchstone,
    waveguide,
    wdm,
)

__all__ = [
    "circuit",
    "components",
    "constants",
    "equalize",
    "extract",
    "fdmode",
    "isi",
    "link",
    "materials",
    "montecarlo",
    "reliability",
    "thermal",
    "touchstone",
    "waveguide",
    "wdm",
]
__version__ = "0.1.0"
