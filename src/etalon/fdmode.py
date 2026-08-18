"""Semi-vectorial finite-difference mode solver for strip/rib waveguides.

This module solves the scalar-per-polarization ("semi-vectorial") wave
equation on the waveguide cross-section with finite differences, giving
Etalon an in-library reference against which its own effective index
method (:class:`etalon.waveguide.Waveguide`) can be checked.

Physics
-------
For the quasi-TE mode (dominant electric field component Ex, parallel to
the wafer plane) the semi-vectorial equation is

    d/dx [ (1/eps) d(eps Ex)/dx ] + d^2 Ex / dy^2 + k0^2 eps Ex = beta^2 Ex

with eps = n(x, y)^2, k0 = 2 pi / wavelength, and n_eff = beta / k0. The
first term enforces continuity of the normal displacement eps*Ex across
vertical (x-normal) interfaces — the Stern (1988) scheme — while Ex and
dEx/dy stay continuous across horizontal interfaces. Quasi-TM (dominant
Ey) is the same with x and y roles swapped. Polarization coupling is
neglected entirely: each mode is a single field component.

Discretization: nodes at the centers of uniform rectangular cells. The
grid is uniform per axis: the x pitch is dx_um snapped *down* so that
x = +/- width/2 falls exactly on cell boundaries, and the y pitch
targets dx_um / 4 (snapped down so y = 0 and y = height fall on cell
boundaries). The finer y pitch is deliberate: the vertical direction
carries the fastest field variation (thin, high-index-contrast core
layer), and its measured O(h^2) error constant is ~5x the horizontal
one, so a quarter-pitch y grid balances the two error contributions at
modest extra cost. Aligning interfaces to cell boundaries removes the
first-order n_eff jitter that node/interface misalignment otherwise
causes. Permittivity is area-averaged over each cell (only rib slab
interfaces can then straddle a cell), interface continuity is built into
the off-diagonal coefficients by harmonic (1/eps) averaging between
neighbouring cells, and the boundary condition is Dirichlet (field = 0)
at the edge of the padded window. The sparse eigenproblem is solved by
shift-invert Arnoldi (:func:`scipy.sparse.linalg.eigs`) around
beta^2 = k0^2 n_core^2, i.e. just above the largest possible guided-mode
eigenvalue, so the largest-n_eff modes converge first.

Honesty limits
--------------
Budgeting-to-design grade, not signoff. The ~1e-3 n_eff accuracy at the
default dx = 0.02 um is DISCRETIZATION accuracy relative to the exact
semi-vectorial solution. The semi-vectorial approximation itself (no
polarization coupling: no hybrid modes, no TE/TM anti-crossings) deviates
from full-vectorial n_eff by a few 1e-2 for high-contrast strips — e.g.
~4e-2 for the 500x220 nm SOI strip TE0, where EIM carries a similar-size
bias in the same direction; the FD-minus-EIM difference therefore probes
the EIM's *lateral* approximation, not its full error. Weakly guided
modes near cutoff and quasi-TM modes with strong corner fields are less
accurate still. No PML (leaky/radiating modes are truncated by the
Dirichlet wall), no loss. Use a full-vectorial solver for final signoff.

Conventions: wavelengths and dimensions in um; quasi-TE means dominant
E-field parallel to the wafer plane. Field arrays are indexed
``field[j, i]`` at ``(x_um[i], y_um[j])`` with y = 0 at the bottom of the
core (top of the buried oxide).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import materials
from .waveguide import slab_neffs


@dataclass(frozen=True, eq=False)
class FdMode:
    """One guided mode from the finite-difference solver.

    Attributes
    ----------
    neff : effective index beta / k0 (dimensionless).
    polarization : 'TE' (dominant Ex) or 'TM' (dominant Ey).
    field : real 2D array, shape (len(y_um), len(x_um)); the dominant
        transverse E component sampled at (x_um[i], y_um[j]) as
        ``field[j, i]``, normalized so max |field| = 1 (arbitrary units —
        the semi-vectorial solve carries no absolute power scale).
    x_um, y_um : 1D grid coordinates in um. x = 0 is the horizontal
        center of the core; y = 0 is the bottom of the core.
    """

    neff: float
    polarization: str
    field: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray


def _cell_fraction(centers: np.ndarray, half: float, lo: float, hi: float) -> np.ndarray:
    """Fraction of each cell [c-half, c+half] that lies inside [lo, hi]."""
    a = centers - half
    b = centers + half
    return np.clip((np.minimum(b, hi) - np.maximum(a, lo)) / (b - a), 0.0, 1.0)


def _epsilon_map(
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
    dy: float,
    width_um: float,
    height_um: float,
    slab_um: float,
    eps_core: float,
    eps_clad: float,
    eps_box: float,
) -> np.ndarray:
    """Cell-averaged permittivity eps(x, y) = n^2, shape (ny, nx).

    Geometry: buried oxide for y < 0; core rectangle |x| <= width/2,
    0 <= y <= height; for a rib, a residual core slab 0 <= y <= slab_um at
    all x; cladding everywhere else. Each grid cell gets the area-weighted
    average of eps over the cell. With the snapped grid the core outline
    coincides with cell boundaries, so only rib slab interfaces are ever
    averaged.
    """
    fx_core = _cell_fraction(x, dx / 2.0, -width_um / 2.0, width_um / 2.0)  # (nx,)
    fy_core = _cell_fraction(y, dy / 2.0, 0.0, height_um)  # (ny,)
    fy_slab = (
        _cell_fraction(y, dy / 2.0, 0.0, slab_um) if slab_um > 0 else np.zeros_like(y)
    )
    fy_box = _cell_fraction(y, dy / 2.0, -1e9, 0.0)  # (ny,)

    f_core = np.outer(fy_core, fx_core) + np.outer(fy_slab, 1.0 - fx_core)
    f_box = np.broadcast_to(fy_box[:, None], (y.size, x.size))
    f_clad = 1.0 - f_core - f_box
    return f_core * eps_core + f_box * eps_box + f_clad * eps_clad


def _assemble(
    eps: np.ndarray, dx: float, dy: float, k0: float, polarization: str
) -> sp.coo_matrix:
    """Assemble the semi-vectorial operator A with A e = beta^2 e.

    Stern coefficients: along the polarization direction (x for TE, y for
    TM) the term d/du [(1/eps) d(eps E)/du] is discretized in flux form
    with u = eps E continuous and flux v = (1/eps) du/du continuous, so
    the cell-boundary coefficient is the harmonic mean of 1/eps, i.e.
    2 / (eps_0 + eps_p). Coupling from node 0 to neighbour p is then
    2 eps_p / (eps_0 + eps_p) / h^2 with matching diagonal term
    -2 eps_0 / (eps_0 + eps_p) / h^2 (Stern 1988). The other direction is
    the plain second difference. Dirichlet boundaries: edge nodes simply
    lack the outside neighbour (field 0 there).
    """
    ny, nx = eps.shape
    n = ny * nx
    dx2 = dx * dx
    dy2 = dy * dy

    epad = np.pad(eps, 1, mode="edge")
    e0 = eps
    e_e = epad[1:-1, 2:]  # east  (x + dx)
    e_w = epad[1:-1, :-2]  # west  (x - dx)
    e_n = epad[2:, 1:-1]  # north (y + dy)
    e_s = epad[:-2, 1:-1]  # south (y - dy)

    if polarization == "TE":  # dominant Ex: Stern averaging along x
        c_e = 2.0 * e_e / (e0 + e_e) / dx2
        c_w = 2.0 * e_w / (e0 + e_w) / dx2
        c_n = np.full_like(e0, 1.0 / dy2)
        c_s = np.full_like(e0, 1.0 / dy2)
        diag = (
            -(2.0 * e0 / (e0 + e_e) + 2.0 * e0 / (e0 + e_w)) / dx2
            - 2.0 / dy2
            + k0 * k0 * e0
        )
    else:  # TM, dominant Ey: Stern averaging along y
        c_n = 2.0 * e_n / (e0 + e_n) / dy2
        c_s = 2.0 * e_s / (e0 + e_s) / dy2
        c_e = np.full_like(e0, 1.0 / dx2)
        c_w = np.full_like(e0, 1.0 / dx2)
        diag = (
            -(2.0 * e0 / (e0 + e_n) + 2.0 * e0 / (e0 + e_s)) / dy2
            - 2.0 / dx2
            + k0 * k0 * e0
        )

    idx = np.arange(n).reshape(ny, nx)
    rows = np.concatenate(
        [
            idx.ravel(),
            idx[:, :-1].ravel(),  # -> east neighbour exists
            idx[:, 1:].ravel(),  # -> west neighbour exists
            idx[:-1, :].ravel(),  # -> north neighbour exists
            idx[1:, :].ravel(),  # -> south neighbour exists
        ]
    )
    cols = np.concatenate(
        [
            idx.ravel(),
            idx[:, 1:].ravel(),
            idx[:, :-1].ravel(),
            idx[1:, :].ravel(),
            idx[:-1, :].ravel(),
        ]
    )
    vals = np.concatenate(
        [
            diag.ravel(),
            c_e[:, :-1].ravel(),
            c_w[:, 1:].ravel(),
            c_n[:-1, :].ravel(),
            c_s[1:, :].ravel(),
        ]
    )
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n))


def solve_modes(
    width_um: float,
    height_um: float,
    wl_um: float,
    *,
    slab_um: float = 0.0,
    core: str = "si",
    cladding: str = "sio2",
    box: str = "sio2",
    polarization: str = "TE",
    n_modes: int = 2,
    dx_um: float = 0.02,
    pad_um: float = 1.5,
) -> list[FdMode]:
    """Solve for guided quasi-TE or quasi-TM modes of a strip/rib waveguide.

    Semi-vectorial finite differences on a rectangular grid, uniform per
    axis: x pitch ~dx_um and y pitch ~dx_um/4 (each snapped down so the
    core outline lies on cell boundaries; see module docstring for why y
    is meshed finer), over a window extending at least pad_um beyond the
    core on all four sides, with field = 0 (Dirichlet) at the window
    edge. See the module docstring for the discretized equation and its
    limits.

    Parameters
    ----------
    width_um, height_um : core cross-section in um.
    wl_um : vacuum wavelength in um.
    slab_um : residual slab thickness in um for a rib; 0 for a strip.
    core, cladding, box : material names for :mod:`etalon.materials`
        (cladding = top/side, box = buried oxide below the core).
    polarization : 'TE' (dominant Ex) or 'TM' (dominant Ey).
    n_modes : maximum number of guided modes to return.
    dx_um : requested horizontal grid pitch in um. The actual x pitch is
        dx_um snapped down so width/2 is an integer number of cells; the
        y pitch is dx_um/4 snapped down so height is an integer number
        of cells. Default 0.02 gives roughly 1e-3 n_eff accuracy for
        well-guided SOI modes; halve it for a convergence check rather
        than trusting one grid.
    pad_um : cladding/box padding around the core in um. Must comfortably
        exceed the evanescent decay length of the slowest-decaying mode
        (1.5 um is ample for well-guided SOI modes at 1.55 um).

    Returns
    -------
    list[FdMode]
        Guided modes with max(n_cladding, n_box) < n_eff < n_core, sorted
        by decreasing n_eff, at most n_modes long. May be shorter (or
        empty) if the structure guides fewer modes.

    Raises
    ------
    ValueError
        On non-physical geometry, bad polarization, non-positive grid
        parameters, or a core index not above both claddings.
    """
    if width_um <= 0 or height_um <= 0:
        raise ValueError("width_um and height_um must be positive")
    if not 0 <= slab_um < height_um:
        raise ValueError("slab_um must satisfy 0 <= slab_um < height_um")
    if dx_um <= 0 or pad_um <= 0:
        raise ValueError("dx_um and pad_um must be positive")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    pol = polarization.upper()
    if pol not in ("TE", "TM"):
        raise ValueError(f"polarization must be 'TE' or 'TM', got {polarization!r}")

    n_core = float(materials.index(core, wl_um))
    n_clad = float(materials.index(cladding, wl_um))
    n_box = float(materials.index(box, wl_um))
    if n_core <= max(n_clad, n_box):
        raise ValueError("core index must exceed cladding and box indices")

    k0 = 2.0 * np.pi / wl_um

    # Snap pitches *down* (ceil on the cell count) so the core outline
    # lands exactly on cell boundaries, then put nodes at cell centers.
    # y is meshed at a quarter of dx_um: the vertical error constant is
    # ~5x the horizontal one (see module docstring). The x grid is
    # mirror-symmetric about x = 0, so mode symmetry is exact on the grid.
    dx = (width_um / 2.0) / max(1, int(np.ceil(width_um / 2.0 / dx_um)))
    dy = height_um / max(1, int(np.ceil(4.0 * height_um / dx_um)))
    mx = int(round(width_um / 2.0 / dx)) + int(np.ceil(pad_um / dx))
    x = (np.arange(2 * mx) - mx + 0.5) * dx
    my_lo = int(np.ceil(pad_um / dy))
    my_hi = int(round(height_um / dy)) + int(np.ceil(pad_um / dy))
    y = (np.arange(my_lo + my_hi) - my_lo + 0.5) * dy

    eps = _epsilon_map(
        x, y, dx, dy, width_um, height_um, slab_um, n_core**2, n_clad**2, n_box**2
    )
    a = _assemble(eps, dx, dy, k0, pol).tocsc()

    # Shift-invert just below the theoretical eigenvalue ceiling
    # beta^2 = k0^2 n_core^2: the nearest eigenvalues are the
    # largest-n_eff (best guided) modes.
    sigma = (k0 * n_core) ** 2 * (1.0 - 1e-6)
    k_req = min(max(n_modes + 2, 4), a.shape[0] - 2)
    vals, vecs = spla.eigs(a, k=k_req, sigma=sigma, which="LM")

    n_min = max(n_clad, n_box)
    if slab_um > 0.0:
        # A rib's residual slab supports laterally-unbound slab modes: any
        # eigenvector with n_eff below the slab's fundamental effective
        # index is lateral radiation discretized by the Dirichlet wall
        # (pad-dependent artifact), not a guided mode of the rib.
        slab_modes = slab_neffs(n_core, n_clad, n_box, slab_um, wl_um, pol)
        if slab_modes:
            n_min = max(n_min, slab_modes[0])
    modes: list[FdMode] = []
    order = np.argsort(-vals.real)
    for m in order:
        beta2 = float(vals[m].real)
        if beta2 <= 0.0:
            continue
        neff = float(np.sqrt(beta2) / k0)
        if not (n_min < neff < n_core):
            continue
        v = vecs[:, m]
        peak = v[np.argmax(np.abs(v))]
        field = (v / peak).real.reshape(y.size, x.size)
        modes.append(FdMode(neff=neff, polarization=pol, field=field, x_um=x, y_um=y))
        if len(modes) == n_modes:
            break
    return modes
