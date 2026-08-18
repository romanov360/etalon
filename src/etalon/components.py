"""Analytic S-parameter models of elementary photonic components.

Conventions
-----------
Every model exposes ``ports: tuple[str, ...]`` and
``s_params(wl: np.ndarray) -> np.ndarray`` of shape
``(len(wl), n_ports, n_ports)``, complex, with ``b_i = sum_j S[k, i, j] a_j``
at wavelength ``wl[k]`` (um). Row/column order equals ``ports`` order.

* Wavelengths and geometry in um; loss in dB (per cm where named); powers
  in mW as named.
* Propagation uses the delay convention ``exp(-1j * 2*pi * n_eff * L / wl)``,
  consistently for ALL models. Ring transfer functions follow Bogaerts et
  al., Laser Photon. Rev. 6, 47 (2012), but with the round-trip phase
  conjugated to ``exp(-1j*phi)`` — Bogaerts writes ``e^{+i phi}`` under the
  physics ``e^{-i omega t}`` convention, and using it unconjugated next to
  ``exp(-1j*beta*L)`` elements flips the sign of ring group delay and breaks
  interferometric composition in :class:`etalon.circuit.Circuit`. Magnitudes,
  resonance positions, and linewidths are identical in either convention.
* All models here are reciprocal (``S == S.T`` at every wavelength).
  Lossless models are unitary, except :class:`YBranch` (see its docstring:
  the reverse combining path radiates, so an ideal 1x2 splitter cannot be
  unitary as a 3-port).
* Dispersion is linearized about ``wl0_um``:
  ``n_eff(wl) = neff0 + (wl - wl0) * (neff0 - ng) / wl0``,
  which follows from ``n_g = n_eff - wl * d(n_eff)/d(wl)``. Valid over
  bandwidths where group-velocity dispersion is negligible (tens of nm for
  typical SOI strip waveguides).
* No back-reflection is modeled anywhere (all diagonal S entries are 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .waveguide import Waveguide


# --- shared helpers -------------------------------------------------------


def _wl_array(wavelength_um) -> np.ndarray:
    wl = np.atleast_1d(np.asarray(wavelength_um, dtype=float))
    if np.any(wl <= 0):
        raise ValueError("wavelength must be positive (in um)")
    return wl


def _amp_from_db(loss_db: float) -> float:
    """Field amplitude factor for a power loss in dB (>= 0 means loss)."""
    return 10.0 ** (-loss_db / 20.0)


def _linear_neff(wl: np.ndarray, neff0: float, ng: float, wl0: float) -> np.ndarray:
    """Linearized effective index n_eff(wl) about wl0 (see module docstring)."""
    return neff0 + (wl - wl0) * (neff0 - ng) / wl0


def _two_port(s21: np.ndarray) -> np.ndarray:
    """Reciprocal 2-port S with zero reflection and given transmission."""
    s = np.zeros((s21.shape[0], 2, 2), dtype=complex)
    s[:, 0, 1] = s21
    s[:, 1, 0] = s21
    return s


def _check_coupling(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a power fraction in [0, 1]; got {value}")


# --- straight waveguide ---------------------------------------------------


@dataclass
class Straight:
    """Straight waveguide section of given length.

    S21 = 10**(-loss_db_total/20) * exp(-1j * 2*pi * n_eff(wl) * L / wl),
    with loss_db_total = loss_db_per_cm * length_um * 1e-4 and n_eff
    linearized about wl0_um (module docstring).

    Parameters
    ----------
    length_um : physical length in um.
    neff0, ng : effective and group index at wl0_um.
    loss_db_per_cm : propagation loss in dB/cm.
    wl0_um : expansion wavelength in um.
    """

    length_um: float
    neff0: float
    ng: float
    loss_db_per_cm: float = 0.0
    wl0_um: float = 1.55

    ports: ClassVar[tuple[str, ...]] = ("in", "out")

    def __post_init__(self):
        if self.length_um < 0:
            raise ValueError("length_um must be non-negative")

    @classmethod
    def from_waveguide(
        cls,
        wg: Waveguide,
        length_um: float,
        loss_db_per_cm: float = 0.0,
        wl0_um: float = 1.55,
        mode: str = "TE",
    ) -> "Straight":
        """Build a Straight from a mode-solver evaluation at wl0_um."""
        return cls(
            length_um=length_um,
            neff0=wg.neff(wl0_um, mode),
            ng=wg.group_index(wl0_um, mode),
            loss_db_per_cm=loss_db_per_cm,
            wl0_um=wl0_um,
        )

    def s_params(self, wl: np.ndarray) -> np.ndarray:
        wl = _wl_array(wl)
        neff = _linear_neff(wl, self.neff0, self.ng, self.wl0_um)
        amp = _amp_from_db(self.loss_db_per_cm * self.length_um * 1e-4)
        s21 = amp * np.exp(-1j * 2.0 * np.pi * neff * self.length_um / wl)
        return _two_port(s21)


# --- directional coupler --------------------------------------------------


@dataclass
class DirectionalCoupler:
    """Wavelength-flat 2x2 directional coupler.

    Through amplitude t = sqrt(1 - coupling), cross amplitude 1j*sqrt(coupling)
    (the 90-degree cross phase required for a lossless reciprocal coupler),
    both scaled by 10**(-loss_db/20). in0->out0 is through, in0->out1 cross.
    No back-reflection and no in0<->in1 coupling. Real couplers have
    wavelength-dependent coupling; treat ``coupling`` as valid near the
    design wavelength only.

    Parameters
    ----------
    coupling : cross-coupled power fraction in [0, 1].
    loss_db : excess insertion loss in dB.
    """

    coupling: float = 0.5
    loss_db: float = 0.0

    ports: ClassVar[tuple[str, ...]] = ("in0", "in1", "out0", "out1")

    def __post_init__(self):
        _check_coupling("coupling", self.coupling)

    def _transfer(self) -> np.ndarray:
        """2x2 forward transfer matrix (outs from ins), symmetric."""
        t = np.sqrt(1.0 - self.coupling)
        k = 1j * np.sqrt(self.coupling)
        return _amp_from_db(self.loss_db) * np.array([[t, k], [k, t]], dtype=complex)

    def s_params(self, wl: np.ndarray) -> np.ndarray:
        wl = _wl_array(wl)
        m = self._transfer()
        s = np.zeros((wl.size, 4, 4), dtype=complex)
        s[:, 2:, :2] = m
        s[:, :2, 2:] = m.T
        return s


# --- Y-branch -------------------------------------------------------------


@dataclass
class YBranch:
    """Ideal 50/50 1x2 splitter/combiner.

    in->out0 and in->out1 each carry half the input power (times excess
    loss). An ideal lossless 1x2 splitter cannot be unitary as a 3-port:
    reciprocity forces the combining direction to radiate half the power of
    a single-arm input into the substrate (the antisymmetric mode), so only
    power conservation ``sum_i |S_i j|^2 <= 1`` holds, not unitarity.
    out0<->out1 coupling and all reflections are 0.

    Parameters
    ----------
    excess_loss_db : excess insertion loss in dB (on top of the 3 dB split).
    """

    excess_loss_db: float = 0.0

    ports: ClassVar[tuple[str, ...]] = ("in", "out0", "out1")

    def s_params(self, wl: np.ndarray) -> np.ndarray:
        wl = _wl_array(wl)
        arm = _amp_from_db(self.excess_loss_db) / np.sqrt(2.0)
        s = np.zeros((wl.size, 3, 3), dtype=complex)
        s[:, 1, 0] = s[:, 0, 1] = arm
        s[:, 2, 0] = s[:, 0, 2] = arm
        return s


# --- phase shifters -------------------------------------------------------


@dataclass
class PhaseShifter:
    """Lumped (wavelength-flat) phase shifter.

    S21 = 10**(-loss_db/20) * exp(-1j * phase_rad); a positive phase_rad is
    an added optical delay, consistent with the exp(-1j*beta*L) convention.

    Parameters
    ----------
    phase_rad : applied phase in radians.
    loss_db : insertion loss in dB.
    """

    phase_rad: float = 0.0
    loss_db: float = 0.0

    ports: ClassVar[tuple[str, ...]] = ("in", "out")

    def s_params(self, wl: np.ndarray) -> np.ndarray:
        wl = _wl_array(wl)
        s21 = _amp_from_db(self.loss_db) * np.exp(-1j * self.phase_rad)
        return _two_port(np.full(wl.size, s21, dtype=complex))


@dataclass
class ThermalPhaseShifter:
    """Thermo-optic phase shifter driven by heater power.

    phase = pi * power_mw / p_pi_mw, applied as in :class:`PhaseShifter`.
    Assumes the linear thermo-optic regime (phase proportional to dissipated
    power); thermal crosstalk and transient response are not modeled.

    Parameters
    ----------
    power_mw : heater drive power in mW.
    p_pi_mw : power for a pi phase shift, in mW.
    loss_db : insertion loss in dB.
    """

    power_mw: float
    p_pi_mw: float
    loss_db: float = 0.0

    ports: ClassVar[tuple[str, ...]] = ("in", "out")

    def __post_init__(self):
        if self.p_pi_mw <= 0:
            raise ValueError("p_pi_mw must be positive")

    @property
    def phase_rad(self) -> float:
        """Applied phase in radians, pi * power_mw / p_pi_mw."""
        return np.pi * self.power_mw / self.p_pi_mw

    def s_params(self, wl: np.ndarray) -> np.ndarray:
        return PhaseShifter(self.phase_rad, self.loss_db).s_params(wl)


# --- grating coupler ------------------------------------------------------


@dataclass
class GratingCoupler:
    """Fiber-to-chip grating coupler with a Gaussian power response.

    IL_db(wl) = peak_il_db + ((wl - center_um) / (bw_1db_um / 2))**2,
    i.e. exactly 1 dB of excess loss at +-bw_1db/2 from center — the usual
    Gaussian (parabolic-in-dB) fit to measured grating spectra. Transmission
    phase is not modeled (S21 real). Valid within roughly +-2 bandwidths of
    center; real gratings roll off asymmetrically further out.

    Parameters
    ----------
    peak_il_db : insertion loss at the center wavelength, dB.
    bw_1db_nm : full 1-dB bandwidth in nm.
    center_um : center wavelength in um.
    """

    peak_il_db: float = 4.0
    bw_1db_nm: float = 35.0
    center_um: float = 1.55

    ports: ClassVar[tuple[str, ...]] = ("in", "out")

    def __post_init__(self):
        if self.bw_1db_nm <= 0:
            raise ValueError("bw_1db_nm must be positive")

    def il_db(self, wl: np.ndarray) -> np.ndarray:
        """Insertion loss in dB at the given wavelengths (um)."""
        wl = _wl_array(wl)
        half_bw_um = 0.5 * self.bw_1db_nm * 1e-3
        return self.peak_il_db + ((wl - self.center_um) / half_bw_um) ** 2

    def s_params(self, wl: np.ndarray) -> np.ndarray:
        return _two_port(_amp_from_db(self.il_db(wl)))


# --- ring resonators ------------------------------------------------------


@dataclass
class RingAllPass:
    """All-pass (notch) ring resonator, single bus.

    through = (t - a e^{-i phi}) / (1 - t a e^{-i phi})  [Bogaerts 2012,
    Eq. 2, conjugated into this library's exp(-1j*beta*L) convention]
    with self-coupling t = sqrt(1 - kappa_power), round-trip amplitude
    a = 10**(-loss_db_per_cm * circumference_um * 1e-4 / 20), and round-trip
    phase phi = 2*pi * n_eff(wl) * circumference_um / wl (linearized n_eff).
    Critical coupling (a == t) gives full extinction at resonance.

    Parameters
    ----------
    circumference_um : ring round-trip length in um.
    neff0, ng : effective and group index at wl0_um.
    kappa_power : bus-ring power coupling in [0, 1].
    loss_db_per_cm : round-trip propagation loss in dB/cm.
    wl0_um : dispersion expansion wavelength in um.
    """

    circumference_um: float
    neff0: float
    ng: float
    kappa_power: float = 0.1
    loss_db_per_cm: float = 0.0
    wl0_um: float = 1.55

    ports: ClassVar[tuple[str, ...]] = ("in", "out")

    def __post_init__(self):
        if self.circumference_um <= 0:
            raise ValueError("circumference_um must be positive")
        _check_coupling("kappa_power", self.kappa_power)

    def _round_trip(self, wl: np.ndarray) -> tuple[float, np.ndarray]:
        a = _amp_from_db(self.loss_db_per_cm * self.circumference_um * 1e-4)
        neff = _linear_neff(wl, self.neff0, self.ng, self.wl0_um)
        phi = 2.0 * np.pi * neff * self.circumference_um / wl
        return a, phi

    def s_params(self, wl: np.ndarray) -> np.ndarray:
        wl = _wl_array(wl)
        a, phi = self._round_trip(wl)
        t = np.sqrt(1.0 - self.kappa_power)
        ae = a * np.exp(-1j * phi)
        return _two_port((t - ae) / (1.0 - t * ae))


@dataclass
class RingAddDrop:
    """Add-drop ring resonator with two bus waveguides.

    Analytic transfer functions per Bogaerts et al., Laser Photon. Rev. 6,
    47 (2012), Eqs. 4-5, conjugated into this library's exp(-1j*beta*L)
    convention (see module docstring), with self-couplings
    t_i = sqrt(1 - kappa_i), round-trip amplitude a and phase phi as in
    :class:`RingAllPass` (D = 1 - t1 t2 a e^{-i phi}):

        in -> through :  (t1 - t2 a e^{-i phi}) / D
        in -> drop    :  -sqrt(kappa1 kappa2) sqrt(a) e^{-i phi/2} / D
        add -> drop   :  (t2 - t1 a e^{-i phi}) / D
        add -> through:  same as in -> drop (device symmetry)

    The drop path traverses half the ring (sqrt(a) e^{-i phi/2}); couplers
    are assumed point-like and lossless. in<->add and through<->drop carry
    counter-propagating signals with no direct coupling, so those S entries
    are 0, as are all reflections. The 4x4 is reciprocal, and unitary when
    loss_db_per_cm == 0.

    Parameters
    ----------
    circumference_um : ring round-trip length in um.
    neff0, ng : effective and group index at wl0_um.
    kappa1_power, kappa2_power : input/drop bus power couplings in [0, 1].
    loss_db_per_cm : round-trip propagation loss in dB/cm.
    wl0_um : dispersion expansion wavelength in um.
    """

    circumference_um: float
    neff0: float
    ng: float
    kappa1_power: float = 0.1
    kappa2_power: float = 0.1
    loss_db_per_cm: float = 0.0
    wl0_um: float = 1.55

    ports: ClassVar[tuple[str, ...]] = ("in", "through", "add", "drop")

    def __post_init__(self):
        if self.circumference_um <= 0:
            raise ValueError("circumference_um must be positive")
        _check_coupling("kappa1_power", self.kappa1_power)
        _check_coupling("kappa2_power", self.kappa2_power)

    def s_params(self, wl: np.ndarray) -> np.ndarray:
        wl = _wl_array(wl)
        a = _amp_from_db(self.loss_db_per_cm * self.circumference_um * 1e-4)
        neff = _linear_neff(wl, self.neff0, self.ng, self.wl0_um)
        phi = 2.0 * np.pi * neff * self.circumference_um / wl
        t1 = np.sqrt(1.0 - self.kappa1_power)
        t2 = np.sqrt(1.0 - self.kappa2_power)
        ae = a * np.exp(-1j * phi)
        half = np.sqrt(a) * np.exp(-1j * phi / 2.0)
        denom = 1.0 - t1 * t2 * ae

        thru = (t1 - t2 * ae) / denom
        drop = -np.sqrt(self.kappa1_power * self.kappa2_power) * half / denom
        add_thru = (t2 - t1 * ae) / denom

        s = np.zeros((wl.size, 4, 4), dtype=complex)
        s[:, 1, 0] = s[:, 0, 1] = thru  # in <-> through
        s[:, 3, 0] = s[:, 0, 3] = drop  # in <-> drop
        s[:, 2, 1] = s[:, 1, 2] = drop  # add <-> through
        s[:, 3, 2] = s[:, 2, 3] = add_thru  # add <-> drop
        return s


# --- Mach-Zehnder interferometer ------------------------------------------


@dataclass
class MZI:
    """Unbalanced Mach-Zehnder interferometer, composed analytically.

    Forward transfer matrix M = C_out @ diag(arm phases) @ C_in, where the
    C's are :class:`DirectionalCoupler` 2x2 transfer blocks and the arms
    carry exp(-1j * 2*pi * n_eff(wl) * L / wl) with lengths (dl_um, 0):
    only the differential path is modeled; common arm length adds a global
    phase and loss omitted here. Propagation loss (loss_db_per_cm) therefore
    acts on the dl_um arm only. Port mapping matches
    :class:`DirectionalCoupler`: in0 -> out0 is the bar path.

    With 50/50 couplers: |bar|^2 = sin^2(dphi/2), |cross|^2 = cos^2(dphi/2)
    with dphi = 2*pi*n_eff(wl)*dl/wl; FSR ~= wl^2/(ng*dl).

    Parameters
    ----------
    dl_um : arm length imbalance in um.
    neff0, ng : arm effective and group index at wl0_um.
    coupling_in, coupling_out : splitter/combiner power couplings in [0, 1].
    loss_db_per_cm : arm propagation loss in dB/cm (differential arm only).
    wl0_um : dispersion expansion wavelength in um.
    """

    dl_um: float
    neff0: float
    ng: float
    coupling_in: float = 0.5
    coupling_out: float = 0.5
    loss_db_per_cm: float = 0.0
    wl0_um: float = 1.55

    ports: ClassVar[tuple[str, ...]] = ("in0", "in1", "out0", "out1")

    def __post_init__(self):
        if self.dl_um < 0:
            raise ValueError("dl_um must be non-negative")
        _check_coupling("coupling_in", self.coupling_in)
        _check_coupling("coupling_out", self.coupling_out)

    def s_params(self, wl: np.ndarray) -> np.ndarray:
        wl = _wl_array(wl)
        c_in = DirectionalCoupler(self.coupling_in)._transfer()
        c_out = DirectionalCoupler(self.coupling_out)._transfer()

        arm = Straight(
            self.dl_um, self.neff0, self.ng, self.loss_db_per_cm, self.wl0_um
        ).s_params(wl)[:, 1, 0]
        phases = np.zeros((wl.size, 2, 2), dtype=complex)
        phases[:, 0, 0] = arm  # differential arm, length dl_um
        phases[:, 1, 1] = 1.0  # reference arm, length 0

        m = c_out @ phases @ c_in  # (n, 2, 2) forward transfer
        s = np.zeros((wl.size, 4, 4), dtype=complex)
        s[:, 2:, :2] = m
        s[:, :2, 2:] = np.swapaxes(m, 1, 2)
        return s


# --- resonator figure-of-merit helpers -------------------------------------


def ring_fsr_um(wl_um: float, ng: float, circumference_um: float) -> float:
    """Ring free spectral range FSR = wl^2 / (n_g L), in um."""
    return wl_um**2 / (ng * circumference_um)


def loaded_q(wl_um: float, fwhm_um: float) -> float:
    """Loaded quality factor Q = wl / FWHM (both in um)."""
    return wl_um / fwhm_um


def finesse(fsr: float, fwhm: float) -> float:
    """Resonator finesse F = FSR / FWHM (same units for both)."""
    return fsr / fwhm
