"""Touchstone (.sNp) S-parameter file I/O — the measured-data on-ramp.

:mod:`etalon.extract` fits Etalon models to measured spectra, but until now
"measured" always meant a numpy array the caller built by hand (or by
perturbing a synthetic spectrum). Real foundry PDKs, wafer-probe stations,
and ATE all speak Touchstone (IEEE/IBIS Touchstone(R) File Format,
versions 1.0 and 2.0): the format every VNA, S-parameter simulator, and PDK
component model exports. This module reads that format into the exact
shape :mod:`etalon.circuit` and :mod:`etalon.extract` already expect
(``ports`` + ``s_params(wl) -> (n_wl, n, n)`` complex, ``S[out, in]``
convention — see :mod:`etalon.circuit`'s module docstring), so a measured
.sNp file drops into ``fit_transmission`` or a ``Circuit`` instance exactly
like any Etalon component model. It also writes Etalon model output back
out as Touchstone, for round-tripping to other tools.

Format notes (the parts that are easy to get silently wrong)
--------------------------------------------------------------
* **Frequency, not wavelength.** Touchstone files are frequency-domain;
  this module converts to/from Etalon's wavelength-in-um convention via
  ``wl_um = C_UM_HZ / f_hz`` (vacuum), so a file's frequency ordering
  (ascending or descending) becomes whatever wavelength ordering results
  — :func:`read_touchstone` does not re-sort; :class:`TouchstoneData`
  handles either order (interpolation only needs monotonicity).
* **DB format is FIELD dB (20*log10|S|), not power dB (10*log10).**
  S-parameters are complex field ratios, and Touchstone's "dB" column is
  ``20*log10(|S|)`` with a separate angle column in degrees — this is a
  DIFFERENT convention from :func:`etalon.constants.db_to_linear`
  (10*log10, power ratio); reusing that helper here would silently be
  wrong by a factor of 2 in the exponent. This module carries its own
  field-dB conversion and never touches ``etalon.constants``'s power-dB
  one.
* **2-port column order is transposed relative to N>=3.** Per the
  Touchstone spec, a 2-port data line is
  ``f  S11  S21  S12  S22`` (each an MA/DB/RI pair) — S21 before S12. A
  naive row-major reshape of that flat order into a (2, 2) matrix gives
  ``[[S11, S21], [S12, S22]]``, which is the TRANSPOSE of the intended
  ``S[out, in]`` matrix. This is the single most common Touchstone
  parsing bug (see e.g. scikit-rf issue #8 / discussion #823) and this
  module special-cases n == 2 to place S21 and S12 correctly rather than
  reshaping. For n == 1 and n >= 3 the file order is already row-major
  (``S11, S12, ..., S1n, S21, ...``) and a plain reshape is correct.
* **Reference impedance** (the ``R <ohms>`` option-line token) is parsed
  and reported but not used: Etalon's S-parameters are already
  power-wave normalized (unitless), consistent with every model in
  :mod:`etalon.components`, so no renormalization is applied. A file
  written from a different reference impedance describes a physically
  different measurement setup; this module does not attempt
  renormalization — flag it via :attr:`TouchstoneData.reference_ohms` if
  it matters to your use.
* **Version 2.0 keywords**: only ``[Number of Ports]`` and ``[End]`` are
  recognized (enough to read the option-line-plus-data-block files this
  toolkit produces and the common PDK export case); other V2 keywords
  (``[Network Data]``, ``[Mixed-Mode Order]``, noise-parameter blocks,
  ...) raise ``NotImplementedError`` naming the keyword rather than
  silently misparsing the file.
* **A second option line is accepted only if it agrees with the first**
  (per spec, additional option lines are ignored — but every data row up
  to that point was already converted assuming the FIRST line's
  units/format applied throughout, so a differing second line most
  likely means two files were naively concatenated; this module raises
  rather than silently reinterpreting only part of the data).
* **Inline trailing comments on data lines** (``<data> ! note``, legal
  per spec and common in real exports) are stripped before parsing.

Conventions: wavelengths in um (Etalon-wide); frequency in Hz internally;
S-parameters complex, ``S[out, in]``, unitless (matching
:mod:`etalon.circuit`'s model protocol exactly, so a
:class:`TouchstoneData` instance IS a valid model: ``ports`` +
``s_params(wl)``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Speed of light expressed so that f[Hz] = C_UM_HZ / wl[um] (vacuum).
C_UM_HZ = 2.99792458e14

_FREQ_UNIT_TO_HZ = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
_DEFAULT_OPTIONS = ("ghz", "s", "ma", "r", "50")


def _port_labels(n: int) -> tuple[str, ...]:
    """Default port names ``p1..pn`` (1-indexed, matching Touchstone Sij)."""
    return tuple(f"p{i}" for i in range(1, n + 1))


@dataclass(frozen=True)
class TouchstoneData:
    """Parsed (or in-memory) Touchstone S-parameter data.

    Implements the Etalon component-model protocol (``ports`` +
    ``s_params(wl)``), so it can be passed directly to
    :func:`etalon.extract.fit_transmission`, wired into a
    :class:`etalon.circuit.Circuit`, or read with
    :meth:`transmission`/:meth:`transmission_db`.

    Attributes
    ----------
    freq_hz:
        Frequencies as stored in the file, ascending or descending
        exactly as given (1-D, length n_f, > 0).
    s:
        Complex S-matrix, shape ``(n_f, n_ports, n_ports)``, ``S[out, in]``.
    ports:
        Port name tuple, length n_ports (default ``p1..pn`` if the file
        carried no port-naming comments).
    reference_ohms:
        Reference impedance from the option line's ``R`` token (default
        50.0 if omitted); see the module docstring — not applied to
        ``s``.
    parameter:
        Touchstone parameter type from the option line ('s', 'y', 'z',
        'g', or 'h'); this module only interprets 's' data numerically
        (others are carried through unconverted, since Etalon's Circuit
        protocol is S-parameter-only) — see :func:`read_touchstone`.
    """

    freq_hz: np.ndarray
    s: np.ndarray
    ports: tuple[str, ...]
    reference_ohms: float = 50.0
    parameter: str = "s"

    def __post_init__(self):
        freq = np.asarray(self.freq_hz, dtype=float)
        s = np.asarray(self.s, dtype=complex)
        n = len(self.ports)
        if freq.ndim != 1 or freq.size < 1:
            raise ValueError("freq_hz must be a non-empty 1-D array")
        if np.any(freq <= 0):
            raise ValueError("freq_hz must be positive")
        if s.shape != (freq.size, n, n):
            raise ValueError(
                f"s shape {s.shape} does not match (n_freq={freq.size}, "
                f"n_ports={n}, n_ports={n})"
            )
        if len(set(self.ports)) != n:
            raise ValueError(f"port names must be unique; got {self.ports}")
        object.__setattr__(self, "freq_hz", freq)
        object.__setattr__(self, "s", s)

    @property
    def wl_um(self) -> np.ndarray:
        """Wavelengths in um, same order as :attr:`freq_hz` (vacuum, C/f)."""
        return C_UM_HZ / self.freq_hz

    def s_params(self, wl_um) -> np.ndarray:
        """Complex S-matrix at ``wl_um``, shape ``(len(wl_um), n, n)``.

        Linear interpolation (real and imaginary parts separately) onto
        the requested wavelength grid; raises ValueError if any requested
        point lies outside the file's frequency span (never
        extrapolates — beyond a 1e-9 relative tolerance that absorbs the
        text round-trip precision loss of writing frequencies through
        :func:`write_touchstone` and reading them back, so querying at
        exactly the wavelengths a file was written from does not spuriously
        fail), naming the valid range.
        """
        wl = np.atleast_1d(np.asarray(wl_um, dtype=float))
        f = C_UM_HZ / wl
        order = np.argsort(self.freq_hz)
        f_sorted = self.freq_hz[order]
        lo, hi = float(f_sorted[0]), float(f_sorted[-1])
        tol = 1e-9 * max(abs(lo), abs(hi))
        if np.any(f < lo - tol) or np.any(f > hi + tol):
            wl_lo, wl_hi = C_UM_HZ / hi, C_UM_HZ / lo
            raise ValueError(
                f"requested wavelength(s) outside the file's span "
                f"[{wl_lo:.6f}, {wl_hi:.6f}] um (refusing to extrapolate)"
            )
        f = np.clip(f, lo, hi)
        s_sorted = self.s[order]
        n = len(self.ports)
        out = np.empty((wl.size, n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                out[:, i, j] = np.interp(f, f_sorted, s_sorted[:, i, j].real) + 1j * np.interp(
                    f, f_sorted, s_sorted[:, i, j].imag
                )
        return out

    def transmission(self, wl_um, inport: str, outport: str) -> np.ndarray:
        """Complex S[outport, inport] interpolated onto ``wl_um``."""
        if inport not in self.ports or outport not in self.ports:
            raise ValueError(f"unknown port name; ports are {self.ports}")
        i, j = self.ports.index(outport), self.ports.index(inport)
        return self.s_params(wl_um)[:, i, j]

    def transmission_db(self, wl_um, inport: str, outport: str) -> np.ndarray:
        """Power transmission in dB (10*log10|S|^2) — Etalon's power-dB
        convention, NOT the file's field-dB, for consistency with
        :meth:`etalon.circuit.Circuit.transmission_db`."""
        t = self.transmission(wl_um, inport, outport)
        floor = 1e-16
        return 10.0 * np.log10(np.maximum(np.abs(t) ** 2, floor))


def _field_db_to_linear(db: np.ndarray) -> np.ndarray:
    """Touchstone DB format: dB = 20*log10|S| -> linear magnitude."""
    return 10.0 ** (np.asarray(db, dtype=float) / 20.0)


def _linear_to_field_db(mag: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_field_db_to_linear`, floored to avoid -inf."""
    return 20.0 * np.log10(np.maximum(np.asarray(mag, dtype=float), 1e-16))


_PORT_NAME_RE = re.compile(r"!\s*Port\[?(\d+)\]?\s*[=:]\s*(\S+)", re.IGNORECASE)


def read_touchstone(path) -> TouchstoneData:
    """Parse a .s1p/.s2p/.../.sNp Touchstone file into :class:`TouchstoneData`.

    Supports Touchstone 1.0 (port count from the ``.sNp`` extension) and
    the subset of 2.0 needed to read ``[Number of Ports]``-only files
    (raises ``NotImplementedError`` naming any other 2.0 keyword it
    encounters, rather than silently misparsing). Handles all three data
    formats (MA, DB, RI) and all four frequency units (Hz/kHz/MHz/GHz);
    see the module docstring for the 2-port column-order and DB-vs-power-dB
    pitfalls this function avoids. Port names are read from
    ``! Port[k] = name`` / ``! Port k : name``-style comment lines when
    present, else default to ``p1..pn``.

    Parameters
    ----------
    path : file path (str or :class:`pathlib.Path`); the number of ports
        for a 1.0-style file is taken from the extension (``.s2p`` -> 2),
        overridden by an explicit ``[Number of Ports]`` keyword if present.

    Raises
    ------
    ValueError : malformed option line, wrong number of data columns for
        the declared port count, non-finite values, or a file with no
        option line / no data.
    NotImplementedError : an unsupported Touchstone 2.0 keyword.
    """
    p = Path(path)
    text = p.read_text()

    m = re.search(r"\.s(\d+)p$", p.name, re.IGNORECASE)
    n_ports = int(m.group(1)) if m else None

    port_names: dict[int, str] = {}
    option_tokens: list[str] | None = None
    data_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("!"):
            pm = _PORT_NAME_RE.match(line)
            if pm:
                port_names[int(pm.group(1))] = pm.group(2)
            continue
        if line.startswith("["):
            key = line.split("]")[0].strip("[").strip().lower()
            if key == "number of ports":
                value = line.split("]", 1)[1].strip()
                try:
                    n_ports = int(value)
                except ValueError as exc:
                    raise ValueError(
                        f"{p}: [Number of Ports] has a non-integer value {value!r}"
                    ) from exc
            elif key == "end":
                break
            else:
                raise NotImplementedError(
                    f"Touchstone keyword [{key}] is not supported by this reader"
                )
            continue
        if line.startswith("#"):
            toks = line[1:].strip().lower().split()
            toks = list(toks) + list(_DEFAULT_OPTIONS[len(toks):])
            if option_tokens is not None:
                # Per spec, additional option lines are ignored — but only
                # if they AGREE with the first; two DIFFERING option lines
                # most likely means concatenated files with mismatched
                # units/format, and every data row up to here was already
                # converted assuming the first line applied throughout.
                # Refuse to guess rather than silently misconvert.
                if toks != option_tokens:
                    raise ValueError(
                        f"{p}: a second option line ('#...') disagrees with "
                        f"the first ({option_tokens} vs {toks}) — refusing to "
                        "guess which option line applies to which data rows "
                        "(likely concatenated files with mismatched units/format)"
                    )
                continue
            option_tokens = toks
            continue
        # Data lines may carry a trailing inline comment (legal per spec,
        # and common in real VNA/PDK exports) — strip it before parsing.
        line = line.split("!", 1)[0].strip()
        if line:
            data_lines.append(line)

    if option_tokens is None:
        raise ValueError(f"{p}: no option line ('#...') found")
    if n_ports is None:
        raise ValueError(
            f"{p}: cannot determine port count (no .sNp extension and no "
            "[Number of Ports] keyword)"
        )
    if n_ports < 1:
        raise ValueError(f"{p}: port count must be >= 1, parsed {n_ports}")

    freq_unit, param, fmt = option_tokens[0], option_tokens[1], option_tokens[2]
    if freq_unit not in _FREQ_UNIT_TO_HZ:
        raise ValueError(f"{p}: unknown frequency unit {freq_unit!r} in option line")
    if fmt not in ("ma", "db", "ri"):
        raise ValueError(f"{p}: unknown format {fmt!r} in option line (want MA/DB/RI)")
    ref_ohms = 50.0
    if "r" in option_tokens:
        r_idx = option_tokens.index("r")
        if r_idx + 1 < len(option_tokens):
            try:
                ref_ohms = float(option_tokens[r_idx + 1])
            except ValueError as exc:
                raise ValueError(
                    f"{p}: option line's R token has a non-numeric value "
                    f"{option_tokens[r_idx + 1]!r}"
                ) from exc

    values_per_freq = 1 + 2 * n_ports * n_ports
    flat = " ".join(data_lines).split()
    try:
        flat = np.array([float(v) for v in flat], dtype=float)
    except ValueError as exc:
        raise ValueError(f"{p}: non-numeric token in data section") from exc
    if flat.size == 0 or flat.size % values_per_freq != 0:
        raise ValueError(
            f"{p}: data section has {flat.size} values, not a multiple of "
            f"{values_per_freq} (= 1 + 2*{n_ports}^2 for a {n_ports}-port file)"
        )
    rows = flat.reshape(-1, values_per_freq)
    if not np.all(np.isfinite(rows)):
        raise ValueError(f"{p}: non-finite value in data section")

    freq_hz = rows[:, 0] * _FREQ_UNIT_TO_HZ[freq_unit]
    pairs = rows[:, 1:].reshape(-1, n_ports * n_ports, 2)

    if fmt == "ri":
        flat_s = pairs[:, :, 0] + 1j * pairs[:, :, 1]
    else:  # ma or db
        mag = pairs[:, :, 0]
        if fmt == "db":
            mag = _field_db_to_linear(mag)
        ang = np.deg2rad(pairs[:, :, 1])
        flat_s = mag * np.exp(1j * ang)

    n_f = freq_hz.size
    if n_ports == 2:
        # File order is S11, S21, S12, S22 — NOT row-major; place explicitly
        # rather than reshape (see module docstring).
        s = np.empty((n_f, 2, 2), dtype=complex)
        s[:, 0, 0] = flat_s[:, 0]  # S11
        s[:, 1, 0] = flat_s[:, 1]  # S21
        s[:, 0, 1] = flat_s[:, 2]  # S12
        s[:, 1, 1] = flat_s[:, 3]  # S22
    else:
        # n==1 or n>=3: file order is row-major (S11, S12, ..., S21, ...).
        s = flat_s.reshape(n_f, n_ports, n_ports)

    ports = tuple(port_names.get(i, f"p{i}") for i in range(1, n_ports + 1))
    return TouchstoneData(
        freq_hz=freq_hz, s=s, ports=ports, reference_ohms=ref_ohms, parameter=param
    )


def write_touchstone(
    path,
    wl_um,
    s,
    ports: tuple[str, ...] | None = None,
    fmt: str = "ri",
    freq_unit: str = "ghz",
    reference_ohms: float = 50.0,
) -> None:
    """Write a complex S-matrix sweep to a Touchstone .sNp file.

    Parameters
    ----------
    path : output file path; the extension should be ``.sNp`` matching
        ``s.shape[-1]`` (not enforced, but :func:`read_touchstone` infers
        port count from it).
    wl_um : 1-D wavelengths in um (converted to the file's frequency unit
        via ``f_hz = C_UM_HZ / wl_um``; the file is written in the same
        order as given — sort first if a monotonic file is required).
    s : complex array, shape ``(len(wl_um), n, n)``, ``S[out, in]``
        (matches :mod:`etalon.circuit`'s model protocol).
    ports : port names for a ``! Port[k] = name`` comment header; omit for
        no port-name comments (readers default to ``p1..pn``).
    fmt : ``"ri"`` (default, exact round-trip), ``"ma"``, or ``"db"``
        (field dB, per the Touchstone convention — see module docstring).
    freq_unit : one of ``"hz"``, ``"khz"``, ``"mhz"``, ``"ghz"``.
    reference_ohms : written as the option line's ``R`` token (informational
        only — see the module docstring on reference-impedance handling).
    """
    wl = np.atleast_1d(np.asarray(wl_um, dtype=float))
    sarr = np.asarray(s, dtype=complex)
    if sarr.ndim != 3 or sarr.shape[0] != wl.size or sarr.shape[1] != sarr.shape[2]:
        raise ValueError(
            f"s must have shape (len(wl_um), n, n); got {sarr.shape} for "
            f"{wl.size} wavelengths"
        )
    n = sarr.shape[1]
    if freq_unit not in _FREQ_UNIT_TO_HZ:
        raise ValueError(f"unknown freq_unit {freq_unit!r}")
    if fmt not in ("ri", "ma", "db"):
        raise ValueError(f"unknown fmt {fmt!r} (want 'ri', 'ma', or 'db')")
    if ports is not None and len(ports) != n:
        raise ValueError(f"ports must have length {n}, got {len(ports)}")

    freq = (C_UM_HZ / wl) / _FREQ_UNIT_TO_HZ[freq_unit]
    lines = ["! Generated by etalon.touchstone.write_touchstone"]
    if ports is not None:
        for i, name in enumerate(ports, start=1):
            lines.append(f"! Port[{i}] = {name}")
    lines.append(f"# {freq_unit.upper()} S {fmt.upper()} R {reference_ohms:g}")

    def _pair(z: complex) -> tuple[float, float]:
        if fmt == "ri":
            return z.real, z.imag
        mag, ang = abs(z), np.rad2deg(np.angle(z))
        return (_linear_to_field_db(np.array([mag]))[0] if fmt == "db" else mag, ang)

    for k in range(wl.size):
        if n == 2:
            order = [(0, 0), (1, 0), (0, 1), (1, 1)]  # S11, S21, S12, S22
        else:
            order = [(i, j) for i in range(n) for j in range(n)]
        vals = []
        for i, j in order:
            a, b = _pair(sarr[k, i, j])
            vals.extend([a, b])
        row = " ".join(f"{v:.10g}" for v in vals)
        lines.append(f"{freq[k]:.10g} {row}")

    Path(path).write_text("\n".join(lines) + "\n")
