"""Frequency-domain S-parameter circuit solver.

Assembles a netlist of S-parameter models into a single external S-matrix by
eliminating internal (connected) ports, the standard multiport interconnection
method (Filipsson 1981; also known as subnetwork growth). Per wavelength the
block-diagonal S of all instances is partitioned into external (e) and
internal (i) ports; with P the permutation exchanging each connected pair
(a_i = P b_i),

    S_ext = S_ee + S_ei P (I - S_ii P)^-1 S_ie.

Model protocol: a model exposes ``ports: tuple[str, ...]`` and
``s_params(wl) -> ndarray`` of shape ``(len(wl), n, n)``, complex, with
b_p = sum_q S[k, p, q] a_q at vacuum wavelength ``wl[k]`` in um. Port order in
the matrix equals the order of ``ports``.

Validity: linear, time-invariant, single-mode-per-port networks only. A
lossless feedback loop exactly on resonance makes (I - S_ii P) singular — the
physical steady state does not exist — and raises rather than returning noise.

Conventions: wavelengths in um; ``transmission_db`` returns power dB.
"""

from __future__ import annotations

import numpy as np

# Condition number of (I - S_ii P) above which the internal solve is treated
# as singular (resonant lossless loop, or a model with gain >= loop loss).
_COND_LIMIT = 1e12

Port = tuple[str, str]  # (instance name, port name)


class Circuit:
    """Netlist of S-parameter model instances solved to one external S-matrix.

    Build with :meth:`add`, wire with :meth:`connect`, name outside-world
    ports with :meth:`expose`, then call :meth:`s_params`. External port
    order in the result is the order of the ``expose`` calls.
    """

    def __init__(self) -> None:
        self._instances: dict[str, object] = {}
        self._connections: list[tuple[Port, Port]] = []
        self._exposed: dict[str, Port] = {}
        self._used_ports: set[Port] = set()

    # --- construction ------------------------------------------------------

    def add(self, name: str, model) -> None:
        """Register a model instance under a unique name."""
        if name in self._instances:
            raise ValueError(f"instance {name!r} already added")
        ports = getattr(model, "ports", None)
        if not ports or not hasattr(model, "s_params"):
            raise TypeError(
                f"model for {name!r} must expose 'ports' and 's_params(wl)'"
            )
        if len(set(ports)) != len(ports):
            raise ValueError(f"model for {name!r} has duplicate port names: {ports}")
        self._instances[name] = model

    def connect(self, a: Port, b: Port) -> None:
        """Join two instance ports. Each port may be used at most once."""
        self._resolve(a)
        self._resolve(b)
        if a == b:
            raise ValueError(f"cannot connect port {a} to itself")
        for port in (a, b):
            if port in self._used_ports:
                raise ValueError(f"port {port} is already connected or exposed")
        self._connections.append((a, b))
        self._used_ports.update((a, b))

    def expose(self, external_name: str, port: Port) -> None:
        """Make an instance port an external port named ``external_name``."""
        self._resolve(port)
        if external_name in self._exposed:
            raise ValueError(f"external port {external_name!r} already defined")
        if port in self._used_ports:
            raise ValueError(f"port {port} is already connected or exposed")
        self._exposed[external_name] = port
        self._used_ports.add(port)

    @property
    def external_ports(self) -> tuple[str, ...]:
        """External port names in expose() order (== S-matrix port order)."""
        return tuple(self._exposed)

    def _resolve(self, port: Port) -> int:
        """Global port index of (instance, port); raises if unknown."""
        inst, pname = port
        if inst not in self._instances:
            raise KeyError(f"unknown instance {inst!r}; known: {sorted(self._instances)}")
        offset = 0
        for name, model in self._instances.items():
            if name == inst:
                break
            offset += len(model.ports)
        model = self._instances[inst]
        if pname not in model.ports:
            raise KeyError(
                f"instance {inst!r} has no port {pname!r}; ports: {model.ports}"
            )
        return offset + model.ports.index(pname)

    # --- solve --------------------------------------------------------------

    def s_params(self, wl: np.ndarray) -> np.ndarray:
        """External S-matrix, shape (len(wl), n_ext, n_ext), wl in um.

        Raises ValueError if any instance port is neither connected nor
        exposed, and numpy.linalg.LinAlgError if the internal solve is
        singular at some wavelength (lossless resonant loop).
        """
        wl = np.atleast_1d(np.asarray(wl, dtype=float))
        if not self._instances:
            raise ValueError("circuit has no instances")

        dangling = [
            (name, pname)
            for name, model in self._instances.items()
            for pname in model.ports
            if (name, pname) not in self._used_ports
        ]
        if dangling:
            raise ValueError(f"unconnected and unexposed ports: {dangling}")

        n_total = sum(len(m.ports) for m in self._instances.values())
        s_block = np.zeros((len(wl), n_total, n_total), dtype=complex)
        offset = 0
        for name, model in self._instances.items():
            n = len(model.ports)
            sm = np.asarray(model.s_params(wl), dtype=complex)
            if sm.shape != (len(wl), n, n):
                raise ValueError(
                    f"instance {name!r}: s_params returned shape {sm.shape}, "
                    f"expected {(len(wl), n, n)}"
                )
            s_block[:, offset : offset + n, offset : offset + n] = sm
            offset += n

        ext = np.array([self._resolve(p) for p in self._exposed.values()], dtype=int)
        internal_ports = [p for pair in self._connections for p in pair]
        internal = np.array([self._resolve(p) for p in internal_ports], dtype=int)
        n_i = len(internal)

        # a_int = P b_int: each connected pair exchanges its waves.
        perm = np.zeros((n_i, n_i))
        for m in range(0, n_i, 2):
            perm[m, m + 1] = perm[m + 1, m] = 1.0

        # All wavelengths solved as one stacked system (matmul, cond and
        # solve broadcast over the leading axis) — same math per slice as a
        # per-wavelength loop, order of magnitude faster on dense sweeps.
        s_ee = s_block[:, ext[:, None], ext[None, :]]
        if n_i == 0:
            return s_ee
        s_ei = s_block[:, ext[:, None], internal[None, :]]
        s_ie = s_block[:, internal[:, None], ext[None, :]]
        s_ii = s_block[:, internal[:, None], internal[None, :]]
        m_int = np.eye(n_i) - s_ii @ perm
        finite = np.isfinite(m_int).reshape(len(wl), -1).all(axis=1)
        if finite.all():
            bad = np.flatnonzero(np.linalg.cond(m_int) > _COND_LIMIT)
        else:
            bad = np.flatnonzero(~finite)
        if bad.size:
            k = int(bad[0])
            raise np.linalg.LinAlgError(
                f"singular internal network at wavelength index {k} "
                f"(wl = {wl[k]} um): lossless resonant loop or gain"
            )
        return s_ee + s_ei @ perm @ np.linalg.solve(m_int, s_ie)

    # --- convenience ---------------------------------------------------------

    def transmission(self, wl: np.ndarray, inport: str, outport: str) -> np.ndarray:
        """Complex field transmission b_out / a_in vs wavelength (um)."""
        for name in (inport, outport):
            if name not in self._exposed:
                raise KeyError(
                    f"unknown external port {name!r}; known: {self.external_ports}"
                )
        names = self.external_ports
        s = self.s_params(wl)
        return s[:, names.index(outport), names.index(inport)]

    def transmission_db(self, wl: np.ndarray, inport: str, outport: str) -> np.ndarray:
        """Power transmission in dB (<= 0 for passive networks); -inf if zero."""
        t = self.transmission(wl, inport, outport)
        with np.errstate(divide="ignore"):
            return 10.0 * np.log10(np.abs(t) ** 2)
