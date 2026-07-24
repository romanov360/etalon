"""Tests for siphon.circuit against analytic S-matrix results.

siphon.components does not exist yet, so minimal stub models implementing the
ports/s_params protocol are defined here.
"""

import numpy as np
import pytest

from siphon.circuit import Circuit

WL = np.linspace(1.53, 1.565, 11)  # um


class Phase:
    """Fixed 2-port element: b_out = amp * exp(1j*phi) * a_in, reciprocal."""

    ports = ("in", "out")

    def __init__(self, phi=0.0, amp=1.0):
        self.phi = phi
        self.amp = amp

    def s_params(self, wl):
        t = self.amp * np.exp(1j * self.phi)
        s = np.array([[0.0, t], [t, 0.0]], dtype=complex)
        return np.broadcast_to(s, (len(wl), 2, 2)).copy()


class Arm:
    """Dispersive 2-port arm: t(wl) = amp * exp(1j * 2*pi*neff*length_um / wl)."""

    ports = ("in", "out")

    def __init__(self, length_um, neff=2.4, amp=1.0):
        self.length_um = length_um
        self.neff = neff
        self.amp = amp

    def t(self, wl):
        return self.amp * np.exp(2j * np.pi * self.neff * self.length_um / np.asarray(wl))

    def s_params(self, wl):
        s = np.zeros((len(wl), 2, 2), dtype=complex)
        s[:, 0, 1] = s[:, 1, 0] = self.t(wl)
        return s


class Coupler:
    """Ideal lossless directional coupler, ports (in1, in2, out1, out2).

    Power cross-coupling kappa: through r = sqrt(1-kappa), cross 1j*sqrt(kappa).
    """

    ports = ("in1", "in2", "out1", "out2")

    def __init__(self, kappa=0.5):
        self.r = np.sqrt(1.0 - kappa)
        self.k = np.sqrt(kappa)

    def s_params(self, wl):
        s = np.zeros((4, 4), dtype=complex)
        s[0, 2] = s[2, 0] = self.r
        s[1, 3] = s[3, 1] = self.r
        s[0, 3] = s[3, 0] = 1j * self.k
        s[1, 2] = s[2, 1] = 1j * self.k
        return np.broadcast_to(s, (len(wl), 4, 4)).copy()


def coupler_2x2(kappa):
    """Analytic 2x2 transfer block [[r, ik], [ik, r]] of the ideal coupler."""
    r, k = np.sqrt(1.0 - kappa), np.sqrt(kappa)
    return np.array([[r, 1j * k], [1j * k, r]])


def test_stub_coupler_unitary():
    s = Coupler(kappa=0.3).s_params(WL)
    ident = np.eye(4)
    for sk in s:
        np.testing.assert_allclose(sk @ sk.conj().T, ident, atol=1e-12)


def test_cascade_two_phase_elements():
    p1 = Phase(phi=0.7, amp=0.9)
    p2 = Phase(phi=-1.3, amp=0.8)
    c = Circuit()
    c.add("p1", p1)
    c.add("p2", p2)
    c.connect(("p1", "out"), ("p2", "in"))
    c.expose("in", ("p1", "in"))
    c.expose("out", ("p2", "out"))

    assert c.external_ports == ("in", "out")
    expected = 0.9 * 0.8 * np.exp(1j * (0.7 - 1.3))
    t = c.transmission(WL, "in", "out")
    np.testing.assert_allclose(t, expected, atol=1e-12)
    # matched elements have no reflections, cascade must not create any
    s = c.s_params(WL)
    np.testing.assert_allclose(s[:, 0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(s[:, 1, 1], 0.0, atol=1e-12)
    # power dB
    np.testing.assert_allclose(
        c.transmission_db(WL, "in", "out"), 10 * np.log10((0.9 * 0.8) ** 2), atol=1e-10
    )


def test_mzi_matches_analytic():
    kappa = 0.5
    arm1 = Arm(length_um=100.0)
    arm2 = Arm(length_um=130.0)
    c = Circuit()
    c.add("c1", Coupler(kappa))
    c.add("c2", Coupler(kappa))
    c.add("a1", arm1)
    c.add("a2", arm2)
    c.connect(("c1", "out1"), ("a1", "in"))
    c.connect(("a1", "out"), ("c2", "in1"))
    c.connect(("c1", "out2"), ("a2", "in"))
    c.connect(("a2", "out"), ("c2", "in2"))
    c.expose("in1", ("c1", "in1"))
    c.expose("in2", ("c1", "in2"))
    c.expose("out1", ("c2", "out1"))
    c.expose("out2", ("c2", "out2"))

    s = c.s_params(WL)
    cp = coupler_2x2(kappa)
    t1, t2 = arm1.t(WL), arm2.t(WL)
    for k in range(len(WL)):
        expected = cp @ np.diag([t1[k], t2[k]]) @ cp
        # rows out1/out2 (indices 2,3), columns in1/in2 (indices 0,1)
        np.testing.assert_allclose(s[k, 2:4, 0:2], expected, atol=1e-12)
        # lossless network: assembled S is unitary
        np.testing.assert_allclose(s[k] @ s[k].conj().T, np.eye(4), atol=1e-12)
    # energy conservation: bar + cross power == 1
    p = np.abs(s[:, 2, 0]) ** 2 + np.abs(s[:, 3, 0]) ** 2
    np.testing.assert_allclose(p, 1.0, atol=1e-12)


def test_all_pass_ring_matches_closed_form():
    kappa = 0.2
    a = 0.9  # round-trip amplitude
    loop = Arm(length_um=31.4159, amp=a)
    c = Circuit()
    c.add("dc", Coupler(kappa))
    c.add("loop", loop)
    c.connect(("dc", "out2"), ("loop", "in"))
    c.connect(("loop", "out"), ("dc", "in2"))
    c.expose("in", ("dc", "in1"))
    c.expose("out", ("dc", "out1"))

    t = c.transmission(WL, "in", "out")
    # b_out1 = r a_in + ik a_in2; a_in2 = L b_out2; b_out2 = ik a_in + r a_in2
    # => t = (r - L) / (1 - r L) with L = loop transmission (Bogaerts 2012).
    r = np.sqrt(1.0 - kappa)
    L = loop.t(WL)
    expected = (r - L) / (1.0 - r * L)
    np.testing.assert_allclose(t, expected, atol=1e-12)
    # closed-form on/off resonance power limits
    t_min = (r - a) / (1.0 - r * a)  # exactly on resonance (phi = 0 mod 2pi)
    t_max = (r + a) / (1.0 + r * a)  # anti-resonance
    power = np.abs(t) ** 2
    assert np.all(power >= t_min**2 - 1e-12)
    assert np.all(power <= t_max**2 + 1e-12)


def test_reciprocity_of_assembled_s():
    c = Circuit()
    c.add("dc", Coupler(kappa=0.35))
    c.add("loop", Arm(length_um=42.0, amp=0.8))
    c.add("lead", Phase(phi=0.3, amp=0.95))
    c.connect(("dc", "out2"), ("loop", "in"))
    c.connect(("loop", "out"), ("dc", "in2"))
    c.connect(("dc", "out1"), ("lead", "in"))
    c.expose("in", ("dc", "in1"))
    c.expose("out", ("lead", "out"))

    s = c.s_params(WL)
    for sk in s:
        np.testing.assert_allclose(sk, sk.T, atol=1e-12)


def test_dangling_port_raises():
    c = Circuit()
    c.add("dc", Coupler())
    c.expose("in", ("dc", "in1"))
    c.expose("out", ("dc", "out1"))
    with pytest.raises(ValueError, match="in2"):
        c.s_params(WL)


def test_double_connect_raises():
    c = Circuit()
    c.add("p1", Phase())
    c.add("p2", Phase())
    c.add("p3", Phase())
    c.connect(("p1", "out"), ("p2", "in"))
    with pytest.raises(ValueError, match="already connected or exposed"):
        c.connect(("p1", "out"), ("p3", "in"))
    with pytest.raises(ValueError, match="already connected or exposed"):
        c.expose("x", ("p2", "in"))
    c.expose("mid", ("p2", "out"))
    with pytest.raises(ValueError, match="already connected or exposed"):
        c.connect(("p2", "out"), ("p3", "in"))


def test_unknown_ports_and_names_raise():
    c = Circuit()
    c.add("p1", Phase())
    with pytest.raises(ValueError, match="already added"):
        c.add("p1", Phase())
    with pytest.raises(KeyError):
        c.connect(("p1", "out"), ("nope", "in"))
    with pytest.raises(KeyError):
        c.expose("x", ("p1", "nope"))
    with pytest.raises(ValueError, match="itself"):
        c.connect(("p1", "in"), ("p1", "in"))


def test_singular_resonant_loop_raises():
    # gain element with amp = 2 makes 1 - r*L = 0 for r = 0.5 at phi = 0
    c = Circuit()
    c.add("dc", Coupler(kappa=0.75))  # r = 0.5
    c.add("loop", Phase(phi=0.0, amp=2.0))
    c.connect(("dc", "out2"), ("loop", "in"))
    c.connect(("loop", "out"), ("dc", "in2"))
    c.expose("in", ("dc", "in1"))
    c.expose("out", ("dc", "out1"))
    with pytest.raises(np.linalg.LinAlgError, match="wavelength index 0"):
        c.s_params(np.array([1.55]))
