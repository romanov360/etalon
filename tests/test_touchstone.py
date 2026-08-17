"""Tests for siphon.touchstone (Touchstone .sNp file I/O)."""

import numpy as np
import pytest

from siphon import touchstone as ts

C_UM_HZ = 2.99792458e14


def _write(tmp_path, name: str, content: str):
    path = tmp_path / name
    path.write_text(content)
    return path


class TestReadOptionLineAndFormats:
    def test_ri_format_ghz(self, tmp_path):
        # 2-port, RI format, GHz — S11=1+0j, S21=0.1-0.2j, S12=0.1-0.2j, S22=0.9+0j
        content = "# GHz S RI R 50\n193.5 1.0 0.0 0.1 -0.2 0.1 -0.2 0.9 0.0\n"
        path = _write(tmp_path, "a.s2p", content)
        data = ts.read_touchstone(path)
        assert data.freq_hz[0] == pytest.approx(193.5e9)
        assert data.reference_ohms == pytest.approx(50.0)
        assert data.s[0, 0, 0] == pytest.approx(1.0 + 0j)
        assert data.s[0, 1, 0] == pytest.approx(0.1 - 0.2j)  # S21
        assert data.s[0, 0, 1] == pytest.approx(0.1 - 0.2j)  # S12
        assert data.s[0, 1, 1] == pytest.approx(0.9 + 0j)

    def test_ma_format(self, tmp_path):
        content = "# MHz S MA R 50\n1000.0 1.0 0.0 0.5 90.0 0.5 -90.0 0.8 180.0\n"
        path = _write(tmp_path, "b.s2p", content)
        data = ts.read_touchstone(path)
        assert data.s[0, 0, 0] == pytest.approx(1.0 + 0j, abs=1e-9)
        assert data.s[0, 1, 0] == pytest.approx(0.5j, abs=1e-9)  # 0.5 @ 90 deg
        assert data.s[0, 1, 1] == pytest.approx(-0.8 + 0j, abs=1e-9)  # 0.8 @ 180 deg

    def test_db_format_is_field_db(self, tmp_path):
        # 20*log10(0.5) = -6.0206 dB
        content = "# GHz S DB R 50\n200.0 -6.0206 0.0 -100 0 -100 0 -6.0206 0.0\n"
        path = _write(tmp_path, "c.s2p", content)
        data = ts.read_touchstone(path)
        assert abs(data.s[0, 0, 0]) == pytest.approx(0.5, rel=1e-3)
        assert abs(data.s[0, 1, 1]) == pytest.approx(0.5, rel=1e-3)

    def test_default_options_when_omitted(self, tmp_path):
        # Bare "#" -> defaults ghz/s/ma/r/50.
        content = "# \n200.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0\n"
        path = _write(tmp_path, "d.s2p", content)
        data = ts.read_touchstone(path)
        assert data.freq_hz[0] == pytest.approx(200e9)
        assert data.parameter == "s"

    @pytest.mark.parametrize(
        "unit,factor", [("hz", 1.0), ("khz", 1e3), ("mhz", 1e6), ("ghz", 1e9)]
    )
    def test_frequency_units(self, tmp_path, unit, factor):
        content = f"# {unit} S RI R 50\n5.0 1 0 0 0 0 0 1 0\n"
        path = _write(tmp_path, f"e_{unit}.s2p", content)
        data = ts.read_touchstone(path)
        assert data.freq_hz[0] == pytest.approx(5.0 * factor)

    def test_comment_lines_ignored(self, tmp_path):
        content = "! this is a comment\n# GHz S RI R 50\n! another\n200.0 1 0 0 0 0 0 1 0\n"
        path = _write(tmp_path, "f.s2p", content)
        data = ts.read_touchstone(path)
        assert data.freq_hz.size == 1

    def test_port_names_from_comments(self, tmp_path):
        content = (
            "! Port[1] = laser_in\n"
            "! Port[2] = detector_out\n"
            "# GHz S RI R 50\n200.0 1 0 0 0 0 0 1 0\n"
        )
        path = _write(tmp_path, "g.s2p", content)
        data = ts.read_touchstone(path)
        assert data.ports == ("laser_in", "detector_out")

    def test_default_port_names(self, tmp_path):
        content = "# GHz S RI R 50\n200.0 1 0 0 0 0 0 1 0\n"
        path = _write(tmp_path, "h.s2p", content)
        data = ts.read_touchstone(path)
        assert data.ports == ("p1", "p2")


class TestOnePortAndThreePort:
    def test_one_port(self, tmp_path):
        content = "# GHz S RI R 50\n200.0 0.5 0.1\n201.0 0.4 0.2\n"
        path = _write(tmp_path, "one.s1p", content)
        data = ts.read_touchstone(path)
        assert data.s.shape == (2, 1, 1)
        assert data.s[0, 0, 0] == pytest.approx(0.5 + 0.1j)

    def test_three_port_row_major(self, tmp_path):
        # 3-port: file order is row-major S11 S12 S13 S21 S22 S23 S31 S32 S33.
        vals = list(range(9))  # S[i,j] = i*3+j (real part), imag=0
        row = " ".join(f"{v}.0 0.0" for v in vals)
        content = f"# GHz S RI R 50\n200.0 {row}\n"
        path = _write(tmp_path, "three.s3p", content)
        data = ts.read_touchstone(path)
        assert data.s.shape == (1, 3, 3)
        for i in range(3):
            for j in range(3):
                assert data.s[0, i, j] == pytest.approx(i * 3 + j)


class TestErrors:
    def test_missing_option_line(self, tmp_path):
        path = _write(tmp_path, "bad.s2p", "200.0 1 0 0 0 0 0 1 0\n")
        with pytest.raises(ValueError, match="option line"):
            ts.read_touchstone(path)

    def test_unknown_port_count(self, tmp_path):
        path = _write(tmp_path, "bad.txt", "# GHz S RI R 50\n200.0 1 0\n")
        with pytest.raises(ValueError, match="port count"):
            ts.read_touchstone(path)

    def test_wrong_column_count(self, tmp_path):
        # 2-port needs 9 values/row (1 + 2*4); give only 5.
        path = _write(tmp_path, "bad.s2p", "# GHz S RI R 50\n200.0 1 0 0 0\n")
        with pytest.raises(ValueError, match="multiple of"):
            ts.read_touchstone(path)

    def test_unknown_frequency_unit(self, tmp_path):
        path = _write(tmp_path, "bad.s2p", "# THz S RI R 50\n200.0 1 0 0 0 0 0 1 0\n")
        with pytest.raises(ValueError, match="frequency unit"):
            ts.read_touchstone(path)

    def test_unknown_format(self, tmp_path):
        path = _write(tmp_path, "bad.s2p", "# GHz S XY R 50\n200.0 1 0 0 0 0 0 1 0\n")
        with pytest.raises(ValueError, match="format"):
            ts.read_touchstone(path)

    def test_unsupported_v2_keyword(self, tmp_path):
        content = "[Network Data]\n# GHz S RI R 50\n200.0 1 0 0 0 0 0 1 0\n"
        path = _write(tmp_path, "bad.ts", content)
        with pytest.raises(NotImplementedError, match="network data"):
            ts.read_touchstone(path)

    def test_number_of_ports_keyword_overrides_extension(self, tmp_path):
        # No .sNp extension at all; explicit keyword sets it.
        content = "[Number of Ports] 2\n# GHz S RI R 50\n200.0 1 0 0 0 0 0 1 0\n"
        path = _write(tmp_path, "custom.ts", content)
        data = ts.read_touchstone(path)
        assert data.s.shape == (1, 2, 2)

    def test_non_numeric_data(self, tmp_path):
        path = _write(tmp_path, "bad.s2p", "# GHz S RI R 50\n200.0 1 0 0 0 0 0 abc 0\n")
        with pytest.raises(ValueError, match="non-numeric"):
            ts.read_touchstone(path)

    def test_disagreeing_second_option_line_raises(self, tmp_path):
        content = (
            "# GHz S RI R 50\n200.0 1 0 0 0 0 0 1 0\n"
            "# MHz S RI R 50\n200.0 1 0 0 0 0 0 1 0\n"
        )
        path = _write(tmp_path, "bad.s2p", content)
        with pytest.raises(ValueError, match="disagrees"):
            ts.read_touchstone(path)

    def test_identical_second_option_line_ignored(self, tmp_path):
        # Spec: additional option lines are ignored IF they agree.
        content = (
            "# GHz S RI R 50\n200.0 1 0 0 0 0 0 1 0\n"
            "# GHz S RI R 50\n201.0 1 0 0 0 0 0 1 0\n"
        )
        path = _write(tmp_path, "ok.s2p", content)
        data = ts.read_touchstone(path)
        assert data.freq_hz.size == 2
        assert np.allclose(data.freq_hz, [200e9, 201e9])

    def test_trailing_inline_comment_on_data_line(self, tmp_path):
        content = "# GHz S RI R 50\n200.0 1 0 0 0 0 0 1 0 ! probe run 3\n"
        path = _write(tmp_path, "ok.s2p", content)
        data = ts.read_touchstone(path)
        assert data.freq_hz[0] == pytest.approx(200e9)
        assert data.s[0, 0, 0] == pytest.approx(1.0 + 0j)

    def test_non_numeric_reference_impedance_raises_with_context(self, tmp_path):
        path = _write(tmp_path, "bad.s2p", "# GHz S RI R fifty\n200.0 1 0 0 0 0 0 1 0\n")
        with pytest.raises(ValueError, match=r"bad\.s2p.*R token"):
            ts.read_touchstone(path)

    def test_non_numeric_number_of_ports_raises_with_context(self, tmp_path):
        content = "[Number of Ports] two\n# GHz S RI R 50\n200.0 1 0 0 0 0 0 1 0\n"
        path = _write(tmp_path, "bad.ts", content)
        with pytest.raises(ValueError, match=r"bad\.ts.*Number of Ports"):
            ts.read_touchstone(path)


class TestTouchstoneDataModel:
    def _make(self):
        freq = np.array([190e12, 195e12, 200e12])  # Hz
        s = np.zeros((3, 2, 2), dtype=complex)
        s[:, 0, 0] = [0.1, 0.15, 0.1]
        s[:, 1, 0] = [0.9, 0.85, 0.8]
        s[:, 0, 1] = [0.9, 0.85, 0.8]
        s[:, 1, 1] = [0.1, 0.15, 0.1]
        return ts.TouchstoneData(freq_hz=freq, s=s, ports=("in", "out"))

    def test_wl_um_matches_c_over_f(self):
        data = self._make()
        assert np.allclose(data.wl_um, C_UM_HZ / data.freq_hz)

    def test_s_params_interpolates(self):
        data = self._make()
        wl_mid = C_UM_HZ / 192.5e12
        s_mid = data.s_params(np.array([wl_mid]))
        assert s_mid.shape == (1, 2, 2)
        # between the 190 and 195 THz rows
        assert 0.1 < abs(s_mid[0, 1, 0]) < 0.9

    def test_s_params_out_of_range_raises(self):
        data = self._make()
        with pytest.raises(ValueError, match="outside the file's span"):
            data.s_params(np.array([C_UM_HZ / 100e12]))

    def test_transmission_matches_s_params(self):
        data = self._make()
        wl = np.array([C_UM_HZ / 195e12])
        t = data.transmission(wl, "in", "out")
        assert t[0] == pytest.approx(data.s_params(wl)[0, 1, 0])

    def test_transmission_unknown_port_raises(self):
        data = self._make()
        with pytest.raises(ValueError, match="unknown port"):
            data.transmission(np.array([C_UM_HZ / 195e12]), "nope", "out")

    def test_transmission_db_is_power_db(self):
        data = self._make()
        wl = np.array([C_UM_HZ / 195e12])
        t = data.transmission(wl, "in", "out")
        db = data.transmission_db(wl, "in", "out")
        assert db[0] == pytest.approx(10.0 * np.log10(np.abs(t[0]) ** 2))

    def test_rejects_mismatched_shape(self):
        with pytest.raises(ValueError):
            ts.TouchstoneData(
                freq_hz=np.array([1e14, 2e14]), s=np.zeros((2, 3, 3)), ports=("a", "b")
            )

    def test_rejects_duplicate_port_names(self):
        with pytest.raises(ValueError, match="unique"):
            ts.TouchstoneData(
                freq_hz=np.array([1e14]),
                s=np.zeros((1, 2, 2)),
                ports=("a", "a"),
            )

    def test_rejects_nonpositive_freq(self):
        with pytest.raises(ValueError):
            ts.TouchstoneData(
                freq_hz=np.array([0.0]), s=np.zeros((1, 1, 1)), ports=("a",)
            )

    def test_descending_frequency_order_interpolates_correctly(self):
        # freq_hz descending -> wl_um ascending; s_params must still sort
        # internally and interpolate correctly regardless of file order.
        freq = np.array([200e12, 195e12, 190e12])  # descending
        s = np.zeros((3, 1, 1), dtype=complex)
        s[:, 0, 0] = [0.9, 0.6, 0.3]  # linear in freq: 0.9@200, 0.3@190
        data = ts.TouchstoneData(freq_hz=freq, s=s, ports=("a",))
        wl_195 = ts.C_UM_HZ / 195e12
        mid = data.s_params(np.array([wl_195]))
        assert mid[0, 0, 0] == pytest.approx(0.6, abs=1e-9)
        # a point 3/4 of the way from 190 to 200 THz -> 0.3 + 0.75*(0.9-0.3)
        f_34 = 190e12 + 0.75 * (200e12 - 190e12)
        val_34 = data.s_params(np.array([ts.C_UM_HZ / f_34]))[0, 0, 0]
        assert val_34 == pytest.approx(0.3 + 0.75 * (0.9 - 0.3), abs=1e-9)


class TestWriteReadRoundTrip:
    @pytest.mark.parametrize("fmt", ["ri", "ma", "db"])
    def test_round_trip_two_port(self, tmp_path, fmt):
        wl = np.linspace(1.30, 1.32, 5)
        rng = np.random.default_rng(1)
        s = (rng.uniform(0.05, 0.95, (5, 2, 2))
             * np.exp(1j * rng.uniform(-np.pi, np.pi, (5, 2, 2))))
        path = tmp_path / "roundtrip.s2p"
        ts.write_touchstone(path, wl, s, ports=("in", "out"), fmt=fmt)
        data = ts.read_touchstone(path)
        assert data.ports == ("in", "out")
        recovered = data.s_params(wl)
        rtol = 1e-6 if fmt == "ri" else 2e-3  # MA/DB round through deg + log10
        assert np.allclose(recovered, s, rtol=rtol, atol=1e-6)

    def test_round_trip_three_port(self, tmp_path):
        wl = np.linspace(1.30, 1.32, 4)
        rng = np.random.default_rng(2)
        s = (rng.uniform(0.05, 0.95, (4, 3, 3))
             * np.exp(1j * rng.uniform(-np.pi, np.pi, (4, 3, 3))))
        path = tmp_path / "roundtrip.s3p"
        ts.write_touchstone(path, wl, s, fmt="ri")
        data = ts.read_touchstone(path)
        assert np.allclose(data.s_params(wl), s, atol=1e-6)

    def test_write_rejects_bad_shape(self, tmp_path):
        with pytest.raises(ValueError):
            ts.write_touchstone(
                tmp_path / "bad.s2p", np.array([1.3, 1.31]), np.zeros((2, 2, 3))
            )

    def test_write_rejects_port_count_mismatch(self, tmp_path):
        with pytest.raises(ValueError):
            ts.write_touchstone(
                tmp_path / "bad.s2p",
                np.array([1.3]),
                np.zeros((1, 2, 2), dtype=complex),
                ports=("only_one",),
            )


class TestIntegrationWithExtractAndCircuit:
    def test_usable_as_extract_measured_source(self, tmp_path):
        """A TouchstoneData quacks like a component model for siphon.extract."""
        from siphon.extract import fit_transmission
        from siphon.components import RingAllPass

        wl = np.linspace(1.545, 1.555, 401)
        truth = RingAllPass(
            circumference_um=40.0, neff0=2.4, ng=4.2, kappa_power=0.15,
            loss_db_per_cm=3.0, wl0_um=1.55,
        )
        s = truth.s_params(wl)  # (n_wl, 2, 2), ports ("in", "out")
        path = tmp_path / "measured_ring.s2p"
        ts.write_touchstone(path, wl, s, ports=("in", "out"), fmt="ri")

        data = ts.read_touchstone(path)
        measured_db = data.transmission_db(wl, "in", "out")

        result = fit_transmission(
            lambda **p: RingAllPass(
                circumference_um=40.0, wl0_um=1.55,
                neff0=p["neff0"], ng=p["ng"],
                kappa_power=p["kappa_power"], loss_db_per_cm=p["loss_db_per_cm"],
            ),
            params0={"neff0": 2.41, "ng": 4.15, "kappa_power": 0.12, "loss_db_per_cm": 2.5},
            wl_um=wl,
            measured=measured_db,
            inport="in",
            outport="out",
            bounds={
                "neff0": (2.0, 3.0), "ng": (3.0, 5.0),
                "kappa_power": (1e-3, 0.9), "loss_db_per_cm": (0.0, 50.0),
            },
        )
        assert result.success
        assert result.params["kappa_power"] == pytest.approx(0.15, abs=0.01)

    def test_usable_in_circuit(self):
        """TouchstoneData exposes ports + s_params like any Circuit-eligible model."""
        from siphon.circuit import Circuit

        freq = np.array([190e12, 195e12, 200e12])
        s = np.zeros((3, 2, 2), dtype=complex)
        s[:, 1, 0] = 0.9
        s[:, 0, 1] = 0.9
        data = ts.TouchstoneData(freq_hz=freq, s=s, ports=("in", "out"))

        c = Circuit()
        c.add("meas", data)
        c.expose("a", ("meas", "in"))
        c.expose("b", ("meas", "out"))
        wl = np.array([C_UM_HZ / 195e12])
        t = c.transmission(wl, "a", "b")
        assert t[0] == pytest.approx(0.9)

    def test_two_touchstone_instances_cascaded_via_connect(self):
        """Two measured components chained through Circuit.connect()."""
        from siphon.circuit import Circuit

        freq = np.array([190e12, 195e12, 200e12])

        def make(t: float) -> ts.TouchstoneData:
            s = np.zeros((3, 2, 2), dtype=complex)
            s[:, 1, 0] = t
            s[:, 0, 1] = t
            return ts.TouchstoneData(freq_hz=freq, s=s, ports=("in", "out"))

        c = Circuit()
        c.add("stage1", make(0.9))
        c.add("stage2", make(0.8))
        c.connect(("stage1", "out"), ("stage2", "in"))
        c.expose("a", ("stage1", "in"))
        c.expose("b", ("stage2", "out"))
        wl = np.array([C_UM_HZ / 195e12])
        t = c.transmission(wl, "a", "b")
        assert t[0] == pytest.approx(0.9 * 0.8, abs=1e-9)
