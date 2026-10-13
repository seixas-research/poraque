# -*- coding: utf-8 -*-
# file: test_codec.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
The C text codec of volumetric blocks must agree with the Python path exactly.

It replaced the part of the cache build that was 86 % of the time per
material (the ``E17.11`` text in and out). A value parsed one ulp off, or a
field written one column wide, would change a cache silently -- or lose a
minus sign to VASP's ``1X`` -- so agreement is bit-for-bit and byte-for-byte,
not approximate. Skipped where no C compiler is available.
"""

import numpy as np
import pytest

from poraque.fields.vasp import codec
from poraque.fields.vasp.volumetric import (fortran_exponential,
                                            read_augmentation_blocks,
                                            read_volumetric)

pytestmark = pytest.mark.skipif(codec.load() is None,
                                reason="the C codec could not be built")


def python_block(values, columns, width, decimals):
    lines = ["".join(" " + fortran_exponential(v, decimals=decimals,
                                               width=width)
                     for v in values[start:start + columns])
             for start in range(0, len(values), columns)]
    return "\n".join(lines) + "\n"


@pytest.fixture
def values():
    rng = np.random.default_rng(0)
    random = rng.normal(size=20000) * 10.0 ** rng.integers(-12, 12, 20000)
    edges = [0.0, -0.0, 1.0, -1.0, 0.1, -0.1, 10.0, 9.99999999999995,
             0.0999999999999996, 1e-300, 5e-301, -2362.5518271, 1e300,
             np.nan, np.inf]
    return np.concatenate([random, edges])


class TestTheWriterMatchesFortranExponential:
    """Byte for byte, in every format the package writes."""

    @pytest.mark.parametrize("width,decimals", [(17, 11), (11, 5), (15, 7)])
    def test_the_block_is_identical(self, values, width, decimals):
        written = codec.format_block(values, 5, width, decimals)
        assert written.decode() == python_block(values, 5, width, decimals)


class TestTheReaderMatchesThePythonPath:
    """``strtod`` rounds correctly, so every value is ``float()``'s double."""

    def test_a_round_trip_is_bit_identical(self, values):
        # Python's float() cannot read Fortran's letterless three-digit
        # exponent; that case has its own test below.
        finite = values[np.isfinite(values)
                        & ((np.abs(values) > 1e-98) | (values == 0))
                        & (np.abs(values) < 1e98)]
        text = python_block(finite, 5, 17, 11).encode()
        parsed = np.empty(finite.size)
        count, end, bad = codec.parse_into(text, 0, len(text), parsed, 0)
        expected = np.array(text.split(), dtype=float)
        assert count == finite.size and not bad
        assert parsed.view(np.uint64).tolist() == \
            expected.view(np.uint64).tolist()
        assert end == len(text.rstrip())        # just past the last token

    def test_a_three_digit_exponent_is_read(self):
        """Fortran drops the letter: ``0.50000000000-299``."""
        text = b" 0.50000000000-299 -.12345678901+105\n"
        parsed = np.empty(2)
        count, _, bad = codec.parse_into(text, 0, len(text), parsed, 0)
        assert count == 2 and not bad
        assert parsed.tolist() == [5e-300, -1.2345678901e+104]

    def test_a_file_reads_the_same_with_and_without_it(self, tmp_path,
                                                       monkeypatch):
        """Both blocks of a spin file, and the records between them."""
        from poraque.fields import FieldGrid, SpinDensity
        from poraque.fields.vasp.poscar import Poscar

        grid = FieldGrid((6, 6, 8), np.eye(3) * 4.0)
        poscar = Poscar(cell=np.eye(3) * 4.0, symbols=["Fe"], counts=[1],
                        scaled_positions=[[0.0, 0.0, 0.0]])
        rng = np.random.default_rng(1)
        field = SpinDensity.from_up_down(rng.random(grid.shape) + 0.5,
                                         rng.random(grid.shape) + 0.5,
                                         grid, poscar)
        path = str(tmp_path / "CHGCAR")
        field.write(path)

        fast = read_volumetric(path, read_all=True)
        fast_records = read_augmentation_blocks(path)
        monkeypatch.setattr(codec, "load", lambda log=None: None)
        slow = read_volumetric(path, read_all=True)

        assert np.array_equal(fast[1], slow[1])
        assert len(fast[2]) == len(slow[2]) == 1
        assert np.array_equal(fast[2][0], slow[2][0])
        assert fast_records == read_augmentation_blocks(path)


class TestPoraqueCompile:
    """``poraque-compile`` builds, checks and reports; its exit code says so."""

    @staticmethod
    def _main(argv):
        import importlib.util
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                            "poraque_compile.py")
        spec = importlib.util.spec_from_file_location("poraque_compile", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(argv)

    def test_it_exits_zero_when_the_codec_loads(self):
        assert self._main(["--only", "codec", "--quiet"]) == 0

    def test_it_exits_one_when_disabled(self, monkeypatch):
        monkeypatch.setenv("PORAQUE_C_BACKEND", "0")
        codec.reset()
        try:
            assert self._main(["--only", "codec", "--quiet"]) == 1
        finally:
            monkeypatch.delenv("PORAQUE_C_BACKEND")
            codec.reset()


class TestTheCacheBuildSaysWhichCodecItUses:
    """
    A codec that failed to build used to drop the cache to the Python path
    -- several times slower -- with nothing in the log; and every spawned
    worker compiled it at once. It is loaded in the parent and reported.
    """

    def test_the_cache_block_names_the_codec(self):
        from poraque.data.cache import _report_codec

        lines = []
        _report_codec(lines.append)
        assert any(line.startswith("  text codec: C (") for line in lines)

    def test_a_disabled_codec_is_reported_as_python(self, monkeypatch):
        from poraque.data.cache import _report_codec

        monkeypatch.setattr(codec, "load", lambda log=None: None)
        monkeypatch.setattr(codec, "available", lambda: False)
        lines = []
        _report_codec(lines.append)
        assert any("text codec: Python" in line for line in lines)


class TestEachWorkerGetsItsShareOfTheCores:
    """
    V_ext's structure factor is a BLAS product, and an unconfigured BLAS in
    each of N spawned workers starts one thread per core: N x cores threads.
    """

    def test_the_share_is_set_while_the_pool_starts_and_then_removed(
            self, monkeypatch):
        import os

        from poraque.data.cache import BLAS_THREAD_VARIABLES, _worker_threads

        for name in BLAS_THREAD_VARIABLES:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        with _worker_threads(4) as share:
            assert share == "2"
            assert all(os.environ[name] == "2"
                       for name in BLAS_THREAD_VARIABLES)
        assert not any(name in os.environ for name in BLAS_THREAD_VARIABLES)

    def test_a_value_the_user_set_is_respected(self, monkeypatch):
        import os

        from poraque.data.cache import _worker_threads

        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        with _worker_threads(2):
            assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["OMP_NUM_THREADS"] == "1"


def _spin_file(tmp_path, name="CHGCAR"):
    """A small spin-polarized CHGCAR with augmentation records, and its path."""
    from poraque.fields import FieldGrid, SpinDensity
    from poraque.fields.vasp.poscar import Poscar

    grid = FieldGrid((6, 6, 8), np.eye(3) * 4.0)
    poscar = Poscar(cell=np.eye(3) * 4.0, symbols=["Fe"], counts=[1],
                    scaled_positions=[[0.0, 0.0, 0.0]])
    rng = np.random.default_rng(2)
    field = SpinDensity.from_up_down(rng.random(grid.shape) + 0.5,
                                     rng.random(grid.shape) + 0.5,
                                     grid, poscar)
    path = tmp_path / name
    field.write(str(path), augmentation=[
        "augmentation occupancies   1  4", "  0.1 0.2 0.3 0.4",
        "augmentation occupancies   1  4", "  0.5 0.6 0.7 0.8"])
    return path


def _python_read(path, monkeypatch):
    """The Python path's answer, with the codec switched off for one call."""
    with monkeypatch.context() as patch:
        patch.setattr(codec, "load", lambda log=None: None)
        return (read_volumetric(str(path), read_all=True),
                read_augmentation_blocks(str(path)))


def _same(fast, slow):
    (s1, d1, e1), r1 = fast
    (s2, d2, e2), r2 = slow
    assert d1.view(np.uint64).tolist() == d2.view(np.uint64).tolist()
    assert len(e1) == len(e2) == 1
    assert e1[0].view(np.uint64).tolist() == e2[0].view(np.uint64).tolist()
    assert r1 == r2


class TestTheStreamingReaderAgreesEverywhere:
    """
    The C reader streams the file in windows (a whole-file buffer was ~870 MB
    for a 288³ spin CHGCAR) and must agree with the Python path on every
    container and line ending, and across window boundaries.
    """

    def test_a_plain_file(self, tmp_path, monkeypatch):
        path = _spin_file(tmp_path)
        _same((read_volumetric(str(path), read_all=True),
               read_augmentation_blocks(str(path))),
              _python_read(path, monkeypatch))

    def test_gzip_and_zip(self, tmp_path, monkeypatch):
        import gzip
        import zipfile

        plain = _spin_file(tmp_path)
        gz = tmp_path / "CHGCAR.gz"
        gz.write_bytes(gzip.compress(plain.read_bytes()))
        zp = tmp_path / "CHGCAR.zip"
        with zipfile.ZipFile(zp, "w") as archive:
            archive.write(plain, "CHGCAR")
        for path in (gz, zp):
            _same((read_volumetric(str(path), read_all=True),
                   read_augmentation_blocks(str(path))),
                  _python_read(path, monkeypatch))

    def test_crlf_line_endings(self, tmp_path, monkeypatch):
        path = _spin_file(tmp_path)
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        _same((read_volumetric(str(path), read_all=True),
               read_augmentation_blocks(str(path))),
              _python_read(path, monkeypatch))

    def test_windows_that_split_tokens_and_lines(self, tmp_path, monkeypatch):
        from poraque.fields.vasp import volumetric

        path = _spin_file(tmp_path)
        slow = _python_read(path, monkeypatch)
        for chunk in (7, 64, 91, 1000):
            monkeypatch.setattr(volumetric, "STREAM_CHUNK", chunk)
            _same((read_volumetric(str(path), read_all=True),
                   read_augmentation_blocks(str(path))), slow)

    def test_a_truncated_file_raises(self, tmp_path):
        path = _spin_file(tmp_path)
        text = path.read_bytes()
        path.write_bytes(text[:text.index(b"augmentation") // 2])
        with pytest.raises(ValueError, match="truncated"):
            read_volumetric(str(path))

    def test_a_field_that_is_not_a_number_is_named(self, tmp_path):
        """A Fortran overflow field is written as asterisks."""
        path = _spin_file(tmp_path)
        lines = path.read_text().splitlines(keepends=True)
        start = next(i for i, line in enumerate(lines)
                     if line.split() == ["6", "6", "8"]) + 1
        lines[start + 1] = " " + "*" * 17 + lines[start + 1][18:]
        path.write_text("".join(lines))
        with pytest.raises(ValueError, match=r"\*{17}"):
            read_volumetric(str(path))


class TestTheLocaleCannotChangeTheNumbers:
    """
    ``strtod`` and ``snprintf`` follow ``LC_NUMERIC``; the codec runs under the
    "C" locale, so a process that set a decimal comma reads and writes the
    same digits.
    """

    def test_a_comma_locale_changes_nothing(self):
        import locale

        values = np.array([0.5, -2362.5518271, 1.25e-7])
        before = codec.format_block(values, 5, 17, 11)
        previous = locale.setlocale(locale.LC_NUMERIC)
        for name in ("de_DE.UTF-8", "de_DE", "pt_BR.UTF-8", "fr_FR.UTF-8"):
            try:
                locale.setlocale(locale.LC_NUMERIC, name)
                break
            except locale.Error:
                continue
        else:
            pytest.skip("no comma-decimal locale installed")
        try:
            assert codec.format_block(values, 5, 17, 11) == before
            out = np.empty(3)
            count, _, bad = codec.parse_into(before, 0, len(before), out, 0)
            assert count == 3 and not bad
            assert out.tolist() == values.tolist()
        finally:
            locale.setlocale(locale.LC_NUMERIC, previous)


class TestTheWriterRefusesBadArguments:
    """``columns <= 0`` divided by zero in C; a huge ``decimals`` overran."""

    @pytest.mark.parametrize("columns,width,decimals",
                             [(0, 17, 11), (5, 0, 11), (5, 65, 11),
                              (5, 17, -1), (5, 17, 41)])
    def test_out_of_range_raises(self, columns, width, decimals):
        with pytest.raises(ValueError):
            codec.format_block(np.ones(3), columns, width, decimals)


class TestTheBuildMachinery:
    """:mod:`poraque.utils.cbuild`, which both kernels go through."""

    @pytest.mark.parametrize("value,expected",
                             [("0", False), ("off", False), ("no", False),
                              ("1", True)])
    def test_the_environment_switch(self, monkeypatch, value, expected):
        from poraque.utils import cbuild

        monkeypatch.setenv("PORAQUE_C_BACKEND", value)
        assert cbuild.enabled() is expected

    def test_abi_and_flags_change_the_name(self):
        from poraque.utils import cbuild

        base = cbuild.library_path(codec.SOURCE, "x", 1, [["-O3"]])
        assert cbuild.library_path(codec.SOURCE, "x", 2, [["-O3"]]) != base
        assert cbuild.library_path(codec.SOURCE, "x", 1,
                                   [["-O3", "-march=native"]]) != base
        assert cbuild.library_path(codec.SOURCE, "x", 1, [["-O3"]]) == base

    def test_only_libraries_nothing_loads_are_stale(self, tmp_path,
                                                    monkeypatch):
        from poraque.utils import cbuild

        monkeypatch.setenv("PORAQUE_CACHE_DIR", str(tmp_path))
        for name in ("poraque_codec_new.dylib", "poraque_codec_old.dylib",
                     "poraque_spectral_old.so", "unrelated.dylib"):
            (tmp_path / name).write_bytes(b"")
        current = [str(tmp_path / "poraque_codec_new.dylib")]
        stale = sorted(p.rsplit("/", 1)[-1]
                       for p in cbuild.stale_libraries(current))
        assert stale == ["poraque_codec_old.dylib", "poraque_spectral_old.so"]


class TestASpinProbeThatFailedIsNotRemembered:
    """A read error answered False for good; it is now probed again."""

    def test_the_next_run_probes_again(self, tmp_path, monkeypatch):
        import poraque.data.cache as cache

        path = str(_spin_file(tmp_path))
        monkeypatch.setattr(cache, "_spin_of", lambda p: None)
        assert cache.probe_spin([path], memo_root=str(tmp_path)) == [False]
        monkeypatch.undo()
        assert cache.probe_spin([path], memo_root=str(tmp_path)) == [True]
