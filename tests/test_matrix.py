"""The MAX7219 LED matrix: driven only on a Raspberry Pi, told the right bytes, and showing the right signs."""

import sys
import time
import types
from typing import ClassVar

import cv2
import numpy as np
import pytest
from synth import render

from resistor_sorter import sorter as sorter_module
from resistor_sorter.cli import main
from resistor_sorter.config import Bin, MatrixConfig, SorterConfig, load_config
from resistor_sorter.decode import decode
from resistor_sorter.matrix import (
    CROSS,
    DIGITS,
    READY,
    SMALL_DIGITS,
    Max7219,
    Panel,
    filling,
    glyph,
    matrix_unavailable,
    number,
)
from resistor_sorter.sorter import Sorter

BLANK = np.zeros((8, 8), dtype=bool)
# An asymmetric picture, to tell every way of mounting the module apart.
L_SHAPE = glyph(("X.......", "X.......", "XXX.....", "........", "........", "........", "........", "........"))


class FakeSpiDev:
    """Stands in for spidev.SpiDev and logs every transfer."""

    instances: ClassVar[list["FakeSpiDev"]] = []

    def __init__(self):
        self.opened = None
        self.max_speed_hz = None
        self.mode = None
        self.writes = []
        self.closed = False
        FakeSpiDev.instances.append(self)

    def open(self, bus, device):
        self.opened = (bus, device)

    def writebytes(self, data):
        self.writes.append(list(data))

    def close(self):
        self.closed = True

    def rows(self):
        """The eight row bytes the first chip holds after the writes so far."""
        held = [0] * 8
        for register, value in (write[-2:] for write in self.writes):
            if 1 <= register <= 8:
                held[register - 1] = value
        return held

    def pictures(self):
        """Every picture written, as eight row bytes, starting with the blank one from the setup."""
        rows = [write[-2:] for write in self.writes if 1 <= write[-2] <= 8]
        assert [register for register, _ in rows] == [i % 8 + 1 for i in range(len(rows))]  # whole, in order
        return [[value for _, value in rows[i:i + 8]] for i in range(0, len(rows), 8)]


def row_bytes(frame):
    return [int(np.packbits(row)[0]) for row in frame]


def same(frames, expected):
    return len(frames) == len(expected) and all(np.array_equal(a, b) for a, b in zip(frames, expected))


@pytest.fixture
def spidev(monkeypatch):
    """A stand-in for the spidev package, which only exists on the Pi."""
    module = types.ModuleType("spidev")
    module.SpiDev = FakeSpiDev
    monkeypatch.setitem(sys.modules, "spidev", module)
    FakeSpiDev.instances.clear()
    return module


def test_setup_wakes_the_chip_blank_and_close_puts_it_to_sleep(spidev):
    matrix = Max7219(MatrixConfig(device=1, brightness=9))
    spi = FakeSpiDev.instances[-1]
    assert spi.opened == (0, 1) and spi.max_speed_hz == 1_000_000 and spi.mode == 0
    assert spi.writes == (
        [[0x0F, 0], [0x0C, 0]]  # display test off, shut down
        + [[0x0B, 7], [0x09, 0], [0x0A, 9]]  # all eight rows, no seven-segment decoding, brightness
        + [[row, 0] for row in range(1, 9)]  # blank
        + [[0x0C, 1]]  # run
    )
    matrix.close()
    assert spi.writes[-9:] == [[row, 0] for row in range(1, 9)] + [[0x0C, 0]] and spi.closed


def test_rows_go_top_down_with_the_left_column_in_the_top_bit(spidev):
    matrix = Max7219(MatrixConfig())
    spi = FakeSpiDev.instances[-1]
    del spi.writes[:]
    matrix.show(L_SHAPE)
    assert spi.writes == [[1, 0x80], [2, 0x80], [3, 0xE0]] + [[row, 0] for row in range(4, 9)]
    with pytest.raises(ValueError, match="8x32"):
        matrix.show(np.zeros((8, 32), dtype=bool))


@pytest.mark.parametrize("rotate, mirror, rows", [
    (0, False, [0x80, 0x80, 0xE0, 0, 0, 0, 0, 0]),
    (90, False, [0x07, 0x04, 0x04, 0, 0, 0, 0, 0]),
    (180, False, [0, 0, 0, 0, 0, 0x07, 0x01, 0x01]),
    (270, False, [0, 0, 0, 0, 0, 0x20, 0x20, 0xE0]),
    (0, True, [0x01, 0x01, 0x07, 0, 0, 0, 0, 0]),
    (90, True, [0xE0, 0x20, 0x20, 0, 0, 0, 0, 0]),
])
def test_rotate_and_mirror_cover_every_way_the_module_can_be_mounted(spidev, rotate, mirror, rows):
    Max7219(MatrixConfig(rotate=rotate, mirror=mirror)).show(L_SHAPE)
    assert FakeSpiDev.instances[-1].rows() == rows


def test_a_chain_of_chips_only_lights_the_first(spidev):
    matrix = Max7219(MatrixConfig(chain=3))
    spi = FakeSpiDev.instances[-1]
    assert spi.writes[:3] == [[0x0F, 0] * 3, [0x0C, 0] * 3, [0, 0, 0, 0, 0x0B, 7]]
    del spi.writes[:]
    matrix.show(L_SHAPE)
    assert spi.writes[0] == [0, 0, 0, 0, 1, 0x80]  # no-ops pass through the chips further along


def test_missing_spidev_is_explained(monkeypatch):
    monkeypatch.setitem(sys.modules, "spidev", None)
    with pytest.raises(ImportError, match="python3-spidev"):
        Max7219(MatrixConfig())


def test_spi_problems_are_explained(spidev, monkeypatch):
    def refuse(self, bus, device):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(FakeSpiDev, "open", refuse)
    with pytest.raises(RuntimeError, match="spidev0.0.*No such file.*raspi-config"):
        Max7219(MatrixConfig())


def test_bin_numbers():
    assert np.array_equal(number(3), glyph((
        ".XXXXX..", "....X...", "...X....", "....X...", ".....X..", ".X...X..", "..XXX...", "........")))
    assert np.array_equal(number(12), glyph((
        "........", ".X..XXX.", "XX....X.", ".X..XXX.", ".X..X...", "XXX.XXX.", "........", "........")))
    assert np.array_equal(number(123), number(99))
    for font, shape in ((DIGITS, (7, 5)), (SMALL_DIGITS, (5, 3))):
        assert all(digit.shape == shape for digit in font)
        assert len({digit.tobytes() for digit in font}) == 10  # all different
    signs = [number(n) for n in range(1, 100)] + [READY, CROSS]
    assert len({sign.tobytes() for sign in signs}) == len(signs) and all(sign.any() for sign in signs)


def test_filling_rises_from_the_bottom():
    assert [int(filling(p).sum()) // 8 for p in (0, 0.1, 0.5, 0.99, 1, 3)] == [1, 1, 4, 8, 8, 8]
    assert filling(0.5)[4:].all() and not filling(0.5)[:4].any()


class FakeMatrix:
    def __init__(self):
        self.frames = []
        self.closed = False

    def show(self, frame):
        self.frames.append(frame)

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_panel(bins):
    clock = Clock()
    sorter = Sorter(SorterConfig(window=1.0, min_readings=2, reject_after=2.0, clear_after=0.5), bins, clock)
    matrix = FakeMatrix()
    panel = Panel(matrix)

    def step(present, reading=None):
        clock.now += 0.25
        panel.update(sorter, sorter.step(present, reading))

    return step, matrix, panel


READING = decode(["yellow", "violet", "red", "gold"])  # 4.7 kΩ: the third default bin
OTHER = decode(["yellow", "violet", "orange", "gold"])  # 47 kΩ: the fourth


def test_panel_follows_the_sorter():
    step, matrix, panel = make_panel(load_config(None).bins)
    assert same(matrix.frames, [READY])
    step(False)
    assert len(matrix.frames) == 1  # unchanged pictures are not sent again
    for _ in range(5):
        step(True, READING)
    # Seen: the fill rises over the window, then the bin is decided.
    assert same(matrix.frames[1:], [filling(0), filling(0.25), filling(0.5), filling(0.75), number(3)])
    step(True, READING)
    step(False)  # a short gap is not a removal
    step(True, READING)
    assert len(matrix.frames) == 6  # the number stays up while the resistor lies there
    for _ in range(3):  # gone for clear_after
        step(False)
    assert same(matrix.frames[6:], [READY])
    panel.close()
    assert matrix.closed


def test_panel_shows_a_cross_for_rejects():
    step, matrix, _ = make_panel(load_config(None).bins)
    for _ in range(8):  # never agree
        step(True, READING)
        step(True, OTHER)
    assert same(matrix.frames[-1:], [CROSS]) and not any(np.array_equal(f, CROSS) for f in matrix.frames[:-1])
    step, matrix, _ = make_panel([Bin("low", max=100)])  # a value no bin takes
    for _ in range(5):
        step(True, READING)
    assert same(matrix.frames[-1:], [CROSS])


def test_matrix_is_only_driven_on_a_pi(monkeypatch, spidev):
    assert matrix_unavailable(MatrixConfig(enabled=False)) == "disabled in the config"
    monkeypatch.setattr(sorter_module, "is_raspberry_pi", lambda: False)
    assert matrix_unavailable(MatrixConfig(enabled="auto")) == "not running on a Raspberry Pi"
    assert matrix_unavailable(MatrixConfig(enabled=True)) is None
    monkeypatch.setattr(sorter_module, "is_raspberry_pi", lambda: True)
    assert matrix_unavailable(MatrixConfig(enabled="auto")) is None
    monkeypatch.setitem(sys.modules, "spidev", None)
    assert matrix_unavailable(MatrixConfig(enabled="auto")) == "python3-spidev is not installed"


def test_config_checks(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[matrix]\nbrightness = 15\nrotate = 270\nmirror = true\nchain = 4\n")
    matrix = load_config(path).matrix
    assert (matrix.brightness, matrix.rotate, matrix.mirror, matrix.chain) == (15, 270, True, 4)
    for text, complaint in [
        ('[matrix]\nenabled = "maybe"\n', "matrix.enabled"),
        ("[matrix]\nbrightness = 16\n", "brightness"),
        ("[matrix]\nrotate = 45\n", "rotate"),
        ("[matrix]\nchain = 0\n", "chain"),
    ]:
        path.write_text(text)
        with pytest.raises(ValueError, match=complaint):
            load_config(path)


def test_matrix_command_shows_each_sign(tmp_path, monkeypatch, capsys, spidev):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(
        '[matrix]\nenabled = true\n[[bin]]\nname = "low"\nmax = 1000\n[[bin]]\nname = "high"\nmin = 1000\n')
    assert main(["matrix", "--hold", "0.5"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("matrix: on /dev/spidev0.0")
    assert out.splitlines()[1:] == ["ready", "bin 1: low", "bin 2: high", "reject"]
    assert sleeps == [0.5] * 4
    spi = FakeSpiDev.instances[-1]
    assert spi.pictures() == [row_bytes(f) for f in (BLANK, READY, number(1), number(2), CROSS, BLANK)]
    assert spi.writes[-1] == [0x0C, 0] and spi.closed


def test_matrix_command_without_a_matrix(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sorter_module, "is_raspberry_pi", lambda: False)
    monkeypatch.chdir(tmp_path)
    assert main(["matrix"]) == 1
    assert "the matrix is off (not running on a Raspberry Pi)" in capsys.readouterr().err


def test_run_shows_the_bin_on_the_matrix(tmp_path, monkeypatch, capsys, spidev):
    class StoppingSpiDev(FakeSpiDev):
        """Ends the run once the bin number is up, as Ctrl-C would."""

        stopped = False

        def writebytes(self, data):
            super().writebytes(data)
            if not self.stopped and self.rows() == row_bytes(number(3)):
                self.stopped = True
                raise KeyboardInterrupt

    spidev.SpiDev = StoppingSpiDev
    monkeypatch.chdir(tmp_path)
    cv2.imwrite("r.png", render(["yellow", "violet", "red", "gold"]))
    (tmp_path / "config.toml").write_text(
        "[web]\nenabled = false\n"
        "[sorter]\nwindow = 0.3\nmin_readings = 2\nreject_after = 2\n"
        "[servo]\nenabled = false\n"
        "[matrix]\nenabled = true\nbrightness = 1\n"
    )
    assert main(["--source", "r.png", "run"]) == 0
    out = capsys.readouterr().out
    assert "servo: off (disabled in the config)" in out and "matrix: on /dev/spidev0.0" in out
    spi = StoppingSpiDev.instances[-1]
    assert spi.stopped and spi.closed
    pictures = spi.pictures()
    assert pictures[:2] == [row_bytes(BLANK), row_bytes(READY)]
    assert pictures[-2:] == [row_bytes(number(3)), row_bytes(BLANK)]  # the bin, then blanked on exit
    fills = [sum(byte == 0xFF for byte in picture) for picture in pictures[2:-2]]
    assert fills and fills == sorted(fills) and all(1 <= rows <= 8 for rows in fills)  # rising while watched
