"""An 8x8 LED matrix on a MAX7219 that shows what the sorter is up to: a ring while it waits for a
resistor, a fill rising while one is read, then the bin's number, or a cross for a reject."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .config import Bin, MatrixConfig
from .sorter import Event, Sorter, unavailable

# MAX7219 registers. The chip was made for seven-segment displays: each "digit"
# register holds one row of the matrix, one bit per LED.
NOOP, DIGIT0, DECODE_MODE, INTENSITY, SCAN_LIMIT, SHUTDOWN, DISPLAY_TEST = 0x00, 0x01, 0x09, 0x0A, 0x0B, 0x0C, 0x0F
SPI_HZ = 1_000_000  # the chip takes up to 10 MHz; slower is safer over jumper wires

Frame = np.ndarray  # 8x8 of bool, row 0 at the top and column 0 at the left


def matrix_unavailable(cfg: MatrixConfig) -> str | None:
    """Why the LED matrix is going to be left alone, or None when it will be driven."""
    return unavailable(cfg.enabled, "spidev", "python3-spidev")


def device_path(cfg: MatrixConfig) -> str:
    return f"/dev/spidev{cfg.bus}.{cfg.device}"


def glyph(rows: Sequence[str]) -> np.ndarray:
    """A bitmap from strings of '.' and 'X'."""
    return np.array([[c == "X" for c in row] for row in rows], dtype=bool)


def place(art: np.ndarray) -> Frame:
    """A frame with `art` (up to 8x8) in the middle."""
    frame = np.zeros((8, 8), dtype=bool)
    top, left = (8 - art.shape[0]) // 2, (8 - art.shape[1]) // 2
    frame[top:top + art.shape[0], left:left + art.shape[1]] = art
    return frame


# Bin numbers: one 5x7 digit, or two 3x5 ones side by side from 10 up.
DIGITS = [glyph(rows) for rows in (
    (".XXX.", "X...X", "X..XX", "X.X.X", "XX..X", "X...X", ".XXX."),
    ("..X..", ".XX..", "..X..", "..X..", "..X..", "..X..", ".XXX."),
    (".XXX.", "X...X", "....X", "...X.", "..X..", ".X...", "XXXXX"),
    ("XXXXX", "...X.", "..X..", "...X.", "....X", "X...X", ".XXX."),
    ("...X.", "..XX.", ".X.X.", "X..X.", "XXXXX", "...X.", "...X."),
    ("XXXXX", "X....", "XXXX.", "....X", "....X", "X...X", ".XXX."),
    ("..XX.", ".X...", "X....", "XXXX.", "X...X", "X...X", ".XXX."),
    ("XXXXX", "....X", "...X.", "..X..", ".X...", ".X...", ".X..."),
    (".XXX.", "X...X", "X...X", ".XXX.", "X...X", "X...X", ".XXX."),
    (".XXX.", "X...X", "X...X", ".XXXX", "....X", "...X.", ".XX.."),
)]
SMALL_DIGITS = [glyph(rows) for rows in (
    ("XXX", "X.X", "X.X", "X.X", "XXX"),
    (".X.", "XX.", ".X.", ".X.", "XXX"),
    ("XXX", "..X", "XXX", "X..", "XXX"),
    ("XXX", "..X", "XXX", "..X", "XXX"),
    ("X.X", "X.X", "XXX", "..X", "..X"),
    ("XXX", "X..", "XXX", "..X", "XXX"),
    ("XXX", "X..", "XXX", "X.X", "XXX"),
    ("XXX", "..X", "..X", "..X", "..X"),
    ("XXX", "X.X", "XXX", "X.X", "XXX"),
    ("XXX", "X.X", "XXX", "..X", "XXX"),
)]
READY = place(glyph(("..XX..", ".X..X.", "X....X", "X....X", ".X..X.", "..XX..")))  # a ring
CROSS = place(glyph(("X.....X", ".X...X.", "..X.X..", "...X...", "..X.X..", ".X...X.", "X.....X")))


def number(n: int) -> Frame:
    """A bin number: one big digit up to 9, two small ones up to 99."""
    if n < 10:
        return place(DIGITS[n])
    tens, ones = divmod(min(n, 99), 10)
    gap = np.zeros((5, 1), dtype=bool)
    return place(np.hstack([SMALL_DIGITS[tens], gap, SMALL_DIGITS[ones]]))


def signs(bins: Sequence[Bin]) -> list[tuple[str, Frame]]:
    """Everything the matrix can show, each with a label: the ready ring, each bin's number, the reject cross."""
    return [("ready", READY), *((f"bin {n}: {b.name}", number(n)) for n, b in enumerate(bins, 1)), ("reject", CROSS)]


def filling(progress: float) -> Frame:
    """Rows lit from the bottom up: one as soon as a resistor is seen, all eight once the window has elapsed."""
    rows = min(8, max(1, math.ceil(progress * 8)))
    frame = np.zeros((8, 8), dtype=bool)
    frame[8 - rows:] = True
    return frame


def oriented(frame: Frame, rotate: int, mirror: bool) -> Frame:
    """`frame` as the chip's rows see it: turned `rotate` degrees clockwise, then flipped left to right if `mirror`."""
    frame = np.rot90(frame, -(rotate // 90))
    return np.fliplr(frame) if mirror else frame


class Max7219:
    """One MAX7219 driving an 8x8 LED matrix over SPI."""

    def __init__(self, cfg: MatrixConfig):
        try:
            import spidev
        except ImportError as error:
            raise ImportError("the LED matrix needs spidev: sudo apt install python3-spidev") from error

        self.cfg = cfg
        self._spi = spidev.SpiDev()
        try:
            self._spi.open(cfg.bus, cfg.device)
        except OSError as error:
            raise RuntimeError(f"cannot open {device_path(cfg)}: {error} (SPI is turned on with "
                               "`sudo raspi-config nonint do_spi 0` and a reboot; "
                               'matrix.enabled = "auto" skips the matrix away from a Pi)') from error
        self._spi.max_speed_hz = SPI_HZ
        self._spi.mode = 0
        # Power-up leaves the chip shut down and blank, but a previous program may not have.
        self._write_all(DISPLAY_TEST, 0)
        self._write_all(SHUTDOWN, 0)  # any further chips in the chain stay dark
        self._write(SCAN_LIMIT, 7)  # drive all eight rows
        self._write(DECODE_MODE, 0)  # one bit per LED, not seven-segment digits
        self._write(INTENSITY, cfg.brightness)
        self.clear()
        self._write(SHUTDOWN, 1)

    def _write(self, register: int, value: int) -> None:
        """Write a register of the first chip; the rest of the chain gets no-ops.

        Each chip passes on what it was holding, so the last pair sent is the
        one that ends up in the chip nearest the Pi.
        """
        self._spi.writebytes([NOOP, 0] * (self.cfg.chain - 1) + [register, value])

    def _write_all(self, register: int, value: int) -> None:
        self._spi.writebytes([register, value] * self.cfg.chain)

    def show(self, frame: Frame) -> None:
        """Light the LEDs set in an 8x8 array (row 0 at the top, column 0 at the left)."""
        frame = np.asarray(frame, dtype=bool)
        if frame.shape != (8, 8):
            raise ValueError(f"a frame is 8x8, not {'x'.join(map(str, frame.shape))}")
        for row, bits in enumerate(oriented(frame, self.cfg.rotate, self.cfg.mirror)):
            self._write(DIGIT0 + row, int(np.packbits(bits)[0]))

    def clear(self) -> None:
        for row in range(8):
            self._write(DIGIT0 + row, 0)

    def close(self) -> None:
        """Blank the display and shut the chip down, so it does not keep showing the last bin."""
        self.clear()
        self._write(SHUTDOWN, 0)
        self._spi.close()


class Panel:
    """Keeps the matrix showing where the sorting loop is: the ready ring, the fill rising while
    a resistor is watched, then its bin number (or a cross for a reject) until it is taken away.
    """

    def __init__(self, matrix: Max7219):
        self._matrix = matrix
        self._frame: Frame | None = None
        self._show(READY)

    def update(self, sorter: Sorter, event: Event | None) -> None:
        """Call once per camera frame, with the sorter's event for it (usually None)."""
        if event is not None:
            self._show(CROSS if event.bin is None else number(sorter.bins.index(event.bin) + 1))
        elif sorter.decided:
            pass  # keep showing the result while the resistor lies there
        elif (progress := sorter.progress) is None:
            self._show(READY)
        else:
            self._show(filling(progress))

    def _show(self, frame: Frame) -> None:
        if self._frame is None or not np.array_equal(frame, self._frame):
            self._frame = frame
            self._matrix.show(frame)

    def close(self) -> None:
        self._matrix.close()
