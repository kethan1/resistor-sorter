"""WS2812 (NeoPixel) strips that light the resistor. They are on whenever the camera is in use, so
that calibration and sorting see the same light, and off again when the program exits."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

from .config import LightConfig
from .sorter import unavailable

REFRESH_EVERY = 1.0  # seconds between resending the colour, so a strip that lost it to a dip in its supply recovers
INSTALL = "sudo pip3 install --break-system-packages rpi_ws281x adafruit-circuitpython-neopixel"


def light_unavailable(cfg: LightConfig) -> str | None:
    """Why the light is going to be left alone, or None when it will be driven."""
    return unavailable(cfg.enabled, "neopixel", "the NeoPixel library")


def location(cfg: LightConfig) -> str:
    """Where the strips are, e.g. "GPIO 12, 21 (8 LEDs each)"."""
    each = " each" if len(cfg.pins) > 1 else ""
    return f"GPIO {', '.join(map(str, cfg.pins))} ({cfg.leds} LEDs{each})"


class Light:
    """NeoPixel strips, one per GPIO, all showing one colour from construction until close()."""

    def __init__(self, cfg: LightConfig, clock: Callable[[], float] = time.monotonic):
        try:
            import board
            import neopixel
        except ImportError as error:
            raise ImportError(f"the light needs the NeoPixel library: {INSTALL}") from error

        self.cfg = cfg
        self._clock = clock
        self._color = tuple(cfg.color)
        self._strips = []
        try:
            for pin in cfg.pins:
                self._strips.append(neopixel.NeoPixel(getattr(board, f"D{pin}"), cfg.leds,
                                                      brightness=cfg.brightness, auto_write=False))
            self._show(self._color)
        except (RuntimeError, OSError) as error:
            self._release()
            raise RuntimeError(f"cannot drive the light on {location(cfg)}: {error} "
                               "(light.enabled = false does without the light)") from error

    def refresh(self) -> None:
        """Call often: resends the colour every REFRESH_EVERY seconds."""
        if self._clock() - self._shown >= REFRESH_EVERY:
            self._show(self._color)

    def _show(self, color: tuple[int, ...]) -> None:
        for strip in self._strips:
            strip.fill(color)
            strip.show()
        self._shown = self._clock()

    def close(self) -> None:
        """Turn the strips off and let go of the pins."""
        try:
            self._show((0, 0, 0))
        finally:
            self._release()

    def _release(self) -> None:
        strips, self._strips = self._strips, []
        for strip in strips:
            # deinit blanks the strip once more, which fails wherever show() just did.
            with contextlib.suppress(Exception):
                strip.deinit()
