"""Decide when a resistor has been read reliably, which bin it goes to, and move the chute."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Bin, ServoConfig, SorterConfig
from .decode import Reading


@dataclass
class Event:
    """One resistor dealt with. `reading` and `bin` are None when it was rejected."""

    reading: Reading | None
    bin: Bin | None


def pick_bin(bins: list[Bin], ohms: float) -> Bin | None:
    return next((b for b in bins if b.accepts(ohms)), None)


class Sorter:
    """Turns a stream of per-frame readings into one decision per resistor.

    A resistor is accepted once the same value has been read on
    `stable_frames` consecutive frames, and rejected if that has not happened
    after `reject_after` frames. Nothing more happens until it leaves the view.
    """

    def __init__(self, cfg: SorterConfig, bins: list[Bin]):
        self.cfg = cfg
        self.bins = bins
        self._reset()

    def _reset(self) -> None:
        self._decided = False
        self._seen = 0
        self._absent = 0
        self._streak = 0
        self._last: float | None = None

    def trusted(self, reading: Reading | None) -> bool:
        return (
            reading is not None
            and reading.margin >= self.cfg.min_margin
            and reading.worst <= self.cfg.max_color_distance
        )

    def step(self, present: bool, reading: Reading | None) -> Event | None:
        if not present:
            self._absent += 1
            if self._absent >= self.cfg.clear_frames:
                self._reset()
            return None
        self._absent = 0
        if self._decided:
            return None

        self._seen += 1
        if self.trusted(reading):
            self._streak = self._streak + 1 if reading.ohms == self._last else 1
            self._last = reading.ohms
        else:
            self._streak, self._last = 0, None

        if self._streak >= self.cfg.stable_frames:
            self._decided = True
            return Event(reading, pick_bin(self.bins, reading.ohms))
        if self._seen >= self.cfg.reject_after:
            self._decided = True
            return Event(None, None)
        return None


class Chute:
    """A hobby servo that aims a chute at the chosen bin."""

    def __init__(self, cfg: ServoConfig):
        from gpiozero import AngularServo

        self.cfg = cfg
        self._servo = AngularServo(
            cfg.pin,
            initial_angle=None,
            min_angle=-90,
            max_angle=90,
            min_pulse_width=cfg.min_pulse_ms / 1000,
            max_pulse_width=cfg.max_pulse_ms / 1000,
        )

    def point(self, angle: float) -> None:
        self._servo.angle = max(-90.0, min(90.0, angle))
        time.sleep(self.cfg.settle)
        # Stop sending pulses so software PWM jitter does not make the servo buzz.
        self._servo.detach()

    def close(self) -> None:
        self._servo.close()
