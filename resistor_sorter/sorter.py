"""Decide when a resistor has been read reliably, which bin it goes to, and move the servo."""

from __future__ import annotations

import importlib
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .config import Bin, ServoConfig, SorterConfig
from .decode import Reading


@dataclass
class Event:
    """One resistor dealt with. `reading` and `bin` are None when it was rejected.

    `tally` lists each distinct value read while the resistor was watched and
    how often, most frequent first; `agreed` of the `votes` readings pointed at
    the chosen bin.
    """

    reading: Reading | None
    bin: Bin | None
    tally: list[tuple[Reading, int]] = field(default_factory=list)
    agreed: int = 0
    votes: int = 0


def pick_bin(bins: list[Bin], ohms: float) -> Bin | None:
    return next((b for b in bins if b.accepts(ohms)), None)


class Sorter:
    """Turns a stream of per-frame readings into one decision per resistor.

    No single frame is trusted on its own. A resistor is watched for `window`
    seconds and goes to the bin that at least `min_agreement` of the trusted
    readings from that time agree on, as long as there are `min_readings` of
    them and the latest `stable_readings` all agree with it too, so the
    current reading matches the rolling consensus rather than drifting away
    from it. Readings older than the window are forgotten, so a bad start while
    the resistor is being put down does not count against it. If the readings
    never agree it is rejected after `reject_after` seconds. Nothing more
    happens until it has been out of view for `clear_after` seconds.
    """

    def __init__(self, cfg: SorterConfig, bins: list[Bin], clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.bins = bins
        self._clock = clock
        self._reset()

    def _reset(self) -> None:
        self._decided = False
        self._since: float | None = None  # when the resistor came into view
        self._gone: float | None = None  # when it dropped out of view (None while in view)
        self._votes: deque[tuple[float, Reading]] = deque()

    def trusted(self, reading: Reading | None) -> bool:
        return (
            reading is not None
            and reading.margin >= self.cfg.min_margin
            and reading.worst <= self.cfg.max_color_distance
        )

    @property
    def decided(self) -> bool:
        """Whether the resistor in view has been dealt with; it is then ignored until it is taken away."""
        return self._decided

    @property
    def progress(self) -> float | None:
        """How far through the window the resistor in view is: 0 when it has just appeared, 1 when
        the window has elapsed, and beyond while it is still undecided. None with nothing in view,
        or once the resistor has been dealt with.
        """
        if self._since is None or self._decided:
            return None
        return (self._clock() - self._since) / self.cfg.window

    def step(self, present: bool, reading: Reading | None) -> Event | None:
        now = self._clock()
        if not present:
            if self._since is not None:
                if self._gone is None:
                    self._gone = now
                if now - self._gone >= self.cfg.clear_after:
                    self._reset()
            return None
        self._gone = None
        if self._decided:
            return None
        if self._since is None:
            self._since = now

        if self.trusted(reading):
            self._votes.append((now, reading))
        while self._votes and now - self._votes[0][0] > self.cfg.window:
            self._votes.popleft()

        watched = now - self._since
        if watched >= self.cfg.window:
            event = self._decide()
            if event is not None:
                self._decided = True
                return event
        if watched >= self.cfg.reject_after:
            self._decided = True
            readings = [reading for _, reading in self._votes]
            return Event(None, None, self._tally(readings), 0, len(readings))
        return None

    def _bin_index(self, reading: Reading) -> int | None:
        chosen = pick_bin(self.bins, reading.ohms)
        return None if chosen is None else self.bins.index(chosen)

    def _decide(self) -> Event | None:
        readings = [reading for _, reading in self._votes]
        if len(readings) < self.cfg.min_readings:
            return None
        counts = Counter(self._bin_index(reading) for reading in readings)
        index, agreed = counts.most_common(1)[0]
        if agreed < self.cfg.min_agreement * len(readings):
            return None
        latest = readings[-self.cfg.stable_readings:]
        if len(latest) < self.cfg.stable_readings or any(self._bin_index(r) != index for r in latest):
            return None
        tally = self._tally(readings)
        # Report the value read most often among those that chose the winning bin.
        reading = next(r for r, _ in tally if self._bin_index(r) == index)
        return Event(reading, None if index is None else self.bins[index], tally, agreed, len(readings))

    @staticmethod
    def _tally(readings: Iterable[Reading]) -> list[tuple[Reading, int]]:
        """Each distinct value with how often it was read, most frequent first (the clearest reading of each)."""
        counts: Counter[float] = Counter()
        clearest: dict[float, Reading] = {}
        for reading in readings:
            counts[reading.ohms] += 1
            if reading.ohms not in clearest or reading.margin > clearest[reading.ohms].margin:
                clearest[reading.ohms] = reading
        return [(clearest[ohms], n) for ohms, n in counts.most_common()]


MODEL_FILES = ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model")


def is_raspberry_pi(model_files: Iterable[str | Path] = MODEL_FILES) -> bool:
    """Whether this is Raspberry Pi hardware (any model), going by the device tree's model name."""
    for path in model_files:
        try:
            return Path(path).read_bytes().startswith(b"Raspberry Pi")
        except OSError:
            continue
    return False


def unavailable(enabled: bool | str, module: str, package: str) -> str | None:
    """Why a peripheral is going to be left alone, or None when it will be driven.

    `enabled` is true, false or "auto": only on a Raspberry Pi where the apt
    `package` has provided the Python `module` that drives it.
    """
    if not enabled:
        return "disabled in the config"
    if enabled == "auto":
        if not is_raspberry_pi():
            return "not running on a Raspberry Pi"
        try:
            importlib.import_module(module)
        except ImportError:
            return f"{package} is not installed"
    return None


def servo_unavailable(cfg: ServoConfig) -> str | None:
    """Why the servo is going to be left alone, or None when it will be driven."""
    return unavailable(cfg.enabled, "gpiozero", "python3-gpiozero")


class Chute:
    """A hobby servo that aims a chute at the chosen bin, or swings towards it and back."""

    def __init__(self, cfg: ServoConfig):
        try:
            from gpiozero import AngularServo, GPIOZeroError
        except ImportError as error:
            raise ImportError("the servo needs gpiozero: sudo apt install python3-gpiozero") from error

        self.cfg = cfg
        try:
            self._servo = AngularServo(
                cfg.pin,
                initial_angle=None,
                min_angle=-90,
                max_angle=90,
                min_pulse_width=cfg.min_pulse_ms / 1000,
                max_pulse_width=cfg.max_pulse_ms / 1000,
            )
        except GPIOZeroError as error:
            raise RuntimeError(f"cannot drive a servo on GPIO {cfg.pin}: {error} "
                               '(servo.enabled = "auto" skips the servo away from a Pi)') from error
        if cfg.mode == "swing":
            self._move(cfg.rest_angle)

    def point(self, angle: float) -> None:
        """Turn to `angle`; in "swing" mode, hold there and then turn back to the rest angle."""
        if self.cfg.mode == "swing":
            # Keep driving the servo through the hold, so the paddle is not pushed back while it waits.
            self._move(angle, detach=False)
            time.sleep(self.cfg.hold)
            self._move(self.cfg.rest_angle)
        else:
            self._move(angle)

    def _move(self, angle: float, detach: bool = True) -> None:
        self._servo.angle = max(-90.0, min(90.0, angle))
        time.sleep(self.cfg.settle)
        if detach:
            # Stop sending pulses so software PWM jitter does not make the servo buzz.
            self._servo.detach()

    def close(self) -> None:
        self._servo.close()
