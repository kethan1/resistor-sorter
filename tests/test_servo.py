"""The servo: driven only on a Raspberry Pi, and told the right things."""

import _thread
import sys
import threading
import time
import types
from typing import ClassVar

import cv2
import pytest
from synth import render

from resistor_sorter import sorter as sorter_module
from resistor_sorter.cli import main
from resistor_sorter.config import ServoConfig
from resistor_sorter.sorter import Chute, is_raspberry_pi, servo_unavailable


class FakeServo:
    """Stands in for gpiozero.AngularServo and logs what the servo was told."""

    instances: ClassVar[list["FakeServo"]] = []

    def __init__(self, pin, **kwargs):
        self.pin, self.kwargs, self.log = pin, kwargs, []
        FakeServo.instances.append(self)

    @property
    def angle(self):
        return next((entry for entry in reversed(self.log) if not isinstance(entry, str)), None)

    @angle.setter
    def angle(self, value):
        self.log.append(value)

    def detach(self):
        self.log.append("detach")

    def close(self):
        self.log.append("close")


@pytest.fixture
def gpiozero(monkeypatch):
    """A stand-in for the gpiozero package, which only exists on the Pi."""
    module = types.ModuleType("gpiozero")
    module.AngularServo = FakeServo
    module.GPIOZeroError = type("GPIOZeroError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "gpiozero", module)
    FakeServo.instances.clear()
    return module


def test_chute_mode_turns_to_the_bin_and_stays(gpiozero):
    chute = Chute(ServoConfig(pin=12, min_pulse_ms=0.5, max_pulse_ms=2.5, settle=0))
    servo = FakeServo.instances[-1]
    assert servo.pin == 12
    assert servo.kwargs == {"initial_angle": None, "min_angle": -90, "max_angle": 90,
                            "min_pulse_width": 0.0005, "max_pulse_width": 0.0025}
    assert servo.log == []  # no pulses until there is somewhere to point
    chute.point(-60)
    chute.point(100)  # clamped to what the servo can do
    chute.close()
    assert servo.log == [-60.0, "detach", 90.0, "detach", "close"]


def test_swing_mode_turns_waits_and_turns_back(gpiozero, monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    chute = Chute(ServoConfig(mode="swing", rest_angle=5, hold=1.5, settle=0.2))
    servo = FakeServo.instances[-1]
    assert servo.log == [5, "detach"]  # starts at rest
    chute.point(60)
    assert servo.log == [5, "detach", 60, "detach", 5, "detach"]
    assert sleeps == [0.2, 0.2, 1.5, 0.2]


def test_gpiozero_problems_are_explained(gpiozero):
    def refuse(*args, **kwargs):
        raise gpiozero.GPIOZeroError("Unable to load any default pin factory!")

    gpiozero.AngularServo = refuse
    with pytest.raises(RuntimeError, match="GPIO 17.*pin factory"):
        Chute(ServoConfig())


def test_missing_gpiozero_is_explained(monkeypatch):
    monkeypatch.setitem(sys.modules, "gpiozero", None)
    with pytest.raises(ImportError, match="python3-gpiozero"):
        Chute(ServoConfig())


def test_is_raspberry_pi(tmp_path):
    pi, other = tmp_path / "pi", tmp_path / "other"
    pi.write_bytes(b"Raspberry Pi 4 Model B Rev 1.4\0")
    other.write_bytes(b"Generic x86 PC\0")
    assert is_raspberry_pi([pi])
    assert not is_raspberry_pi([other])
    assert not is_raspberry_pi([tmp_path / "missing"])
    assert is_raspberry_pi([tmp_path / "missing", pi])  # the next file is tried when one is absent


def test_servo_is_only_driven_on_a_pi(monkeypatch, gpiozero):
    assert servo_unavailable(ServoConfig(enabled=False)) == "disabled in the config"
    monkeypatch.setattr(sorter_module, "is_raspberry_pi", lambda: False)
    assert servo_unavailable(ServoConfig(enabled="auto")) == "not running on a Raspberry Pi"
    assert servo_unavailable(ServoConfig(enabled=True)) is None  # forced, e.g. for a remote or mock pin factory
    monkeypatch.setattr(sorter_module, "is_raspberry_pi", lambda: True)
    assert servo_unavailable(ServoConfig(enabled="auto")) is None
    monkeypatch.setitem(sys.modules, "gpiozero", None)
    assert servo_unavailable(ServoConfig(enabled="auto")) == "python3-gpiozero is not installed"


def test_run_turns_the_servo_towards_the_bin(tmp_path, monkeypatch, capsys, gpiozero):
    class StoppingServo(FakeServo):
        """Ends the run once the servo has been pointed, as Ctrl-C would."""

        @FakeServo.angle.setter
        def angle(self, value):
            self.log.append(value)
            raise KeyboardInterrupt

    gpiozero.AngularServo = StoppingServo
    monkeypatch.chdir(tmp_path)
    cv2.imwrite("r.png", render(["yellow", "violet", "red", "gold"]))
    (tmp_path / "config.toml").write_text(
        "[web]\nenabled = false\n"
        "[sorter]\nwindow = 0.3\nmin_readings = 2\nreject_after = 2\n"
        "[servo]\nenabled = true\nsettle = 0\n"
    )
    assert main(["--source", "r.png", "run"]) == 0
    out = capsys.readouterr().out
    assert "servo: on GPIO 17, chute mode" in out
    assert "4.7 kΩ ±5%  [yellow violet red gold]  -> 1 kΩ – 10 kΩ  (" in out
    assert StoppingServo.instances[-1].log == [0.0, "close"]


def test_run_without_a_pi_reports_but_leaves_the_servo_alone(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sorter_module, "is_raspberry_pi", lambda: False)
    monkeypatch.setitem(sys.modules, "gpiozero", None)  # as on a laptop
    monkeypatch.chdir(tmp_path)
    cv2.imwrite("r.png", render(["yellow", "violet", "red", "gold"]))
    (tmp_path / "config.toml").write_text("[web]\nenabled = false\n[sorter]\nwindow = 0.3\nmin_readings = 2\n")

    stop = threading.Timer(1.5, _thread.interrupt_main)  # Ctrl-C after the first decision
    stop.start()
    try:
        assert main(["--source", "r.png", "run"]) == 0
    finally:
        stop.cancel()
    out = capsys.readouterr().out
    assert "servo: off (not running on a Raspberry Pi)" in out
    assert "matrix: off (not running on a Raspberry Pi)" in out
    assert "light: off (not running on a Raspberry Pi)" in out
    assert "-> 1 kΩ – 10 kΩ  (" in out
