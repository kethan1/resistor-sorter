"""The servo: driven only on a Raspberry Pi, and told the right things."""

import _thread
import threading
import time

import cv2
import pytest
from synth import render

from resistor_sorter import sorter as sorter_module
from resistor_sorter.cli import main
from resistor_sorter.config import ServoConfig, load_config
from resistor_sorter.sorter import Chute, is_raspberry_pi, servo_unavailable


@pytest.fixture
def pwm(monkeypatch, tmp_path):
    """A stand-in for /sys/class/pwm/pwmchip0 with channel 0 already exported."""
    chip = tmp_path / "pwmchip0"
    (chip / "pwm0").mkdir(parents=True)
    monkeypatch.setattr(sorter_module, "PWM_CHIP", chip)
    return chip


def state(pwm) -> dict[str, str]:
    """What the PWM channel was last told."""
    return {f.name: f.read_text() for f in sorted((pwm / "pwm0").iterdir())}


def test_chute_mode_turns_to_the_bin_and_stays(pwm, monkeypatch):
    seen = []
    monkeypatch.setattr(time, "sleep", lambda seconds: seen.append((seconds, state(pwm))))
    chute = Chute(ServoConfig(pin=12, min_pulse_ms=0.5, max_pulse_ms=2.5, settle=0.5))
    assert state(pwm) == {"period": "20000000"}  # 50 Hz, but no pulses until there is somewhere to point
    chute.point(-0.6)
    chute.point(1.5)  # clamped to what the servo can do
    assert seen == [(0.5, {"duty_cycle": "900000", "enable": "1", "period": "20000000"}),
                    (0.5, {"duty_cycle": "2500000", "enable": "1", "period": "20000000"})]
    chute.close()
    assert (pwm / "pwm0" / "enable").read_text() == "0"
    assert (pwm / "unexport").read_text() == "0"


def test_swing_mode_turns_holds_and_turns_back(pwm, monkeypatch):
    seen = []
    monkeypatch.setattr(time, "sleep", lambda seconds: seen.append((seconds, state(pwm)["duty_cycle"])))
    chute = Chute(ServoConfig(mode="swing", min_pulse_ms=1, max_pulse_ms=2, rest_angle=0, hold=1.5, settle=0.2))
    chute.point(0.5)
    # Starts at rest, and is still driven while it holds the bin angle.
    assert seen == [(0.2, "1500000"), (0.2, "1750000"), (1.5, "1750000"), (0.2, "1500000")]


def test_the_channel_is_exported_when_needed(pwm):
    (pwm / "pwm0").rmdir()
    with pytest.raises(RuntimeError, match="GPIO 18.*dtoverlay=pwm"):  # the fake kernel makes no pwm0
        Chute(ServoConfig())
    assert (pwm / "export").read_text() == "0"


def test_servo_pin_must_have_hardware_pwm(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[servo]\npin = 19\n")
    assert load_config(path).servo.pin == 19
    path.write_text("[servo]\npin = 17\n")
    with pytest.raises(ValueError, match="servo.pin"):
        load_config(path)


def test_is_raspberry_pi(tmp_path):
    pi, other = tmp_path / "pi", tmp_path / "other"
    pi.write_bytes(b"Raspberry Pi 4 Model B Rev 1.4\0")
    other.write_bytes(b"Generic x86 PC\0")
    assert is_raspberry_pi([pi])
    assert not is_raspberry_pi([other])
    assert not is_raspberry_pi([tmp_path / "missing"])
    assert is_raspberry_pi([tmp_path / "missing", pi])  # the next file is tried when one is absent


def test_servo_is_only_driven_on_a_pi(monkeypatch, pwm):
    assert servo_unavailable(ServoConfig(enabled=False)) == "disabled in the config"
    monkeypatch.setattr(sorter_module, "is_raspberry_pi", lambda: False)
    assert servo_unavailable(ServoConfig(enabled="auto")) == "not running on a Raspberry Pi"
    assert servo_unavailable(ServoConfig(enabled=True)) is None
    monkeypatch.setattr(sorter_module, "is_raspberry_pi", lambda: True)
    assert servo_unavailable(ServoConfig(enabled="auto")) is None
    monkeypatch.setattr(sorter_module, "PWM_CHIP", pwm / "missing")
    assert servo_unavailable(ServoConfig(enabled="auto")).startswith("hardware PWM is off (add dtoverlay=pwm")


def test_run_turns_the_servo_towards_the_bin(tmp_path, monkeypatch, capsys, pwm):
    monkeypatch.chdir(tmp_path)
    cv2.imwrite("r.png", render(["yellow", "violet", "red", "gold"]))
    (tmp_path / "config.toml").write_text(
        "[web]\nenabled = false\n"
        "[sorter]\nwindow = 0.3\nmin_readings = 2\nreject_after = 2\n"
        "[servo]\nenabled = true\nsettle = 0\n"
    )
    stop = threading.Timer(1.5, _thread.interrupt_main)  # Ctrl-C after the first decision
    stop.start()
    try:
        assert main(["--source", "r.png", "run"]) == 0
    finally:
        stop.cancel()
    out = capsys.readouterr().out
    assert "servo: on GPIO 18, chute mode" in out
    assert "4.7 kΩ ±5%  [yellow violet red gold]  -> 1 kΩ and up  (" in out
    assert state(pwm)["duty_cycle"] == "2162500"  # 0.75 of the way from the middle to max_pulse_ms
    assert (pwm / "pwm0" / "enable").read_text() == "0"  # and let go at the end


def test_run_without_a_pi_reports_but_leaves_the_servo_alone(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sorter_module, "is_raspberry_pi", lambda: False)
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
    assert "-> 1 kΩ and up  (" in out
