"""The NeoPixel light: driven only on a Raspberry Pi, on while the camera is in use, and told the right things."""

import sys
import types
from typing import ClassVar

import cv2
import pytest
from synth import render

from resistor_sorter import cli
from resistor_sorter import light as light_module
from resistor_sorter import sorter as sorter_module
from resistor_sorter.cli import main
from resistor_sorter.config import LightConfig, load_config
from resistor_sorter.light import Light, light_unavailable, location

WHITE, BLACK = (255, 255, 255), (0, 0, 0)


class FakeNeoPixel:
    """Stands in for neopixel.NeoPixel and logs every colour a strip was shown."""

    instances: ClassVar[list["FakeNeoPixel"]] = []

    def __init__(self, pin, n, *, brightness=1.0, auto_write=True):
        self.pin, self.n, self.brightness, self.auto_write = pin, n, brightness, auto_write
        self.color = None  # what fill() set and show() has yet to send
        self.log = []  # the colours shown, then "deinit"
        FakeNeoPixel.instances.append(self)

    def fill(self, color):
        self.color = tuple(color)

    def show(self):
        self.log.append(self.color)

    def deinit(self):
        self.log.append("deinit")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def neopixel(monkeypatch):
    """Stand-ins for the board and neopixel packages, which only exist on the Pi."""
    board = types.ModuleType("board")
    for n in range(28):
        setattr(board, f"D{n}", f"D{n}")
    module = types.ModuleType("neopixel")
    module.NeoPixel = FakeNeoPixel
    monkeypatch.setitem(sys.modules, "board", board)
    monkeypatch.setitem(sys.modules, "neopixel", module)
    FakeNeoPixel.instances.clear()
    return module


def lit_when_opening(monkeypatch):
    """Records what every strip had shown by the time the camera was opened."""
    seen, real = [], cli.open_source

    def open_source(cfg):
        seen.append([strip.log[:] for strip in FakeNeoPixel.instances])
        return real(cfg)

    monkeypatch.setattr(cli, "open_source", open_source)
    return seen


def test_every_strip_is_lit_then_turned_off(neopixel):
    light = Light(LightConfig(pins=[12, 21], leds=8, brightness=0.5, color=[255, 200, 150]))
    strips = FakeNeoPixel.instances
    assert [(s.pin, s.n, s.brightness, s.auto_write) for s in strips] == [("D12", 8, 0.5, False), ("D21", 8, 0.5, False)]
    assert all(s.log == [(255, 200, 150)] for s in strips)
    light.close()
    assert all(s.log == [(255, 200, 150), BLACK, "deinit"] for s in strips)


def test_the_colour_is_resent_once_a_second(neopixel):
    clock = Clock()
    light = Light(LightConfig(pins=[18]), clock)
    strip = FakeNeoPixel.instances[-1]
    for clock.now in (0.2, 0.9):
        light.refresh()
    assert strip.log == [WHITE]
    for clock.now in (1.0, 1.5):
        light.refresh()
    assert strip.log == [WHITE, WHITE]
    clock.now = 2.0
    light.refresh()
    assert strip.log == [WHITE] * 3


def test_location():
    assert location(LightConfig()) == "GPIO 21 (8 LEDs)"
    assert location(LightConfig(pins=[12, 21])) == "GPIO 12, 21 (8 LEDs each)"
    assert location(LightConfig(pins=[18], leds=30)) == "GPIO 18 (30 LEDs)"


def test_missing_neopixel_is_explained(monkeypatch):
    monkeypatch.setitem(sys.modules, "board", None)
    monkeypatch.setitem(sys.modules, "neopixel", None)
    with pytest.raises(ImportError, match="adafruit-circuitpython-neopixel"):
        Light(LightConfig())


def test_neopixel_problems_are_explained(neopixel):
    class RootlessNeoPixel(FakeNeoPixel):
        def show(self):
            raise RuntimeError("NeoPixel support requires running with sudo, please try again!")

    neopixel.NeoPixel = RootlessNeoPixel
    with pytest.raises(RuntimeError, match="GPIO 12, 21.*sudo"):
        Light(LightConfig(pins=[12, 21]))
    assert [s.log for s in FakeNeoPixel.instances] == [["deinit"]] * 2  # the pins are let go of


def test_light_is_only_driven_on_a_pi(monkeypatch, neopixel):
    assert light_unavailable(LightConfig(enabled=False)) == "disabled in the config"
    monkeypatch.setattr(sorter_module, "is_raspberry_pi", lambda: False)
    assert light_unavailable(LightConfig(enabled="auto")) == "not running on a Raspberry Pi"
    assert light_unavailable(LightConfig(enabled=True)) is None
    monkeypatch.setattr(sorter_module, "is_raspberry_pi", lambda: True)
    assert light_unavailable(LightConfig(enabled="auto")) is None
    monkeypatch.setitem(sys.modules, "neopixel", None)
    assert light_unavailable(LightConfig(enabled="auto")) == "the NeoPixel library is not installed"


def test_config_checks(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[light]\npins = [18]\nleds = 30\nbrightness = 0.3\ncolor = [255, 240, 220]\n")
    light = load_config(path).light
    assert (light.pins, light.leds, light.brightness, light.color) == ([18], 30, 0.3, [255, 240, 220])
    for text, complaint in [
        ('[light]\nenabled = "maybe"\n', "light.enabled"),
        ("[light]\npins = []\n", "light.pins"),
        ("[light]\npins = 12\n", "light.pins"),
        ("[light]\npins = [40]\n", "light.pins"),
        ("[light]\nleds = 0\n", "light.leds"),
        ("[light]\nbrightness = 1.5\n", "light.brightness"),
        ("[light]\ncolor = [255, 255]\n", "light.color"),
        ("[light]\ncolor = [0, 0, 300]\n", "light.color"),
    ]:
        path.write_text(text)
        with pytest.raises(ValueError, match=complaint):
            load_config(path)


def test_run_lights_the_scene_before_the_camera_opens_and_darkens_it_after(tmp_path, monkeypatch, capsys, neopixel):
    class StoppingNeoPixel(FakeNeoPixel):
        """Ends the run at the first resend, as Ctrl-C would."""

        stopped = False

        def show(self):
            super().show()
            if not StoppingNeoPixel.stopped and len(self.log) == 2:
                StoppingNeoPixel.stopped = True
                raise KeyboardInterrupt

    neopixel.NeoPixel = StoppingNeoPixel
    monkeypatch.setattr(light_module, "REFRESH_EVERY", 0)  # resend on the very first frame
    opened = lit_when_opening(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cv2.imwrite("r.png", render(["yellow", "violet", "red", "gold"]))
    (tmp_path / "config.toml").write_text("[web]\nenabled = false\n[servo]\nenabled = false\n[light]\nenabled = true\npins = [12, 21]\n")
    assert main(["--source", "r.png", "run"]) == 0
    assert "light: on GPIO 12, 21 (8 LEDs each)" in capsys.readouterr().out
    assert opened == [[[WHITE], [WHITE]]]  # both strips were lit before the camera opened
    first, second = FakeNeoPixel.instances
    assert first.log == [WHITE, WHITE, BLACK, "deinit"]  # lit, resent (and stopped), then off
    assert second.log == [WHITE, BLACK, "deinit"]


def test_snapshots_are_taken_under_the_light(tmp_path, monkeypatch, neopixel):
    opened = lit_when_opening(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cv2.imwrite("r.png", render(["yellow", "violet", "red", "gold"]))
    (tmp_path / "config.toml").write_text("[light]\nenabled = true\npins = [18]\n")
    assert main(["--source", "r.png", "snap", "out.png"]) == 0
    assert (tmp_path / "out.png").exists()
    assert opened == [[[WHITE]]]
    assert FakeNeoPixel.instances[-1].log == [WHITE, BLACK, "deinit"]

