"""Settings, with defaults that suit a Pi 4 + OBSBOT Meet 2 USB camera. See config.example.toml."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


OBSBOT_MEET_2 = "/dev/v4l/by-id/usb-Remo_Tech_Co.__Ltd._OBSBOT_Meet_2-video-index0"
SERVO_MODES = ("chute", "swing")
MATRIX_ROTATIONS = (0, 90, 180, 270)


@dataclass
class CameraConfig:
    source: str = OBSBOT_MEET_2  # a /dev/video* path, a V4L2 index ("0"), "picamera", or a file
    width: int = 3840
    height: int = 2160
    lock_after: float = 2.0  # seconds to let exposure/white balance settle; the Pi camera then locks them
    fps: float = 30.0
    focus: int = -1  # manual lens position, 0-100 on the OBSBOT; -1 = autofocus
    zoom: int = -1  # 0 (widest) - 100 on the OBSBOT; -1 = leave the camera's setting alone


@dataclass
class VisionConfig:
    work_width: int = 640  # frame is shrunk to this width to locate the body
    strip_width: int = 320  # width of the straightened body crop
    fg_threshold: float = 14.0  # Lab distance from the background that counts as "object"
    shadow_weight: float = 0.5  # how much pure brightness differences count (shadows, glare)
    min_body_frac: float = 0.0005  # smallest body, as a fraction of the frame area
    max_diameter_frac: float = 0.3  # thickest body, as a fraction of the frame width
    min_aspect: float = 1.4  # body length / diameter limits
    max_aspect: float = 8.0
    band_threshold: float = 9.0  # Lab step along the body that starts a new segment
    body_tolerance: float = 12.0  # segments this close to the body colour are body
    min_band_frac: float = 0.025  # narrowest band, as a fraction of body length
    end_trim: float = 0.04  # ignored at each end of the body


@dataclass
class SorterConfig:
    window: float = 2.0  # seconds a resistor is watched; the readings over that time decide its bin
    min_readings: int = 3  # trusted readings a decision needs
    min_agreement: float = 0.75  # share of the window's trusted readings that must agree on the bin
    stable_readings: int = 3  # the latest trusted readings that must all agree with that bin too
    reject_after: float = 6.0  # seconds without a decision before the resistor is rejected
    clear_after: float = 0.5  # seconds with nothing in view before the next resistor is expected
    min_margin: float = 4.0  # readings less clear-cut than this are not trusted
    max_color_distance: float = 15.0  # nor readings with a band this far from every reference


@dataclass
class ServoConfig:
    enabled: bool | str = "auto"  # true, false, or "auto": only on a Raspberry Pi with hardware PWM turned on
    pin: int = 18  # BCM numbering; a hardware PWM pin (12 or 18, 13 or 19) set up by the pwm overlay
    min_pulse_ms: float = 0.5  # an SG90's full travel; 1.0 / 2.0 for a servo that strains at +-90
    max_pulse_ms: float = 2.4
    settle: float = 0.5  # seconds to wait for the servo to arrive
    mode: str = "chute"  # "chute": stay aimed at the bin; "swing": go there, wait `hold`, return to `rest_angle`
    rest_angle: float = 0.0  # ("swing" only) where the servo waits between resistors
    hold: float = 3.0  # ("swing" only) seconds at the bin angle before swinging back


@dataclass
class MatrixConfig:
    """A MAX7219-driven 8x8 LED matrix that shows which bin each resistor went to."""

    enabled: bool | str = "auto"  # true, false, or "auto": only on a Raspberry Pi with spidev installed
    bus: int = 0  # /dev/spidev<bus>.<device>; SPI0 with chip select CE0 (GPIO 8) by default
    device: int = 0
    brightness: int = 4  # 0 (dimmest) to 15
    rotate: int = 0  # 0, 90, 180 or 270 degrees clockwise, for however the module is mounted
    mirror: bool = False  # also flip left to right (after rotating), for modules wired the other way round
    chain: int = 1  # MAX7219s daisy-chained on the module (4 on a "4-in-1"); only the first is used


@dataclass
class LightConfig:
    """WS2812 (NeoPixel) strips that light the resistor whenever the camera is in use."""

    enabled: bool | str = "auto"  # true, false, or "auto": only on a Raspberry Pi with the neopixel library installed
    pins: list[int] = field(default_factory=lambda: [21])  # BCM, one per data line: 21 (PCM); 12/18 (PWM) and 10 (SPI) clash with the servo and matrix
    leds: int = 8  # LEDs on each strip
    brightness: float = 1.0  # 0 to 1
    color: list[int] = field(default_factory=lambda: [255, 255, 255])  # red, green, blue, 0-255 each


@dataclass
class WebConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    max_fps: float = 15.0  # preview frame rate cap, to spare the CPU


@dataclass
class Bin:
    name: str
    min: float = 0.0  # ohms, inclusive
    max: float = math.inf  # ohms, exclusive
    values: list[float] = field(default_factory=list)  # exact values; overrides min/max
    angle: float = 0.0  # chute servo angle, -90..90

    def accepts(self, ohms: float) -> bool:
        if self.values:
            return any(math.isclose(ohms, value, rel_tol=1e-3) for value in self.values)
        return self.min <= ohms < self.max


def _default_bins() -> list[Bin]:
    return [
        Bin("under 1 kΩ", max=100, angle=-0.75),
        Bin("1 kΩ and up", min=1e3, max=1e4, angle=0.75),
    ]


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    sorter: SorterConfig = field(default_factory=SorterConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    light: LightConfig = field(default_factory=LightConfig)
    web: WebConfig = field(default_factory=WebConfig)
    bins: list[Bin] = field(default_factory=_default_bins)
    calibration: str = "calibration.json"


def _apply(section, data: dict, where: str) -> None:
    known = {f.name for f in fields(section)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown option {where}.{key}")
        setattr(section, key, value)


def _check(config: Config) -> None:
    """Catch the settings that would otherwise fail late, or quietly do something else."""
    servo, matrix, light, sorter = config.servo, config.matrix, config.light, config.sorter
    for name, section in (("servo", servo), ("matrix", matrix), ("light", light)):
        if not isinstance(section.enabled, bool) and section.enabled != "auto":
            raise ValueError(f'{name}.enabled must be true, false or "auto"')
    if servo.pin not in (12, 13, 18, 19):
        raise ValueError("servo.pin must be a hardware PWM pin: 12 or 18 (channel 0), 13 or 19 (channel 1)")
    if servo.mode not in SERVO_MODES:
        raise ValueError(f"servo.mode must be one of {', '.join(SERVO_MODES)}")
    if not 0 <= matrix.brightness <= 15:
        raise ValueError("matrix.brightness goes from 0 (dimmest) to 15")
    if matrix.rotate not in MATRIX_ROTATIONS:
        raise ValueError(f"matrix.rotate must be one of {', '.join(map(str, MATRIX_ROTATIONS))}")
    if matrix.chain < 1:
        raise ValueError("matrix.chain is the number of MAX7219s on the module, at least 1")
    if (not isinstance(light.pins, list) or not light.pins
            or any(not isinstance(pin, int) or not 0 <= pin <= 27 for pin in light.pins)):
        raise ValueError("light.pins lists the BCM GPIO (0-27) of each strip's data line, e.g. [21]; "
                         "light.enabled = false does without a light")
    if light.leds < 1:
        raise ValueError("light.leds is the number of LEDs on each strip, at least 1")
    if not 0 <= light.brightness <= 1:
        raise ValueError("light.brightness goes from 0 to 1")
    if (not isinstance(light.color, list) or len(light.color) != 3
            or any(not isinstance(c, int) or not 0 <= c <= 255 for c in light.color)):
        raise ValueError("light.color is [red, green, blue], 0 to 255 each")
    if not 0 < sorter.min_agreement <= 1:
        raise ValueError("sorter.min_agreement is a share between 0 and 1, e.g. 0.75")
    if sorter.stable_readings < 1:
        raise ValueError("sorter.stable_readings is how many of the latest readings must agree, at least 1")
    if sorter.reject_after < sorter.window:
        raise ValueError("sorter.reject_after must not be shorter than sorter.window")


def load_config(path: str | Path | None) -> Config:
    """Load a TOML file over the defaults. With no path, ./config.toml is used if present."""
    config = Config()
    if path is None:
        path = Path("config.toml")
        if not path.exists():
            return config
    with open(path, "rb") as f:
        data = tomllib.load(f)

    for key, value in data.items():
        if key == "bin":
            config.bins = [Bin(**entry) for entry in value]
        elif key == "calibration":
            config.calibration = value
        elif key in ("camera", "vision", "sorter", "servo", "matrix", "light", "web"):
            _apply(getattr(config, key), value, key)
        else:
            raise ValueError(f"unknown section [{key}]")
    _check(config)
    return config
