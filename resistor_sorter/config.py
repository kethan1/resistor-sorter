"""Settings, with defaults that suit a Pi 4 + OBSBOT Meet 2 USB camera. See config.example.toml."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


OBSBOT_MEET_2 = "/dev/v4l/by-id/usb-Remo_Tech_Co.__Ltd._OBSBOT_Meet_2-video-index0"


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
    stable_frames: int = 5  # identical consecutive readings needed to accept
    reject_after: int = 30  # frames without a stable reading before giving up
    clear_frames: int = 5  # empty frames that mean the resistor has left
    min_margin: float = 4.0  # readings less clear-cut than this are not trusted
    max_color_distance: float = 15.0  # nor readings with a band this far from every reference


@dataclass
class ServoConfig:
    enabled: bool = False
    pin: int = 18  # BCM numbering
    min_pulse_ms: float = 1.0
    max_pulse_ms: float = 2.0
    reject_angle: float = 90.0  # where unreadable resistors go
    settle: float = 0.5  # seconds to wait for the servo to arrive


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
        Bin("under 100 Ω", max=100, angle=-60),
        Bin("100 Ω – 1 kΩ", min=100, max=1e3, angle=-30),
        Bin("1 kΩ – 10 kΩ", min=1e3, max=1e4, angle=0),
        Bin("10 kΩ – 100 kΩ", min=1e4, max=1e5, angle=30),
        Bin("100 kΩ and up", min=1e5, angle=60),
    ]


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    sorter: SorterConfig = field(default_factory=SorterConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)
    web: WebConfig = field(default_factory=WebConfig)
    bins: list[Bin] = field(default_factory=_default_bins)
    calibration: str = "calibration.json"


def _apply(section, data: dict, where: str) -> None:
    known = {f.name for f in fields(section)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown option {where}.{key}")
        setattr(section, key, value)


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
        elif key in ("camera", "vision", "sorter", "servo", "web"):
            _apply(getattr(config, key), value, key)
        else:
            raise ValueError(f"unknown section [{key}]")
    return config
