"""Resistor colour code tables and the reference palette used to name band colours."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

COLORS = (
    "black", "brown", "red", "orange", "yellow",
    "green", "blue", "violet", "grey", "white",
    "gold", "silver",
)

DIGIT = {color: i for i, color in enumerate(COLORS[:10])}
MULTIPLIER = {**{color: 10.0**i for color, i in DIGIT.items()}, "gold": 0.1, "silver": 0.01}
# Percent, per IEC 60062.
TOLERANCE = {
    "brown": 1.0, "red": 2.0, "orange": 0.05, "yellow": 0.02, "green": 0.5,
    "blue": 0.25, "violet": 0.1, "grey": 0.01, "gold": 5.0, "silver": 10.0,
}
# ppm/K, sixth band.
TEMPCO = {
    "black": 250, "brown": 100, "red": 50, "orange": 15, "yellow": 25,
    "green": 20, "blue": 10, "violet": 5, "grey": 1,
}

# Roughly how each band paint looks to a camera once the background has been
# white-balanced. Only a starting point: `calibrate` replaces these per rig.
DEFAULT_RGB = {
    "black": (30, 30, 30),
    "brown": (110, 60, 30),
    "red": (200, 40, 35),
    "orange": (235, 120, 30),
    "yellow": (235, 205, 40),
    "green": (40, 140, 70),
    "blue": (35, 80, 180),
    "violet": (130, 60, 160),
    "grey": (130, 130, 130),
    "white": (240, 240, 240),
    "gold": (190, 150, 60),
    "silver": (190, 190, 195),
}


# A band narrower than the blur, or a camera that smooths colour more than
# brightness, mixes some of the body's colour into the band's. `Palette.rank`
# allows the band to show between these fractions of its paint, the rest being
# body (above 1: less body than in the reference)...
MIX_RANGE = (0.6, 1.25)
# ...at this cost per unit of mixing.
MIX_COST = 10.0


def to_lab(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 image -> float32 CIELAB (L 0..100, a/b roughly -127..127)."""
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def rgb_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    return to_lab(np.array([[rgb[::-1]]], dtype=np.uint8))[0, 0]


def _lab_to_linear(lab: np.ndarray) -> np.ndarray:
    """(n, 3) Lab -> (n, 3) linear-light BGR, where colours mix by adding."""
    lab = np.asarray(lab, np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2LBGR).reshape(-1, 3)


def _linear_to_lab(bgr: np.ndarray) -> np.ndarray:
    bgr = np.clip(bgr, 0.0, None).astype(np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(bgr, cv2.COLOR_LBGR2Lab).reshape(-1, 3)


class Palette:
    """Names a colour by its nearest reference in Lab space."""

    def __init__(self, samples: dict[str, list[list[float]]] | None = None):
        self._refs = {c: rgb_to_lab(rgb).reshape(1, 3) for c, rgb in DEFAULT_RGB.items()}
        for color, labs in (samples or {}).items():
            if color not in DIGIT and color not in MULTIPLIER:
                raise ValueError(f"unknown colour in calibration: {color!r}")
            if labs:
                # Measured samples replace the built-in guess for that colour.
                self._refs[color] = np.asarray(labs, dtype=np.float32).reshape(-1, 3)
        self._linear = {color: _lab_to_linear(refs) for color, refs in self._refs.items()}

    def rank(self, lab: np.ndarray, body: np.ndarray | None = None) -> list[tuple[str, float]]:
        """All colours with their distance to `lab`, nearest first.

        With the `body` colour around it, each reference may also be mixed
        with some of the body (see MIX_RANGE) to match.
        """
        lab = np.asarray(lab, np.float32)
        distances = []
        for color, refs in self._refs.items():
            distance = np.linalg.norm(refs - lab, axis=1)
            if body is not None:
                paint = self._linear[color]
                base = _lab_to_linear(body)[0]
                seen = _lab_to_linear(lab)[0]
                away = paint - base
                # The mix of paint and body nearest to what is seen, in linear light.
                share = (away @ (seen - base)) / np.maximum((away * away).sum(axis=1), 1e-9)
                share = np.clip(share, *MIX_RANGE)
                mixed = _linear_to_lab(base + share[:, None] * away)
                distance = np.minimum(distance, np.linalg.norm(mixed - lab, axis=1)
                                      + MIX_COST * np.abs(1 - share))
            distances.append((color, float(distance.min())))
        return sorted(distances, key=lambda item: item[1])


def load_calibration(path: str | Path) -> dict[str, list[list[float]]]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())["samples"]


def save_calibration(path: str | Path, samples: dict[str, list[list[float]]]) -> None:
    Path(path).write_text(json.dumps({"samples": samples}, indent=2) + "\n")
