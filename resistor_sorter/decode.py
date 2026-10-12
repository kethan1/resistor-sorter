"""Turn a sequence of band colours into a resistance."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Sequence

from .colors import DIGIT, MULTIPLIER, TEMPCO, TOLERANCE

E24 = frozenset({10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30,
                 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91})
E96 = frozenset(round(100 * 10 ** (i / 96)) for i in range(96))
_STANDARD = {2: E24, 3: E96 | {value * 10 for value in E24}}

# Score penalties, in the same units as colour distance (Lab delta-E).
NONSTANDARD_PENALTY = 10.0  # significant digits are not a preferred E-series value
ORIENTATION_PENALTY = 10.0  # band spacing suggests the resistor is read backwards
# Three bands mean a rare 20% resistor, or more often a 4- or 5-band code with
# bands missed.
THREE_BAND_PENALTY = 8.0
# Four-band resistors are nearly always 5% or 10% (gold, silver) and five- and
# six-band ones 1% or 2% (brown, red); anything else is more likely a misread.
USUAL_TOLERANCE = {4: {"gold", "silver"}, 5: {"brown", "red"}, 6: {"brown", "red"}}
RARE_TOLERANCE_PENALTY = 6.0
# How much wider the first gap may be than the last, as a fraction of the mean
# gap, before the reading direction is doubted, and how much more for full doubt.
SPACING_SLACK, SPACING_FULL = 0.1, 0.25

# Per band, only colours this close to the best match are considered.
MAX_CANDIDATES = 3
CANDIDATE_SLACK = 25.0

Candidates = Sequence[tuple[str, float]]


@dataclass(frozen=True)
class Reading:
    ohms: float
    tolerance: float  # percent
    colors: tuple[str, ...]
    tempco: int | None = None  # ppm/K
    standard: bool = True  # significant digits are an E24/E96 value
    flipped: bool = False  # bands were read right to left
    score: float = 0.0  # lower is better
    margin: float = math.inf  # score gap to the best reading with a different value
    worst: float = 0.0  # colour distance of the worst-matching band

    def __str__(self) -> str:
        return self.text()

    def text(self, ascii: bool = False) -> str:
        ohm, plus_minus = ("ohm", "+/-") if ascii else ("Ω", "±")
        if len(self.colors) == 1:
            return f"0 {ohm} jumper"
        text = f"{format_ohms(self.ohms, ohm)} {plus_minus}{self.tolerance:g}%"
        if self.tempco is not None:
            text += f" {self.tempco} ppm/K"
        return text


def format_ohms(ohms: float, unit: str = "Ω") -> str:
    for factor, prefix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if ohms >= factor:
            return f"{ohms / factor:g} {prefix}{unit}"
    return f"{ohms:g} {unit}"


def decode(colors: Sequence[str]) -> Reading | None:
    """Decode bands in the order given. None if they are not a valid code."""
    colors = tuple(colors)
    n = len(colors)
    if n == 1:
        return Reading(0.0, 0.0, colors) if colors[0] == "black" else None
    if n not in (3, 4, 5, 6):
        return None

    ndigits = 2 if n <= 4 else 3
    digits, rest = colors[:ndigits], colors[ndigits:]
    if any(c not in DIGIT for c in digits) or digits[0] == "black":
        return None
    if rest[0] not in MULTIPLIER:
        return None
    tolerance, tempco = 20.0, None
    if len(rest) > 1:
        if rest[1] not in TOLERANCE:
            return None
        tolerance = TOLERANCE[rest[1]]
    if len(rest) > 2:
        if rest[2] not in TEMPCO:
            return None
        tempco = TEMPCO[rest[2]]

    significant = int("".join(str(DIGIT[c]) for c in digits))
    ohms = float(f"{significant * MULTIPLIER[rest[0]]:.6g}")
    return Reading(ohms, tolerance, colors, tempco, significant in _STANDARD[ndigits])


def _orientation_penalty(centers: Sequence[float] | None) -> float:
    """Penalise a reading direction that the band spacing contradicts.

    The tolerance band sits apart from the others, so the first gap being
    wider than the last means we are reading from the wrong end. On some
    resistors it is only a little wider, so the doubt grows with the
    difference. With only three bands the tolerance band is absent and leaves
    empty body at the far end instead.
    """
    if centers is None or len(centers) < 3:
        return 0.0
    if len(centers) == 3:
        return ORIENTATION_PENALTY if centers[0] > (1 - centers[-1]) + 0.05 else 0.0
    gaps = [b - a for a, b in zip(centers, centers[1:])]
    lopsided = (gaps[0] - gaps[-1]) / (sum(gaps) / len(gaps))
    return ORIENTATION_PENALTY * min(1.0, max(0.0, (lopsided - SPACING_SLACK) / SPACING_FULL))


def _shortlist(ranked: Candidates) -> Candidates:
    best = ranked[0][1]
    return [(c, d) for c, d in ranked[:MAX_CANDIDATES] if d <= best + CANDIDATE_SLACK]


def interpret(
    candidates: Sequence[Candidates], centers: Sequence[float] | None = None
) -> Reading | None:
    """Pick the most plausible reading of a row of detected bands.

    `candidates` holds, for each band from left to right, its possible colours
    as (name, distance) pairs sorted nearest first. `centers` are the bands'
    positions along the body (0..1). Both reading directions and the runner-up
    colours of each band are tried; the winner is the valid code with the
    lowest total colour distance plus penalties.
    """
    if not candidates:
        return None
    shortlists = [_shortlist(ranked) for ranked in candidates]

    best: dict[float, Reading] = {}
    for flipped in (False, True):
        bands = shortlists[::-1] if flipped else shortlists
        positions = centers
        if centers is not None and flipped:
            positions = [1 - c for c in reversed(centers)]
        bias = _orientation_penalty(positions)
        for combo in itertools.product(*bands):
            reading = decode([color for color, _ in combo])
            if reading is None:
                continue
            score = sum(distance for _, distance in combo) + bias
            if not reading.standard:
                score += NONSTANDARD_PENALTY
            if len(reading.colors) == 3:
                score += THREE_BAND_PENALTY
            elif len(reading.colors) >= 4:
                tolerance = reading.colors[3 if len(reading.colors) == 4 else 4]
                if tolerance not in USUAL_TOLERANCE[len(reading.colors)]:
                    score += RARE_TOLERANCE_PENALTY
            if reading.ohms not in best or score < best[reading.ohms].score:
                worst = max(distance for _, distance in combo)
                best[reading.ohms] = replace(reading, score=score, flipped=flipped, worst=worst)

    if not best:
        return None
    ranked = sorted(best.values(), key=lambda r: r.score)
    margin = ranked[1].score - ranked[0].score if len(ranked) > 1 else math.inf
    return replace(ranked[0], margin=margin)
