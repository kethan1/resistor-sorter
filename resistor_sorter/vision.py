"""Find a resistor lying on a plain light background and measure its colour bands.

Pipeline: model the light falling on the paper and divide it out, segment
everything that is not paper, erode away the leads to leave the body, and
straighten the body into a horizontal strip. A colour profile is read along the
strip's middle. The body colour is the one at the end caps and in the gaps;
the bands are the stretches that stand out from it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .colors import DEFAULT_RGB, Palette, to_lab
from .config import VisionConfig
from .decode import Reading, interpret

# Grey level the background is normalised to.
_BACKGROUND = 230
_BACKGROUND_LAB = to_lab(np.full((1, 1, 3), _BACKGROUND, np.uint8))[0, 0]
# Score cost of ignoring a segment at the end of the body (see `read`). About
# a poor colour match, so a segment is only ignored if it matches no colour well.
END_DROP_PENALTY = 15.0
# A segment standing out from the body by less than this many times
# `body_tolerance` may be a bright or shaded patch of bare body: glare and the
# bulging end caps catch the light. `read` may ignore it, at a cost of its contrast.
WEAK = 2.0
# At most this many segments are considered for ignoring.
MAX_DROPPED = 3
# How strongly being at the ends of the resistor marks a colour as the body's
# (see `_body_runs`), against the fraction of the length it covers.
END_BONUS = 0.25
# A second guess at the body colour wins only if its reading is better by
# this much per band (see `_body_runs`).
RUNNER_UP_PENALTY = 3.0
# A stretch that stands out from the body is cut in two where its contrast
# falls below this fraction of the peaks either side: a blurred gap of body.
DIP = 0.75
# A stretch this many times the usual band width may be two bands run together.
WIDE = 1.6
# The strip is cut this much longer and wider than the located body, so that
# the body's true outline can be found inside it.
_MARGIN_ALONG, _MARGIN_ACROSS = 0.05, 0.25


@dataclass(eq=False)
class Band:
    lab: np.ndarray
    start: float  # position along the body, 0..1
    end: float
    candidates: list[tuple[str, float]]  # (colour, distance), nearest first
    at_end: bool = False  # runs into the end of the body; may be end cap, not paint
    drop_cost: float = math.inf  # score cost of reading it as bare body instead (see `read`)
    middle: float | None = None  # where it stands out most; the centre if not given

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2 if self.middle is None else self.middle

    @property
    def color(self) -> str:
        return self.candidates[0][0]


@dataclass
class Detection:
    origin: np.ndarray  # corner of the body in frame pixels
    along: np.ndarray  # vector spanning the body's length
    across: np.ndarray  # vector spanning the body's diameter
    strip: np.ndarray  # straightened, white-balanced BGR crop of the body
    body_lab: np.ndarray
    bands: list[Band]

    def point(self, fx: float, fy: float) -> np.ndarray:
        """Frame coordinates of a position given as fractions of length and diameter."""
        return self.origin + fx * self.along + fy * self.across

    @property
    def corners(self) -> np.ndarray:
        return np.array([self.point(0, 0), self.point(1, 0), self.point(1, 1), self.point(0, 1)])


def _border(frame: np.ndarray) -> np.ndarray:
    """A sample of the pixels around the edge of the frame, as an (n, 3) array."""
    h, w = frame.shape[:2]
    m = max(2, min(h, w) // 20)
    return np.concatenate([
        frame[:m:2, ::4].reshape(-1, 3),
        frame[-m::2, ::4].reshape(-1, 3),
        frame[m:-m:4, :m:2].reshape(-1, 3),
        frame[m:-m:4, -m::2].reshape(-1, 3),
    ])


def normalise(frame: np.ndarray) -> np.ndarray:
    """Scale each channel so the background, sampled at the frame edge, is neutral grey."""
    background = np.maximum(np.median(_border(frame), axis=0), 1.0)
    gains = np.clip(_BACKGROUND / background, 0.5, 4.0)
    lut = np.clip(np.arange(256)[:, None] * gains, 0, 255).astype(np.uint8)
    return cv2.LUT(frame, lut.reshape(256, 1, 3))


def scene_problem(frame: np.ndarray, cfg: VisionConfig) -> str | None:
    """Explain why the scene cannot work, for when no resistor is found. None if it looks fine."""
    if np.median(_border(frame).mean(axis=1)) < 70:
        return "too dark - add light"
    # The edge of the frame should be plain paper. Leads crossing it are fine;
    # hands, desks and walls are not.
    diff = to_lab(_border(normalise(frame))[None])[0] - _BACKGROUND_LAB
    diff[:, 0] *= cfg.shadow_weight
    cluttered = (np.sqrt((diff * diff).sum(axis=1)) > cfg.fg_threshold).mean()
    if cluttered > 0.25:
        return "background not plain - lay it on white paper filling the view"
    return None


# The paper's colour is modelled on a grid this many cells wide (see `_illumination`).
_LIGHT_GRID = 64


def _basis(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1)


def _illumination(small: np.ndarray) -> np.ndarray:
    """Fit the paper's colour across the frame as a smooth surface; returns its coefficients.

    Light is seldom even: lamps sit to one side and lenses darken the corners.
    A quadratic per channel follows both, and fitting it with the resistor,
    leads and other clutter rejected as outliers measures it from the paper
    alone. Coordinates run from -0.5 to 0.5 across the frame (see `_paper`).
    """
    h, w = small.shape[:2]
    gh = max(4, round(_LIGHT_GRID * h / w))
    cells = cv2.resize(small, (_LIGHT_GRID, gh), interpolation=cv2.INTER_AREA)
    cells = cells.reshape(-1, 3).astype(np.float32)
    v, u = np.mgrid[0:gh, 0:_LIGHT_GRID]
    basis = _basis((u.ravel() + 0.5) / _LIGHT_GRID - 0.5, (v.ravel() + 0.5) / gh - 0.5)
    keep = np.ones(len(cells), bool)
    for _ in range(4):
        coef = np.linalg.lstsq(basis[keep], cells[keep], rcond=None)[0]
        error = np.abs(cells - basis @ coef).max(axis=1)
        inliers = error < max(3 * float(np.median(error[keep])), 4.0)
        if inliers.sum() < len(cells) // 4:
            break
        keep = inliers
    return coef


def _paper(coef: np.ndarray, xs: np.ndarray, ys: np.ndarray, shape) -> np.ndarray:
    """The fitted paper colour (BGR) at pixel coordinates of a frame of `shape`."""
    h, w = shape[:2]
    return (_basis((xs + 0.5) / w - 0.5, (ys + 0.5) / h - 0.5) @ coef).astype(np.float32)


def _flatten(image: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Divide out the light so the paper becomes even neutral grey."""
    if paper.shape != image.shape:
        # The surface is smooth, so evaluating it coarsely and enlarging is enough.
        paper = cv2.resize(paper, image.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    gains = np.clip(_BACKGROUND / np.maximum(paper, 1.0), 0.5, 4.0)
    return cv2.multiply(image, gains, dtype=cv2.CV_8U)


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


# Morphology cost grows with the square of the disc size, so masks are shrunk
# until the body's radius is about this many pixels before cleaning them up.
_MORPH_RADIUS = 8


def _body_rect(mask: np.ndarray, cfg: VisionConfig):
    """Rotated rectangle around the resistor body in a foreground mask, or None."""
    h, w = mask.shape
    min_area = cfg.min_body_frac * h * w
    outline, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, outline, -1, 1, cv2.FILLED)  # fill glare holes

    # The body is the thickest part of a resistor. Measure each object's
    # thickness and drop those too thick to be one (hands, boxes, shadows).
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    max_radius = cfg.max_diameter_frac * w / 2
    keep = np.zeros(count, bool)
    radius = 0.0
    for label in np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:6] + 1:
        x, y, bw, bh, area = stats[label]
        if area < min_area / 4:
            break
        thickness = dist[y:y + bh, x:x + bw][labels[y:y + bh, x:x + bw] == label].max()
        if 3 <= thickness <= max_radius:
            keep[label] = True
            radius = max(radius, float(thickness))
    if radius == 0:
        return None
    mask = keep[labels].astype(np.uint8)

    scale = min(1.0, _MORPH_RADIUS / radius)
    if scale < 1.0:
        mask = cv2.resize(mask * 255, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = (mask > 127).astype(np.uint8)
    disc = int(round(radius * scale)) | 1

    # A band close to the paper colour (white, silver, grey) cuts the body in
    # pieces; closing rejoins them. Opening with a disc about as wide as the
    # body's radius then removes the much thinner leads.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(disc))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(disc))

    mh, mw = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, min_area * scale * scale
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < best_area:
            continue
        left, top, cw, ch = cv2.boundingRect(contour)
        if left <= 1 or top <= 1 or left + cw >= mw - 1 or top + ch >= mh - 1:
            continue  # cut off by the frame edge
        rect = cv2.minAreaRect(contour)
        (rw, rh) = rect[1]
        if not cfg.min_aspect <= max(rw, rh) / max(min(rw, rh), 1e-6) <= cfg.max_aspect:
            continue
        best, best_area = rect, area
    if best is None:
        return None
    (cx, cy), (rw, rh), angle = best
    return ((cx + 0.5) / scale - 0.5, (cy + 0.5) / scale - 0.5), (rw / scale, rh / scale), angle


def find_body(small: np.ndarray, cfg: VisionConfig):
    """Locate the body in a shrunk, light-flattened frame.

    Returns (origin, along, across) in that image's pixels, or None.
    """
    diff = to_lab(cv2.GaussianBlur(small, (5, 5), 0)) - _BACKGROUND_LAB
    diff[..., 0] *= cfg.shadow_weight
    mask = (np.sqrt((diff * diff).sum(axis=2)) > cfg.fg_threshold).astype(np.uint8)

    rect = _body_rect(mask, cfg)
    if rect is None:
        return None
    (cx, cy), (length, diameter), angle = rect
    if length < diameter:
        length, diameter, angle = diameter, length, angle + 90.0
    theta = np.deg2rad(angle)
    along = np.array([np.cos(theta), np.sin(theta)]) * length
    across = np.array([-np.sin(theta), np.cos(theta)]) * diameter
    origin = np.array([cx, cy]) - along / 2 - across / 2
    return origin, along, across


def rectify(frame: np.ndarray, origin, along, across, width: int, light=None) -> np.ndarray:
    """Crop a rectangle out of the frame as a horizontal strip `width` pixels long.

    With `light` (from `_illumination`), the light falling on it is divided out.
    """
    height = max(8, round(width * np.linalg.norm(across) / np.linalg.norm(along)))
    src = np.float32([origin, origin + along, origin + across])
    dst = np.float32([[0, 0], [width, 0], [0, height]])
    strip = cv2.warpAffine(frame, cv2.getAffineTransform(src, dst), (width, height))
    if light is None:
        return strip
    fx = (np.arange(width) + 0.5) / width
    fy = (np.arange(height) + 0.5) / height
    xy = origin + fx[None, :, None] * along + fy[:, None, None] * across
    return _flatten(strip, _paper(light, xy[..., 0], xy[..., 1], frame.shape))


def _runs(profile: np.ndarray, threshold: float) -> list[tuple[int, int, np.ndarray]]:
    """Split a colour profile into (start, end, mean colour) runs of similar colour."""
    runs = []
    start, mean, count = 0, profile[0].astype(np.float64), 1
    for x in range(1, len(profile)):
        if np.linalg.norm(profile[x] - mean) > threshold:
            runs.append((start, x, mean))
            start, mean, count = x, profile[x].astype(np.float64), 1
        else:
            count += 1
            mean = mean + (profile[x] - mean) / count
    runs.append((start, len(profile), mean))
    return runs


def _merge(runs, threshold: float):
    """Join neighbouring runs whose colours are within `threshold`."""
    merged = [runs[0]]
    for start, end, lab in runs[1:]:
        last_start, last_end, last_lab = merged[-1]
        if np.linalg.norm(lab - last_lab) < threshold:
            a, b = last_end - last_start, end - start
            merged[-1] = (last_start, end, (last_lab * a + lab * b) / (a + b))
        else:
            merged.append((start, end, lab))
    return merged


def _foreground(lab: np.ndarray, cfg: VisionConfig) -> np.ndarray:
    diff = lab - _BACKGROUND_LAB
    diff[..., 0] *= cfg.shadow_weight
    return np.sqrt((diff * diff).sum(axis=-1)) > cfg.fg_threshold


def _longest(flags: np.ndarray, around: int | None = None) -> tuple[int, int] | None:
    """(start, end) of the longest run of True, or of the run containing `around`."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))
    runs = list(zip(edges[::2], edges[1::2]))
    if around is not None:
        runs = [(s, e) for s, e in runs if s <= around < e]
    if not runs:
        return None
    s, e = max(runs, key=lambda r: r[1] - r[0])
    return int(s), int(e)


def _body_extent(lab: np.ndarray, cfg: VisionConfig):
    """Rows and columns of a strip that are resistor body: ((top, bottom), (first, last)).

    The located rectangle can take in a shadow or a stain beside the body, and
    overhang onto the leads. The body's rows are told apart by the band
    pattern running across them, which plain paper, shadows and stains lack.
    """
    h, w = lab.shape[:2]
    fg = _foreground(lab, cfg)
    middle = lab[:, w * 15 // 100:w * 85 // 100]
    fill = fg[:, w * 15 // 100:w * 85 // 100].mean(axis=1)
    # Typical difference of each row from its own median colour: high where
    # bands cross, noise level on paper.
    contrast = np.linalg.norm(middle - np.median(middle, axis=1, keepdims=True), axis=2).mean(axis=1)
    contrast = np.convolve(contrast * (fill > 0.5), np.ones(3) / 3, mode="same")
    rows = None
    if contrast.max() > 4.0:
        rows = _longest(contrast >= 0.25 * contrast.max(), around=int(np.argmax(contrast)))
    if rows is None or rows[1] - rows[0] < 3:
        # No visible band pattern: fall back to the rows the object fills.
        rows = _longest(fill > 0.6)
        if rows is None:
            return None
    top, bottom = rows
    # The body proper is where the object fills most of the rows; the leads
    # are thinner and fill only the middle ones.
    filled = np.flatnonzero(fg[top:bottom].mean(axis=0) > 0.6)
    if len(filled) < w // 4:
        return None
    return (top, bottom), (int(filled[0]), int(filled[-1]) + 1)


def _body_runs(runs, n: int, cfg: VisionConfig) -> list[np.ndarray]:
    """Which runs of a profile `n` long may be bare body: a boolean per run, for
    each plausible choice, likeliest first.

    The body is the colour that covers the most length, both ends (the end
    caps carry no paint) and the gaps; a band colour seldom does all three.
    Dense 5-band codes cover more length than the gaps between them, so the
    ends weigh in heavily. But when the end caps are lost in blur, a code that
    starts and ends with the same colour (brown ... brown) has that colour at
    both ends too, while blur breaks the true body into scraps, so the
    runner-up is kept for the decoder to judge.
    """
    labs = np.array([run[2] for run in runs])
    lengths = np.array([run[1] - run[0] for run in runs], dtype=np.float64)
    near = np.linalg.norm(labs[:, None] - labs[None], axis=2) < cfg.body_tolerance
    score = near @ lengths / n
    reach = 0.12 * n
    if runs[0][0] <= reach:
        score += END_BONUS * near[:, 0]
    if runs[-1][1] >= n - reach:
        score += END_BONUS * near[:, -1]
    choices: list[np.ndarray] = []
    for i in np.argsort(-score, kind="stable"):
        if len(choices) == 2:
            break
        if not any((near[i] == choice).all() for choice in choices):
            choices.append(near[i])
    return choices


def _body_model(profile: np.ndarray, runs, is_body: np.ndarray) -> np.ndarray:
    """The body's colour at every point of the profile.

    A straight-line fit along the length through the body runs, since light
    falling more on one end changes the body colour along it.
    """
    xs = [np.arange(s + (e - s) // 5, e - (e - s) // 5) for (s, e, _), body in zip(runs, is_body) if body]
    xs = np.concatenate(xs)
    n = len(profile)
    if len(xs) < 2 or xs.max() - xs.min() < 0.4 * n:
        return np.broadcast_to(np.median(profile[xs], axis=0), profile.shape)
    fit = np.polynomial.polynomial.polyfit(xs, profile[xs], 1)
    return np.polynomial.polynomial.polyval(np.arange(n), fit).T


def _split(contrast: np.ndarray, start: int, end: int, min_length: int) -> list[tuple[int, int]]:
    """Cut a stretch that stands out from the body into single bands.

    Two bands with a narrow gap between them blur into one stretch, but the
    gap still shows as a dip in contrast, which is where it is cut.
    """
    part = contrast[start:end]
    if len(part) >= 2 * min_length:
        left = np.maximum.accumulate(part)
        right = np.maximum.accumulate(part[::-1])[::-1]
        depth = part / np.maximum(np.minimum(left, right), 1e-6)
        inner = np.arange(min_length, len(part) - min_length)
        if len(inner):
            cut = int(inner[np.argmin(depth[inner])])
            if depth[cut] < DIP:
                return (_split(contrast, start, start + cut, min_length)
                        + _split(contrast, start + cut, end, min_length))
    return [(start, end)]


def _unmerge(profile: np.ndarray, pieces: list[tuple[int, int]], usual: float, min_length: int,
             step: float) -> list[tuple[int, int]]:
    """Cut stretches much wider than the `usual` band where two bands' colours meet.

    Two bands of different colours with no gap left between them (black then
    red, say) show no dip in contrast, only a change of colour halfway.
    """
    result = []
    for start, end in pieces:
        best, best_jump = None, step
        if end - start >= WIDE * usual:
            for cut in range(start + min_length, end - min_length + 1):
                # Compare the middles of the two halves; their edges are blurred.
                a, b = (cut - start) // 4, (end - cut) // 4
                jump = float(np.linalg.norm(np.median(profile[start + a:cut - a], axis=0)
                                            - np.median(profile[cut + b:end - b], axis=0)))
                if jump > best_jump:
                    best, best_jump = cut, jump
        result += [(start, end)] if best is None else [(start, best), (best, end)]
    return result


def _bands(inner: np.ndarray, body: np.ndarray, difference: np.ndarray, palette: Palette,
           cfg: VisionConfig, min_length: int, trim: int, length: int) -> list[Band]:
    """The stretches of a profile that stand out from the body.

    `body` is the body's colour at each point of the profile `inner`, and
    `difference` how far the profile is from it, as Lab vectors.
    """
    n = len(inner)
    contrast = np.linalg.norm(difference, axis=1)
    contrast = np.convolve(contrast, np.ones(3) / 3, mode="same")
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (contrast > cfg.body_tolerance).astype(np.int8), [0]])))
    pieces = []
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start >= min_length:
            pieces += _split(contrast, int(start), int(end), min_length)
    # The usual band width, from the clear bands rather than faint glare.
    strong = [end - start for start, end in pieces
              if contrast[start:end].max() >= WEAK * cfg.body_tolerance]
    if len(strong) >= 2:
        pieces = _unmerge(inner, pieces, float(np.median(strong)), min_length, 3 * cfg.band_threshold)

    bands = []
    for start, end in pieces:
        # Measure the middle half only: the edges are blurred into the body.
        quarter = (end - start) // 4
        band_lab = np.median(inner[start + quarter:end - quarter], axis=0)
        at_end = start <= min_length or end >= n - min_length
        part = contrast[start:end]
        peak = float(part.max())
        # Its middle, weighted by contrast: the edges of a band blur unevenly
        # into the body, so the midpoint of its extent wanders.
        middle = float((np.arange(start, end) * part).sum() / max(part.sum(), 1e-6) + 0.5)
        drop_cost = peak if peak < WEAK * cfg.body_tolerance else math.inf
        if at_end:
            drop_cost = min(drop_cost, END_DROP_PENALTY)
        bands.append(Band(band_lab, (start + trim) / length, (end + trim) / length,
                          palette.rank(band_lab, body[(start + end) // 2]), at_end, drop_cost,
                          (middle + trim) / length))
    return bands


def find_bands(strip: np.ndarray, palette: Palette, cfg: VisionConfig):
    """Measure the body colour and the bands of a straightened, light-flattened strip.

    Returns ((top, bottom) rows, (first, last) columns of the body, choices),
    where each choice is a (body colour, bands) reading of it, likeliest first;
    or None.
    """
    lab = to_lab(strip)
    extent = _body_extent(lab, cfg)
    if extent is None:
        return None
    (top, bottom), (first, last) = extent
    length = last - first

    # Median over the middle half of the body's rows ignores the glare line
    # and the shaded edges.
    quarter = (bottom - top) // 4
    profile = np.median(lab[top + quarter:max(bottom - quarter, top + quarter + 1), first:last], axis=0)
    profile = cv2.medianBlur(np.ascontiguousarray(profile[None]), 5)[0]

    trim = int(length * cfg.end_trim)
    inner = profile[trim:length - trim]
    min_length = max(2, int(strip.shape[1] * cfg.min_band_frac))

    # First pass: rough runs of constant colour, to find the body colour.
    # Second pass: bands are wherever the profile stands out from it.
    runs = _runs(inner, cfg.band_threshold)
    runs = [run for run in runs if run[1] - run[0] >= min_length]
    if not runs:
        return None
    runs = _merge(runs, cfg.band_threshold)
    choices = []
    for is_body in _body_runs(runs, len(inner), cfg):
        body = _body_model(inner, runs, is_body)
        difference = inner - body
        # Glare and the bulging end caps make patches of bare body brighter
        # but barely change its hue, so being lighter than the body counts
        # for less. Being darker does not: that is how dark bands stand out.
        difference[:, 0] = np.where(difference[:, 0] > 0, difference[:, 0] * cfg.shadow_weight,
                                    difference[:, 0])
        bands = _bands(inner, body, difference, palette, cfg, min_length, trim, length)
        choices.append((np.median(body, axis=0), bands))
    return (top, bottom), (first, last), choices


# The located rectangle may be turned by up to this many degrees to line it up
# with the bands (see `_align`).
_MAX_TURN = 40


def _turn(vector: np.ndarray, degrees: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def _align(frame: np.ndarray, center: np.ndarray, along: np.ndarray, across: np.ndarray):
    """Turn the rectangle about its centre so the bands run straight across it.

    A stain or shadow touching the body skews the rectangle fitted around
    both. The band edges only stay sharp in a column-by-column profile when
    the rectangle follows the body's axis, so the angle that makes them
    sharpest wins; the fitted angle stands unless another is clearly better.
    (Sharp steps, not overall variation: a rectangle turned across the body
    also varies from end to end, but smoothly.)
    """
    width = 96

    def sharpness(degrees: float) -> float:
        a, c = _turn(along, degrees), _turn(across, degrees)
        strip = rectify(frame, center - a / 2 - c / 2, a, c, width).astype(np.float32)
        profile = strip[:, width * 15 // 100:width * 85 // 100].mean(axis=0)
        return float((np.diff(profile, axis=0) ** 2).sum())

    scores = {0.0: sharpness(0.0)}
    for degrees in range(-_MAX_TURN, _MAX_TURN + 1, 5):
        scores.setdefault(float(degrees), sharpness(degrees))
    coarse = max(scores, key=scores.get)
    for degrees in (coarse - 2, coarse - 1, coarse + 1, coarse + 2):
        scores.setdefault(degrees, sharpness(degrees))
    best = max(scores, key=scores.get)
    if scores[best] < 1.05 * scores[0.0]:
        best = 0.0
    return _turn(along, best), _turn(across, best)


def detect(frame: np.ndarray, palette: Palette, cfg: VisionConfig) -> Detection | None:
    """Find the resistor in a BGR frame and measure its bands. None if there is none."""
    scale = min(1.0, cfg.work_width / frame.shape[1])
    small = frame
    if scale < 1.0:
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    light = _illumination(small)
    h, w = small.shape[:2]
    v, u = np.mgrid[0:h:8, 0:w:8] + 3.5  # centres of 8x8 cells
    body = find_body(_flatten(small, _paper(light, u, v, small.shape)), cfg)
    if body is None:
        return None
    origin, along, across = (np.asarray(vector) / scale for vector in body)
    center = origin + along / 2 + across / 2
    along, across = _align(frame, center, along, across)

    # Cut generously, then find the body's true outline inside.
    along = along * (1 + 2 * _MARGIN_ALONG)
    across = across * (1 + 2 * _MARGIN_ACROSS)
    origin = center - along / 2 - across / 2
    strip = rectify(frame, origin, along, across, cfg.strip_width, light)
    found = find_bands(strip, palette, cfg)
    if found is None:
        return None
    (top, bottom), (first, last), choices = found
    # Shrink the rectangle to the body proper so band positions line up.
    height, width = strip.shape[:2]
    origin = origin + along * first / width + across * top / height
    along = along * (last - first) / width
    across = across * (bottom - top) / height
    crop = strip[top:bottom, first:last]
    return _likeliest([Detection(origin, along, across, crop, body_lab, bands) for body_lab, bands in choices])


def _likeliest(detections: list[Detection]) -> Detection:
    """Of readings that differ in what they take for the body, the one whose
    bands make the most convincing code. The first unless another reads clearly better.
    """
    best, best_cost = detections[0], math.inf
    if len(detections) == 1:
        return best
    for rank, detection in enumerate(detections):
        reading = read(detection)
        if reading is None:
            continue
        cost = reading.score / len(reading.colors) + rank * RUNNER_UP_PENALTY
        if cost < best_cost:
            best, best_cost = detection, cost
    return best


def read(detection: Detection | None) -> Reading | None:
    """Decode a detection's bands into a resistance."""
    if detection is None or not detection.bands:
        return None
    bands = detection.bands
    if len(bands) == 1 and not 0.3 < bands[0].center < 0.7:
        return None  # a zero-ohm jumper's single band is in the middle
    # A segment touching either end of the body may be the end cap rather
    # than a band, and a faint one may be glare on bare body, so also try
    # reading without them.
    droppable = sorted((i for i, band in enumerate(bands) if band.drop_cost < math.inf),
                       key=lambda i: bands[i].drop_cost)[:MAX_DROPPED]
    options = []
    for count in range(len(droppable) + 1):
        for dropped in itertools.combinations(droppable, count):
            subset = [band for i, band in enumerate(bands) if i not in dropped]
            if not count or len(subset) >= 3:
                options.append((subset, sum(bands[i].drop_cost for i in dropped)))

    readings = []
    for subset, penalty in options:
        reading = interpret([b.candidates for b in subset], [b.center for b in subset])
        if reading is not None:
            readings.append(replace(reading, score=reading.score + penalty))
    if not readings:
        return None
    readings.sort(key=lambda r: r.score)
    best = readings[0]
    rivals = [r.score - best.score for r in readings[1:] if r.ohms != best.ohms]
    return replace(best, margin=min([best.margin, *rivals]))


def annotate(
    frame: np.ndarray,
    detection: Detection | None,
    reading: Reading | None,
    hint: str | None = None,
    trusted: bool = True,
) -> np.ndarray:
    """Draw the detection and reading onto a copy of the frame.

    `hint` replaces the "no resistor" label, e.g. with `scene_problem`'s advice.
    An untrusted reading (see `Sorter.trusted`) is shown with a question mark.
    """
    out = frame.copy()
    unit = frame.shape[1] / 640
    label = hint or "no resistor - place one in the middle, ~6 cm from the lens"
    if detection is not None:
        cv2.polylines(out, [detection.corners.astype(np.int32)], True, (0, 255, 0), max(1, round(unit)))
        # Dots show what each segment looks like; the label shows the decoded
        # reading, which may use runner-up colours or skip an end segment.
        for band in detection.bands:
            center = tuple(int(v) for v in detection.point(band.center, -0.6))
            cv2.circle(out, center, round(7 * unit), DEFAULT_RGB[band.color][::-1], cv2.FILLED)
            cv2.circle(out, center, round(7 * unit), (0, 0, 0), max(1, round(unit)))
        label = f"{len(detection.bands)} bands, unreadable"
        if reading is not None:
            label = f"{reading.text(ascii=True)}{'' if trusted else '?'}  ({' '.join(reading.colors)})"
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.8 * unit
    lines = [""]
    for word in label.split():
        line = f"{lines[-1]} {word}".strip()
        if lines[-1] and cv2.getTextSize(line, font, scale, 2)[0][0] > out.shape[1] - 20 * unit:
            lines.append(word)
        else:
            lines[-1] = line
    for row, line in enumerate(lines):
        position = (round(10 * unit), round((28 + 30 * row) * unit))
        for color, thickness in (((0, 0, 0), 5), ((255, 255, 255), 2)):
            cv2.putText(out, line, position, font, scale, color,
                        max(1, round(thickness * unit / 1.5)), cv2.LINE_AA)
    return out
