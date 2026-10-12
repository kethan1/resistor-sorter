"""Draws synthetic photos of resistors for testing the vision pipeline."""

from __future__ import annotations

import cv2
import numpy as np

from resistor_sorter.colors import DEFAULT_RGB

BODY_RGB = {"beige": (214, 186, 140), "blue": (110, 170, 205)}

# Band centres along the body, as fractions of its length.
LAYOUTS = {
    3: (0.16, 0.32, 0.48),
    4: (0.16, 0.32, 0.48, 0.80),
    5: (0.13, 0.26, 0.39, 0.52, 0.83),
    6: (0.11, 0.22, 0.33, 0.44, 0.66, 0.86),
}
# Tightly packed bands, as on small 1% metal film resistors: the gaps of bare
# body between the bands are narrower than the bands themselves (use a band
# width of about 0.08).
DENSE_LAYOUTS = {
    4: (0.26, 0.39, 0.52, 0.76),
    5: (0.24, 0.36, 0.48, 0.60, 0.78),
}


def render(
    colors,
    *,
    body="beige",
    angle=0.0,
    size=(1296, 972),
    length=260,
    light=(1.0, 1.0, 1.0),
    offset=(0, 0),
    jitter=0.0,
    seed=0,
    layout=None,
    band_width=None,
    caps=False,
    blur=1.5,
    vignette=0.0,
    gradient=(0.0, 0.0),
    shadow=None,
    jpeg=None,
):
    """A BGR frame showing one resistor with the given bands on white paper.

    `light` is an RGB tint applied to the whole scene; `jitter` randomly shifts
    each paint colour by up to that many sRGB levels. The rest make the picture
    harder, like a real camera's:

    - `layout`, `band_width`: band centres and width as fractions of the length.
    - `caps`: give the body the thicker end caps of a real resistor.
    - `blur`: Gaussian sigma in pixels, for focus.
    - `vignette`: how much darker the frame corners are than the centre (0..1).
    - `gradient`: (x, y) change in brightness across the frame, e.g. a lamp at one side.
    - `shadow`: (dx, dy, darkness) of a soft shadow the body casts on the paper.
    - `jpeg`: compress at this quality.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.full((h, w, 3), 236, np.float32)
    cx, cy = w // 2 + offset[0], h // 2 + offset[1]
    dia = round(length / 2.7)
    x0, x1 = cx - length // 2, cx + length // 2

    def paint(rgb):
        shift = rng.uniform(-jitter, jitter, 3)
        return tuple(float(v) for v in np.clip(np.array(rgb[::-1]) + shift[::-1], 0, 255))

    # Body outline: rounded ends, optionally swelling into end caps.
    mask = np.zeros((h, w), np.uint8)
    cap_dia = round(dia * 1.2) if caps else dia
    cap_len = round(length * 0.2) if caps else dia // 2
    cv2.rectangle(mask, (x0 + dia // 2, cy - dia // 2), (x1 - dia // 2, cy + dia // 2), 1, cv2.FILLED)
    for ex, inner in ((x0 + cap_dia // 2, x0 + cap_len), (x1 - cap_dia // 2, x1 - cap_len)):
        cv2.ellipse(mask, (ex, cy), (cap_dia // 2, cap_dia // 2), 0, 0, 360, 1, cv2.FILLED)
        cv2.rectangle(mask, (min(ex, inner), cy - cap_dia // 2), (max(ex, inner), cy + cap_dia // 2),
                      1, cv2.FILLED)
    y0, y1 = cy - cap_dia // 2, cy + cap_dia // 2

    if shadow is not None:
        dx, dy, darkness = shadow
        moved = cv2.warpAffine(mask.astype(np.float32), np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
        soft = cv2.GaussianBlur(moved, (0, 0), dia * 0.3)
        img *= 1 - darkness * soft[..., None]

    # Leads run off both edges of the frame.
    cv2.line(img, (0, cy), (w, cy), (150, 150, 150), max(2, round(dia * 0.22)))
    body_color = paint(BODY_RGB[body])
    img[mask > 0] = body_color
    width = band_width or (0.085 if len(colors) <= 4 else 0.07)
    for color, center in zip(colors, layout or LAYOUTS[len(colors)]):
        bx = x0 + center * length
        left, right = round(bx - width * length / 2), round(bx + width * length / 2) + 1
        band = mask[:, left:right] > 0
        img[:, left:right][band] = paint(DEFAULT_RGB[color])

    # Cylinder shading: darker towards the edges, with a glare line off-centre.
    region = img[y0:y1 + 1, x0:x1 + 1]
    inside = mask[y0:y1 + 1, x0:x1 + 1] > 0
    # Distance from the axis as a fraction of the local radius, column by column.
    radius = np.maximum(inside.sum(axis=0) / 2, 1)
    t = (np.arange(y0, y1 + 1)[:, None] - cy) / radius[None]
    shade = 0.78 + 0.22 * np.sqrt(np.clip(1 - t * t, 0, 1))
    glare = np.exp(-((t + 0.35) / 0.12) ** 2)
    lit = region * shade[..., None]
    lit += (255 - lit) * 0.6 * glare[..., None]
    region[:] = np.where(inside[..., None], lit, region)

    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (w, h), borderValue=(236, 236, 236))
    img = cv2.GaussianBlur(img, (0, 0), blur)
    img *= np.array(light[::-1], np.float32)
    if vignette or any(gradient):
        u = (np.arange(w) / w - 0.5)[None]
        v = (np.arange(h) / h - 0.5)[:, None]
        field = (1 - vignette * 2 * (u * u + v * v)) * (1 + gradient[0] * u + gradient[1] * v)
        img *= field[..., None].astype(np.float32)
    img += rng.normal(0, 2.0, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if jpeg is not None:
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], cv2.IMREAD_COLOR)
    return img
