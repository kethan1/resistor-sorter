import cv2
import numpy as np

from resistor_sorter.colors import DEFAULT_RGB, Palette, rgb_to_lab


def mix(rgb_a, rgb_b, share):
    """sRGB colour seen when `share` of the light comes from a, the rest from b."""
    linear = [cv2.cvtColor(np.float32([[rgb[::-1]]]) / 255, cv2.COLOR_BGR2LAB) for rgb in (rgb_a, rgb_b)]
    a, b = (cv2.cvtColor(lab, cv2.COLOR_Lab2LBGR) for lab in linear)
    return cv2.cvtColor(share * a + (1 - share) * b, cv2.COLOR_LBGR2Lab)[0, 0]


def test_rank_without_body_is_nearest_reference():
    palette = Palette()
    for color, rgb in DEFAULT_RGB.items():
        assert palette.rank(rgb_to_lab(rgb))[0] == (color, 0.0)


def test_rank_allows_for_body_colour_bleeding_into_a_band():
    palette = Palette()
    body_rgb = (70, 165, 195)  # a metal film resistor's blue
    seen = mix(DEFAULT_RGB["brown"], body_rgb, 0.75)
    assert palette.rank(seen)[0][0] != "brown"  # it looks grey-blue on its own
    color, distance = palette.rank(seen, rgb_to_lab(body_rgb))[0]
    assert color == "brown" and distance < 5


def test_rank_does_not_let_the_body_explain_everything():
    palette = Palette()
    body = rgb_to_lab((70, 165, 195))
    # A plain grey band on a blue body is still grey, not black or white
    # mixed with blue.
    assert palette.rank(rgb_to_lab(DEFAULT_RGB["grey"]), body)[0] == ("grey", 0.0)
