import cv2
import numpy as np
import pytest
from synth import DENSE_LAYOUTS, render

from resistor_sorter.colors import Palette
from resistor_sorter.config import VisionConfig
from resistor_sorter.vision import Band, Detection, annotate, detect, read

CFG = VisionConfig()
PALETTE = Palette()

RESISTORS = [
    (("yellow", "violet", "red", "gold"), "beige", 4700),
    (("brown", "black", "orange", "gold"), "beige", 10e3),
    (("red", "red", "brown", "gold"), "beige", 220),
    (("green", "blue", "yellow", "gold"), "beige", 560e3),
    (("orange", "white", "black", "silver"), "beige", 39),
    (("grey", "red", "red", "gold"), "beige", 8200),
    (("brown", "black", "green"), "beige", 1e6),
    (("brown", "black", "black", "brown", "brown"), "blue", 1000),
    (("yellow", "violet", "black", "red", "brown"), "blue", 47e3),
    (("orange", "orange", "black", "black", "brown"), "blue", 330),
    (("red", "yellow", "white", "orange", "red"), "blue", 249e3),
]


def scan(frame):
    detection = detect(frame, PALETTE, CFG)
    return detection, read(detection)


@pytest.mark.parametrize("colors, body, ohms", RESISTORS)
@pytest.mark.parametrize("angle", [0, 35, 90, 160, 215])
def test_reads_value_at_any_angle(colors, body, ohms, angle):
    detection, reading = scan(render(colors, body=body, angle=angle))
    assert detection is not None
    assert [b.color for b in detection.bands] in (list(colors), list(colors[::-1]))
    assert reading.ohms == ohms
    assert reading.colors == colors
    assert reading.worst <= 15 and reading.margin >= 4  # the sorter would trust it


@pytest.mark.parametrize("colors, body, ohms", RESISTORS)
@pytest.mark.parametrize("light", [(1.0, 0.9, 0.7), (0.8, 0.9, 1.0), (0.7, 0.7, 0.7)])
def test_reads_value_under_tinted_or_dim_light(colors, body, ohms, light):
    _, reading = scan(render(colors, body=body, angle=20, light=light))
    assert reading is not None and reading.ohms == ohms


@pytest.mark.parametrize("colors, body, ohms", RESISTORS)
@pytest.mark.parametrize("length, offset", [(110, (0, 0)), (180, (-300, 200)), (420, (150, -120))])
def test_reads_value_at_other_sizes_and_positions(colors, body, ohms, length, offset):
    _, reading = scan(render(colors, body=body, angle=-15, length=length, offset=offset))
    assert reading is not None and reading.ohms == ohms


@pytest.mark.parametrize("seed", range(8))
def test_tolerates_paint_variation(seed):
    colors = ("yellow", "violet", "red", "gold")
    _, reading = scan(render(colors, angle=10 * seed, jitter=12, seed=seed))
    assert reading is not None and reading.ohms == 4700


def test_band_positions_are_reported():
    detection, _ = scan(render(("yellow", "violet", "red", "gold"), angle=30))
    centers = [b.center for b in detection.bands]
    assert 0 < centers[0] < centers[-1] < 1
    # Drawn at 0.16, 0.32, 0.48, 0.80 of the body; the strip may run either way.
    gaps = np.diff(centers)
    if gaps[0] > gaps[-1]:
        gaps = gaps[::-1]
    assert gaps[0] == pytest.approx(gaps[1], rel=0.1)
    assert gaps[2] == pytest.approx(2 * gaps[1], rel=0.15)


def test_empty_scene_has_no_resistor():
    blank = np.full((972, 1296, 3), 200, np.uint8)
    assert detect(blank, PALETTE, CFG) is None
    assert read(None) is None


def test_resistor_cut_off_by_frame_edge_is_ignored():
    frame = render(("yellow", "violet", "red", "gold"), offset=(600, 0))
    assert detect(frame, PALETTE, CFG) is None


def test_annotate_draws_on_a_copy():
    frame = render(("yellow", "violet", "red", "gold"), angle=30)
    detection, reading = scan(frame)
    out = annotate(frame, detection, reading)
    assert out.shape == frame.shape and not np.array_equal(out, frame)
    assert annotate(frame, None, None).shape == frame.shape


def test_scene_problems_are_explained():
    from resistor_sorter.vision import scene_problem

    assert scene_problem(render(("yellow", "violet", "red", "gold")), CFG) is None
    assert scene_problem(np.full((972, 1296, 3), 20, np.uint8), CFG) == "too dark - add light"
    cluttered = np.random.default_rng(0).integers(0, 255, (972, 1296, 3), dtype=np.uint8)
    assert "not plain" in scene_problem(cluttered, CFG)


# What a real camera adds: thicker end caps, focus blur and JPEG compression.
REAL = dict(caps=True, blur=2.2, jpeg=75)


@pytest.mark.parametrize("angle", [0, 40, 95, 150])
def test_reads_dense_dark_bands_on_blue_body(angle):
    # Five dark bands packed tightly on a blue body cover more of its length
    # than the bare body between them; the body colour is the one at the ends.
    colors = ("brown", "black", "black", "red", "brown")
    frame = render(colors, body="blue", angle=angle, length=150, layout=DENSE_LAYOUTS[5],
                   band_width=0.08, **REAL)
    detection, reading = scan(frame)
    assert [b.color for b in detection.bands] in (list(colors), list(colors[::-1]))
    assert reading.ohms == 10e3


@pytest.mark.parametrize("colors, body, ohms", RESISTORS)
def test_reads_value_under_uneven_light(colors, body, ohms):
    # A lamp off to one side, dark corners and the resistor's own shadow.
    frame = render(colors, body=body, angle=25, offset=(250, 150), vignette=0.35,
                   gradient=(0.35, -0.3), shadow=(10, 8, 0.4), **REAL)
    _, reading = scan(frame)
    assert reading is not None and reading.ohms == ohms


def test_stain_touching_the_body_does_not_skew_it():
    frame = render(("yellow", "violet", "red", "gold"), **REAL).astype(np.float32)
    # A brownish stain on the paper running into the body from above-left.
    stain = np.zeros(frame.shape[:2], np.float32)
    cv2.ellipse(stain, (560, 405), (110, 40), -20, 0, 360, 1.0, cv2.FILLED)
    stain = cv2.GaussianBlur(stain, (0, 0), 8)
    paper = (np.abs(frame - 236) < 15).all(axis=2)
    tint = 1 - stain[..., None] * (1 - np.float32([0.7, 0.82, 0.92]))
    frame = np.where(paper[..., None], frame * tint, frame).astype(np.uint8)
    detection, reading = scan(frame)
    assert reading.ohms == 4700
    angle = np.degrees(np.arctan2(detection.along[1], detection.along[0])) % 180
    assert min(angle, 180 - angle) < 4


def _band(color, center, **kw):
    return Band(np.zeros(3), center - 0.04, center + 0.04, [(color, 5.0)], **kw)


def _detection(*bands):
    return Detection(np.zeros(2), np.array([100.0, 0.0]), np.array([0.0, 40.0]),
                     np.zeros((4, 4, 3), np.uint8), np.zeros(3), list(bands))


def test_lone_band_left_after_dropping_end_segments_is_not_a_jumper():
    end = dict(at_end=True, drop_cost=15.0)
    assert read(_detection(_band("silver", 0.06, **end), _band("black", 0.92, **end))) is None
    assert read(_detection(_band("black", 0.9))) is None
    assert read(_detection(_band("black", 0.5))).ohms == 0


def test_faint_segment_can_be_read_as_bare_body():
    # Glare on the body between two bands shows up as a faint extra segment.
    bands = [_band(c, x) for c, x in zip(("yellow", "violet", "red"), (0.16, 0.32, 0.48))]
    bands += [_band("silver", 0.62, drop_cost=14.0), _band("gold", 0.8)]
    assert read(_detection(*bands)).ohms == 4700
