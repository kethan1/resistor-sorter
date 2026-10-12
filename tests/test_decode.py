import math

import pytest

from resistor_sorter.decode import E96, decode, format_ohms, interpret


def sure(*colors):
    """Candidate lists for bands whose colour is unambiguous."""
    return [[(color, 1.0)] for color in colors]


@pytest.mark.parametrize("colors, ohms, tolerance", [
    (("yellow", "violet", "red", "gold"), 4700, 5),
    (("brown", "black", "black", "gold"), 10, 5),
    (("red", "red", "gold", "gold"), 2.2, 5),
    (("yellow", "violet", "silver", "silver"), 0.47, 10),
    (("brown", "black", "green"), 1e6, 20),
    (("brown", "black", "black", "brown", "brown"), 1000, 1),
    (("yellow", "white", "white", "red", "brown"), 49900, 1),
    (("orange", "orange", "black", "black", "brown", "red"), 330, 1),
])
def test_decode(colors, ohms, tolerance):
    reading = decode(colors)
    assert reading.ohms == ohms
    assert reading.tolerance == tolerance


def test_decode_extras():
    assert decode(["black"]).ohms == 0
    assert decode(["orange", "orange", "black", "black", "brown", "red"]).tempco == 50
    assert decode(["brown", "black", "red", "gold"]).standard
    assert not decode(["brown", "yellow", "red", "gold"]).standard  # 1.4k is not E24


@pytest.mark.parametrize("colors", [
    ("black", "brown", "red", "gold"),  # leading zero
    ("gold", "red", "violet", "yellow"),  # backwards
    ("brown", "black", "red", "white"),  # white is not a tolerance
    ("brown", "gold", "red", "gold"),  # gold is not a digit
    ("brown", "black"),
    ("red",),
])
def test_decode_rejects_invalid(colors):
    assert decode(colors) is None


def test_e96():
    assert len(E96) == 96
    assert {100, 102, 499, 976} <= E96


@pytest.mark.parametrize("ohms, text", [
    (0.47, "0.47 Ω"), (330, "330 Ω"), (4700, "4.7 kΩ"), (49900, "49.9 kΩ"),
    (1e6, "1 MΩ"), (2.2e9, "2.2 GΩ"),
])
def test_format(ohms, text):
    assert format_ohms(ohms) == text


def test_interpret_reads_from_either_end():
    forward = interpret(sure("yellow", "violet", "red", "gold"))
    backward = interpret(sure("gold", "red", "violet", "yellow"))
    assert forward.ohms == backward.ohms == 4700
    assert not forward.flipped and backward.flipped
    assert forward.margin == math.inf


def test_interpret_uses_spacing_to_pick_direction():
    # 1 kΩ 1% reads as a valid 110 Ω 1% backwards; only the gap tells them apart.
    bands = sure("brown", "black", "black", "brown", "brown")
    assert interpret(bands, [0.13, 0.26, 0.39, 0.52, 0.83]).ohms == 1000
    assert interpret(bands, [0.17, 0.48, 0.61, 0.74, 0.87]).ohms == 110
    # Three bands sit towards the end that is read first.
    bands = sure("yellow", "violet", "red")
    assert interpret(bands, [0.16, 0.32, 0.48]).ohms == 4700
    assert interpret(bands, [0.52, 0.68, 0.84]).ohms == 270000


def test_interpret_falls_back_to_runner_up_colour():
    # Silver cannot be a digit, so the second band must be the runner-up, grey.
    bands = [[("brown", 3.0)], [("silver", 5.0), ("grey", 8.0)], [("red", 2.0)], [("gold", 4.0)]]
    reading = interpret(bands)
    assert reading.colors == ("brown", "grey", "red", "gold")
    assert reading.ohms == 1800


def test_interpret_prefers_standard_values():
    # Red vs orange is a toss-up on colour alone; 4.3k is E24 and 4.2k is not.
    bands = [[("yellow", 2.0)], [("red", 6.0), ("orange", 7.0)], [("red", 2.0)], [("gold", 3.0)]]
    reading = interpret(bands)
    assert reading.ohms == 4300
    assert reading.margin == pytest.approx(9.0)


def test_interpret_gives_up_on_nonsense():
    assert interpret([]) is None
    assert interpret(sure("gold", "gold", "gold", "gold")) is None
    assert interpret(sure("red", "red")) is None


def test_interpret_doubts_direction_in_proportion_to_spacing():
    # 220 Ω 1% reads as a valid 10 kΩ 2% backwards. On many metal film
    # resistors the tolerance band is only a little further out.
    bands = sure("red", "red", "black", "black", "brown")
    assert interpret(bands, [0.19, 0.32, 0.46, 0.62, 0.81]).ohms == 220
    assert interpret(bands, [0.19, 0.38, 0.54, 0.68, 0.81]).ohms == 10e3
    # Evenly spaced bands give no clue, so neither reading wins clearly.
    even = interpret(bands, [0.2, 0.35, 0.5, 0.65, 0.8])
    assert even.margin < 1


def test_interpret_wants_more_evidence_for_three_bands():
    # Brown black red (1 kΩ 20%) or, with a weak extra band, 10 Ω 1%:
    # a three-band code is the rarer explanation.
    bands = [[("brown", 2.0)], [("black", 2.0)], [("black", 2.0)], [("brown", 6.0)]]
    assert len(interpret(bands).colors) == 4


def test_interpret_prefers_usual_tolerances():
    # A 5-band 220 Ω 1% with its two red bands blurred into one leaves
    # brown black brown red: valid as 100 Ω 2%, but 4-band codes end in gold
    # or silver, so it is not trusted over a near rival.
    bands = [[("brown", 3.0)], [("black", 3.0)], [("brown", 3.0), ("red", 6.0)], [("red", 3.0), ("gold", 9.0)]]
    reading = interpret(bands)
    assert reading.score >= 12 + 6
