import json

import cv2
import pytest
from synth import render

from resistor_sorter.cli import _outcome, main
from resistor_sorter.config import Bin, Config, SorterConfig, load_config
from resistor_sorter.decode import decode
from resistor_sorter.sorter import Event, Sorter, pick_bin

READING = decode(["yellow", "violet", "red", "gold"])  # 4.7 kΩ: bin "1 kΩ – 10 kΩ"
SAME_BIN = decode(["brown", "black", "red", "gold"])  # 1 kΩ: the same bin
OTHER = decode(["yellow", "violet", "orange", "gold"])  # 47 kΩ: bin "10 kΩ – 100 kΩ"
ABSENT = "absent"  # nothing under the camera (None is a resistor that could not be read)

FPS = 8  # frame times are then multiples of 1/8, exact in floating point


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_sorter(**overrides):
    settings = {"window": 1.0, "min_readings": 3, "min_agreement": 0.75, "reject_after": 3.0, "clear_after": 0.25}
    clock = Clock()
    return Sorter(SorterConfig(**{**settings, **overrides}), Config().bins, clock), clock


def feed(sorter, clock, frames):
    """One step per frame at FPS; each frame is a reading, None (unreadable) or ABSENT."""
    events = []
    for frame in frames:
        clock.now = round(clock.now * FPS + 1) / FPS
        events.append(sorter.step(frame is not ABSENT, None if frame is ABSENT else frame))
    return events


# With window = 1.0 the first chance to decide is the 9th frame (1 second after the first).


def test_decides_from_a_window_of_readings_then_waits_for_removal():
    sorter, clock = make_sorter()
    events = feed(sorter, clock, [READING] * 9)
    assert events[:8] == [None] * 8
    event = events[8]
    assert event.reading.ohms == 4700 and event.bin.name == "1 kΩ – 10 kΩ"
    assert (event.agreed, event.votes) == (9, 9) and event.tally == [(READING, 9)]
    # Same resistor still lying there: nothing more happens.
    assert feed(sorter, clock, [READING] * 8) == [None] * 8
    # Removed, then a new one placed.
    assert feed(sorter, clock, [ABSENT] * 3) == [None] * 3
    event = feed(sorter, clock, [OTHER] * 9)[-1]
    assert event.reading.ohms == 47000 and event.bin.name == "10 kΩ – 100 kΩ"


def test_a_few_stray_readings_are_outvoted():
    sorter, clock = make_sorter()
    events = feed(sorter, clock, [READING] * 4 + [OTHER, None, OTHER] + [READING] * 3)
    assert events[:9] == [None] * 9  # the latest three readings did not all agree yet
    event = events[9]
    assert event.reading.ohms == 4700 and event.bin.name == "1 kΩ – 10 kΩ"
    assert (event.agreed, event.votes) == (6, 8)  # exactly min_agreement
    assert event.tally == [(READING, 6), (OTHER, 2)]


def test_waits_until_the_latest_readings_agree_with_the_rest():
    sorter, clock = make_sorter()
    events = feed(sorter, clock, [READING] * 7 + [OTHER] * 2 + [READING] * 3)
    assert events[:11] == [None] * 11  # 7 of 9 agree, but the latest readings had wandered off
    event = events[11]
    assert event.reading.ohms == 4700 and (event.agreed, event.votes) == (7, 9)


def test_values_in_the_same_bin_count_together():
    sorter, clock = make_sorter()
    event = feed(sorter, clock, [READING] * 5 + [SAME_BIN] * 4)[-1]
    assert event.bin.name == "1 kΩ – 10 kΩ" and (event.agreed, event.votes) == (9, 9)
    assert event.reading.ohms == 4700  # the value read most often
    assert event.tally == [(READING, 5), (SAME_BIN, 4)]


def test_disagreement_is_rejected_after_a_while():
    sorter, clock = make_sorter()
    events = feed(sorter, clock, [READING, OTHER] * 13)
    assert events[:24] == [None] * 24
    event = events[24]  # 3 seconds after the first frame
    assert event.reading is None and event.bin is None
    assert event.tally == [(READING, 5), (OTHER, 4)] and event.votes == 9  # the last second's worth
    assert events[25] is None


def test_a_bad_start_is_forgotten():
    sorter, clock = make_sorter()
    events = feed(sorter, clock, [OTHER] * 3 + [READING] * 7)
    assert events[8] is None  # 6 of 9 agree: not enough
    event = events[9]  # the oldest stray reading has dropped out of the window
    assert event.reading.ohms == 4700 and (event.agreed, event.votes) == (7, 9)


def test_needs_enough_readings():
    sorter, clock = make_sorter()
    events = feed(sorter, clock, [None] * 8 + [READING] * 3)
    assert events[:10] == [None] * 10
    assert events[10].reading.ohms == 4700 and events[10].votes == 3


def test_rejects_when_never_readable():
    sorter, clock = make_sorter()
    events = feed(sorter, clock, [None] * 26)
    assert events[:24] == [None] * 24
    assert events[24] == Event(None, None) and events[25] is None


def test_uncertain_readings_are_not_trusted():
    from dataclasses import replace

    for shaky in (replace(READING, margin=1.0), replace(READING, worst=30.0)):
        sorter, clock = make_sorter(min_margin=4.0, max_color_distance=15.0)
        events = feed(sorter, clock, [shaky] * 25)
        assert events[24] == Event(None, None)


def test_a_flickering_detection_does_not_reset():
    sorter, clock = make_sorter()
    events = feed(sorter, clock, [READING] * 4 + [ABSENT] + [READING] * 4)
    assert events[8] is not None and events[8].votes == 8


def test_only_a_proper_absence_ends_a_resistor():
    sorter, clock = make_sorter()
    assert feed(sorter, clock, [READING] * 9)[-1] is not None
    assert feed(sorter, clock, [ABSENT] * 2 + [READING] * 9) == [None] * 11  # too short a gap
    assert feed(sorter, clock, [ABSENT] * 3 + [READING] * 9)[-1] is not None


def test_pick_bin():
    bins = [Bin("common", values=[1e3, 10e3]), Bin("low", max=1e3), Bin("high", min=1e3)]
    assert pick_bin(bins, 1e3).name == "common"
    assert pick_bin(bins, 470).name == "low"
    assert pick_bin(bins, 4700).name == "high"
    assert pick_bin([Bin("low", max=1e3)], 4700) is None


def test_outcome_lines():
    bins = Config().bins
    accepted = Event(READING, bins[2], [(READING, 8), (OTHER, 2)], 8, 10)
    assert _outcome(accepted, None) == "4.7 kΩ ±5%  [yellow violet red gold]  -> 1 kΩ – 10 kΩ  (8 of 10 readings)"
    homeless = Event(READING, None, [(READING, 3)], 3, 3)
    assert _outcome(homeless, None).endswith("-> reject (no bin for this value; 3 of 3 readings)")
    undecided = Event(None, None, [(READING, 5), (OTHER, 4)], 0, 9)
    assert _outcome(undecided, None) == "undecided (4.7 kΩ ±5% x5, 47 kΩ ±5% x4)  -> reject"
    assert _outcome(Event(None, None), None) == "unreadable (no resistor found)  -> reject"


def test_config_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[sorter]\nwindow = 9\nreject_after = 10\n[[bin]]\nname = "all"\nangle = 10\n')
    cfg = load_config(path)
    assert cfg.sorter.window == 9
    assert [b.name for b in cfg.bins] == ["all"] and cfg.bins[0].accepts(1e6)
    for text, complaint in [
        ("[sorter]\nwindoe = 9\n", "windoe"),
        ("[sorter]\nmin_agreement = 75\n", "min_agreement"),
        ("[sorter]\nstable_readings = 0\n", "stable_readings"),
        ("[sorter]\nwindow = 9\n", "reject_after"),
        ('[servo]\nenabled = "sometimes"\n', "enabled"),
        ('[servo]\nmode = "flap"\n', "mode"),
    ]:
        path.write_text(text)
        with pytest.raises(ValueError, match=complaint):
            load_config(path)


def test_example_config_matches_defaults(tmp_path, monkeypatch):
    from pathlib import Path

    example = Path(__file__).parent.parent / "config.example.toml"
    monkeypatch.chdir(tmp_path)  # away from any local config.toml
    assert load_config(example) == load_config(None)


def test_cli_read_and_calibrate(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    colors = ["yellow", "violet", "red", "gold"]
    cv2.imwrite("r.png", render(colors, angle=200))

    assert main(["read", "r.png", "--debug", "debug.png"]) == 0
    assert "4.7 kΩ ±5%" in capsys.readouterr().out
    assert (tmp_path / "debug.png").exists()

    # The picture is upside down relative to the order given; calibrate works it out.
    assert main(["calibrate", *colors, "--image", "r.png"]) == 0
    samples = json.loads((tmp_path / "calibration.json").read_text())["samples"]
    assert sorted(samples) == sorted(colors)
    assert samples["yellow"][0][2] > 40  # yellow's b* is strongly positive
    assert main(["read", "r.png"]) == 0

    assert main(["calibrate", "red", "red", "--image", "r.png"]) == 1
    assert main(["calibrate", "--reset"]) == 0
    assert not (tmp_path / "calibration.json").exists()
