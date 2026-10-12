import json

import cv2
import pytest
from synth import render

from resistor_sorter.cli import main
from resistor_sorter.config import Bin, SorterConfig, load_config
from resistor_sorter.decode import decode
from resistor_sorter.sorter import Sorter, pick_bin

READING = decode(["yellow", "violet", "red", "gold"])
OTHER = decode(["yellow", "violet", "orange", "gold"])


def make_sorter(**overrides):
    cfg = SorterConfig(stable_frames=3, reject_after=6, clear_frames=2, **overrides)
    return Sorter(cfg, load_config(None).bins)


def test_accepts_after_stable_frames_then_waits_for_removal():
    sorter = make_sorter()
    assert sorter.step(True, READING) is None
    assert sorter.step(True, READING) is None
    event = sorter.step(True, READING)
    assert event.reading.ohms == 4700 and event.bin.name == "1 kΩ – 10 kΩ"
    # Same resistor still lying there: nothing more happens.
    assert all(sorter.step(True, READING) is None for _ in range(10))
    # Removed, then a new one placed.
    sorter.step(False, None)
    sorter.step(False, None)
    events = [sorter.step(True, OTHER) for _ in range(3)]
    assert events[-1].reading.ohms == 47000


def test_changing_reading_restarts_the_count():
    sorter = make_sorter()
    steps = [READING, READING, OTHER, READING, READING]
    assert all(sorter.step(True, r) is None for r in steps)
    assert sorter.step(True, READING) is not None


def test_rejects_when_never_stable():
    sorter = make_sorter()
    events = [sorter.step(True, None) for _ in range(6)]
    assert events[:5] == [None] * 5
    assert events[5].reading is None and events[5].bin is None


def test_uncertain_readings_are_not_trusted():
    from dataclasses import replace

    for shaky in (replace(READING, margin=1.0), replace(READING, worst=30.0)):
        sorter = make_sorter(min_margin=4.0, max_color_distance=15.0)
        events = [sorter.step(True, shaky) for _ in range(6)]
        assert events[5].reading is None


def test_a_flickering_detection_does_not_reset():
    sorter = make_sorter()
    sorter.step(True, READING)
    sorter.step(False, None)  # one dropped frame, fewer than clear_frames
    sorter.step(True, READING)
    assert sorter.step(True, READING) is not None


def test_pick_bin():
    bins = [Bin("common", values=[1e3, 10e3]), Bin("low", max=1e3), Bin("high", min=1e3)]
    assert pick_bin(bins, 1e3).name == "common"
    assert pick_bin(bins, 470).name == "low"
    assert pick_bin(bins, 4700).name == "high"
    assert pick_bin([Bin("low", max=1e3)], 4700) is None


def test_config_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[sorter]\nstable_frames = 9\n[[bin]]\nname = "all"\nangle = 10\n')
    cfg = load_config(path)
    assert cfg.sorter.stable_frames == 9
    assert [b.name for b in cfg.bins] == ["all"] and cfg.bins[0].accepts(1e6)
    path.write_text("[sorter]\nstabel_frames = 9\n")
    with pytest.raises(ValueError, match="stabel_frames"):
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
