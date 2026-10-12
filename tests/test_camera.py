import time

import cv2
import numpy as np
import pytest

from resistor_sorter import cli
from resistor_sorter.camera import FrameGrabber, open_source
from resistor_sorter.config import CameraConfig


class FakeSource:
    def __init__(self, count, fail=False):
        self.left, self.fail, self.closed = count, fail, False

    def read(self):
        time.sleep(0.01)
        if self.left == 0:
            if self.fail:
                raise OSError("unplugged")
            return None
        self.left -= 1
        return np.full((2, 2, 3), self.left, np.uint8)

    def close(self):
        self.closed = True


def test_slow_consumer_skips_to_the_newest_frame():
    source = FakeSource(10)
    grabber = FrameGrabber(source)
    number, _ = grabber.next(0)
    time.sleep(0.3)  # "processing" while the source runs dry
    later, frame = grabber.next(number)
    assert later == 10 and frame[0, 0, 0] == 0  # the last frame, nothing in between
    assert grabber.next(later) is None
    grabber.close()
    assert source.closed


def test_camera_errors_reach_the_consumer():
    grabber = FrameGrabber(FakeSource(1, fail=True))
    seen = 0
    with pytest.raises(RuntimeError, match="unplugged"):
        while (item := grabber.next(seen)) is not None:
            seen = item[0]
    grabber.close()


class FakeCapture:
    """Stands in for cv2.VideoCapture; `refuse` lists properties the "camera" lacks."""

    def __init__(self, source, backend, refuse=()):
        self.source, self.backend, self.refuse = source, backend, set(refuse)
        self.props = {cv2.CAP_PROP_EXPOSURE: 330, cv2.CAP_PROP_GAIN: 4, cv2.CAP_PROP_WB_TEMPERATURE: 4800}
        self.log = []

    def isOpened(self):
        return True

    def set(self, prop, value):
        if prop in self.refuse:
            return False
        self.props[prop] = value
        self.log.append((prop, value))
        return True

    def get(self, prop):
        return self.props.get(prop, -1)

    def read(self):
        return True, np.zeros((2, 2, 3), np.uint8)

    def release(self):
        pass


def _open(monkeypatch, refuse=(), **settings):
    monkeypatch.setattr(cv2, "VideoCapture", lambda source, backend: FakeCapture(source, backend, refuse))
    cfg = CameraConfig(**{"source": "/dev/video4", "lock_after": 0.0, **settings})
    return open_source(cfg)._capture


def test_manual_focus_and_zoom_are_set(monkeypatch):
    capture = _open(monkeypatch, focus=40, zoom=25)
    assert capture.backend == cv2.CAP_V4L2
    # Autofocus has to be off before the lens position is accepted.
    assert capture.log.index((cv2.CAP_PROP_AUTOFOCUS, 0)) < capture.log.index((cv2.CAP_PROP_FOCUS, 40))
    assert capture.props[cv2.CAP_PROP_ZOOM] == 25


def test_defaults_use_autofocus_and_leave_zoom_alone(monkeypatch):
    capture = _open(monkeypatch)
    assert capture.props[cv2.CAP_PROP_AUTOFOCUS] == 1
    assert cv2.CAP_PROP_ZOOM not in capture.props


def test_unsupported_zoom_is_an_error(monkeypatch):
    with pytest.raises(RuntimeError, match="zoom"):
        _open(monkeypatch, refuse=[cv2.CAP_PROP_ZOOM], zoom=10)


def test_usb_camera_keeps_exposure_and_white_balance_automatic(monkeypatch):
    def left_manual(source, backend):
        capture = FakeCapture(source, backend)
        # As a crashed run or the OBSBOT app can leave it.
        capture.props.update({cv2.CAP_PROP_AUTO_EXPOSURE: 1, cv2.CAP_PROP_AUTO_WB: 0})
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", left_manual)
    capture = open_source(CameraConfig(source="/dev/video4", lock_after=0.01))._capture
    assert capture.props[cv2.CAP_PROP_AUTO_EXPOSURE] == 3
    assert capture.props[cv2.CAP_PROP_AUTO_WB] == 1
    written = {prop for prop, _ in capture.log}
    assert not written & {cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_GAIN, cv2.CAP_PROP_WB_TEMPERATURE}


class SlowStartCapture(FakeCapture):
    """Like the OBSBOT Meet 2: the stream starts late, with black frames and a white one."""

    def __init__(self, source, backend, frames=(0, 0, 0, 0, 0, 255)):
        super().__init__(source, backend)
        self.reads, self.frames = 0, frames

    def read(self):
        self.reads += 1
        if self.reads == 1:
            time.sleep(0.05)
        value = self.frames[self.reads - 1] if self.reads <= len(self.frames) else 120
        frame = np.full((32, 32, 3), value, np.uint8)
        if value == 120:
            frame[0, 0] = 90  # real pictures are never perfectly flat
        return True, frame


def test_blank_startup_frames_are_skipped(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", SlowStartCapture)
    source = open_source(CameraConfig(source="/dev/video4", lock_after=0.0))
    assert source.read().mean() > 100
    assert source._capture.reads == 8  # the 6 blank ones, one while opening, this one


def test_settling_starts_at_the_first_frame(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", SlowStartCapture)
    source = open_source(CameraConfig(source="/dev/video4", lock_after=0.03))
    assert source._capture.reads > 8  # the 0.05 s wait for the first frame did not use up the settling


def test_a_camera_that_only_sends_black_still_returns_frames(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda source, backend: SlowStartCapture(source, backend, (0,) * 1000))
    source = open_source(CameraConfig(source="/dev/video4", lock_after=0.0))
    assert source.read().max() == 0  # for the "too dark" message


def test_close_hands_the_camera_back(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda source, backend: FakeCapture(source, backend))
    source = open_source(CameraConfig(source="/dev/video4", lock_after=0.01, focus=40, zoom=25))
    capture = source._capture
    assert (cv2.CAP_PROP_ZOOM, 25) in capture.log  # it was -1 before
    source.close()
    # Automatic again, for the next program that opens the camera.
    assert capture.props[cv2.CAP_PROP_AUTO_EXPOSURE] == 3
    assert capture.props[cv2.CAP_PROP_AUTO_WB] == 1
    assert capture.props[cv2.CAP_PROP_AUTOFOCUS] == 1
    assert capture.props[cv2.CAP_PROP_ZOOM] == -1


def test_cli_overrides_focus_and_zoom(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cli, "cmd_snap", lambda cfg, args: seen.update(focus=cfg.camera.focus, zoom=cfg.camera.zoom) or 0)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["--focus", "35", "--zoom", "0", "snap", "x.jpg"]) == 0
    assert seen == {"focus": 35, "zoom": 0}
