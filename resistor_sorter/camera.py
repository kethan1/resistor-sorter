"""Frame sources. Every source yields BGR frames from read(), or None when it runs out."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import CameraConfig

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_MAX_BLANK = 30  # blank frames in a row that read() skips (the OBSBOT starts with 5-6)


class PiCamera:
    """Raspberry Pi camera through picamera2/libcamera (Raspberry Pi OS Bookworm or newer)."""

    def __init__(self, cfg: CameraConfig):
        from picamera2 import Picamera2

        self._camera = Picamera2()
        # picamera2's "RGB888" is B,G,R in memory, which is what OpenCV expects.
        # A fixed frame duration stops auto exposure from slowing the sensor
        # down in dim light; it raises the gain instead.
        frame_us = int(1e6 / cfg.fps)
        self._camera.configure(self._camera.create_video_configuration(
            main={"size": (cfg.width, cfg.height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_us, frame_us)},
        ))
        self._camera.start()
        if cfg.lock_after > 0:
            # Let auto exposure and white balance settle, then freeze them so
            # colours do not drift as different resistors come and go.
            time.sleep(cfg.lock_after)
            metadata = self._camera.capture_metadata()
            self._camera.set_controls({
                "AeEnable": False,
                "AwbEnable": False,
                "ExposureTime": metadata["ExposureTime"],
                "AnalogueGain": metadata["AnalogueGain"],
                "ColourGains": metadata["ColourGains"],
            })

    def read(self) -> np.ndarray | None:
        return self._camera.capture_array()

    def close(self) -> None:
        self._camera.stop()
        self._camera.close()


class CvCamera:
    """A V4L2/USB camera (such as the OBSBOT Meet 2) or a video file through OpenCV."""

    def __init__(self, source: int | str, cfg: CameraConfig):
        self._is_device = isinstance(source, int) or source.startswith("/dev/")
        self._zoom_before: float | None = None
        backend = cv2.CAP_V4L2 if isinstance(source, str) and self._is_device else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(source, backend)
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open camera source {source!r}")
        if not self._is_device:
            return
        # Uncompressed 4K or 1080p video is too much for USB; most webcams
        # only reach full frame rate in MJPEG.
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._capture.set(cv2.CAP_PROP_FPS, cfg.fps)
        # Exposure and white balance stay automatic. Like many UVC cameras, the
        # OBSBOT Meet 2 reports the stored manual values rather than what auto
        # mode is using, so freezing them turns the picture almost black or
        # off-colour. normalise() cancels their drift against the paper instead.
        # Another program (or an older version of this one) may have left them manual.
        self._set_automatic()
        # The OBSBOT's stream can take ~3 s to start, and begins with black
        # frames, so time the settling from the first real frame rather than
        # from opening the camera.
        self.read()
        # Focus and zoom go in only once the stream is running. After a few
        # idle seconds the OBSBOT takes ~5 s to start streaming and drops a
        # zoom set before that partway through, while still reporting it as set.
        self.set_focus(None if cfg.focus < 0 else cfg.focus)
        if cfg.zoom >= 0:
            self._zoom_before = self._capture.get(cv2.CAP_PROP_ZOOM)
            self.set_zoom(cfg.zoom)
        # Let exposure and white balance (and the lens, after a focus/zoom
        # change) settle before frames are used.
        end = time.monotonic() + cfg.lock_after
        while time.monotonic() < end:
            self._capture.read()

    def _set_automatic(self) -> None:
        """Hand exposure and white balance back to the camera."""
        # V4L2 auto exposure is mode 3 (aperture priority) on most webcams, 0 on some.
        if not self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3):
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self._capture.set(cv2.CAP_PROP_AUTO_WB, 1)

    def _set(self, prop: int, value: float, what: str) -> None:
        if not self._capture.set(prop, value):
            raise RuntimeError(f"the camera refused {what} = {value:g}")

    def set_focus(self, value: int | None) -> None:
        """Fix the lens at `value` (0-100 on the OBSBOT Meet 2), or None for autofocus."""
        if value is None:
            # Cameras without autofocus have a fixed lens, which is fine too.
            self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            return
        self._set(cv2.CAP_PROP_AUTOFOCUS, 0, "autofocus")
        self._set(cv2.CAP_PROP_FOCUS, value, "focus")

    def set_zoom(self, value: int) -> None:
        """Zoom in to `value` (0 = widest, 100 = closest on the OBSBOT Meet 2)."""
        self._set(cv2.CAP_PROP_ZOOM, value, "zoom")

    def read(self) -> np.ndarray | None:
        # The OBSBOT sends flat black frames while its stream starts, and now
        # and then a flat white one. Skip them, but not forever: a covered
        # lens should still reach the "too dark" message.
        for _ in range(_MAX_BLANK):
            ok, frame = self._capture.read()
            if not ok:
                return None
            if not (self._is_device and _blank(frame)):
                break
        return frame

    def close(self) -> None:
        try:
            if self._is_device:
                # V4L2 settings outlive this process; leave the camera as other programs expect it.
                self._set_automatic()
                self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                if self._zoom_before is not None:
                    self._capture.set(cv2.CAP_PROP_ZOOM, self._zoom_before)
        finally:
            self._capture.release()


def _blank(frame: np.ndarray) -> bool:
    """A frame of one flat value; real pictures always have some noise and shading."""
    sample = frame[::16, ::16]
    return int(sample.max()) - int(sample.min()) < 10


class StillImage:
    """Repeats one image forever; handy for trying `run` without a camera."""

    def __init__(self, path: str):
        self._frame = load_image(path)

    def read(self) -> np.ndarray | None:
        time.sleep(0.05)
        return self._frame

    def close(self) -> None:
        pass


def load_image(path: str) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot read image {path!r}")
    return frame


def open_source(cfg: CameraConfig):
    source = cfg.source
    if source == "picamera":
        return PiCamera(cfg)
    if source.isdigit():
        return CvCamera(int(source), cfg)
    if source.startswith("/dev/"):
        return CvCamera(source, cfg)
    if Path(source).suffix.lower() in _IMAGE_SUFFIXES:
        return StillImage(source)
    return CvCamera(source, cfg)


class FrameGrabber:
    """Reads a source on a background thread, always holding the newest frame.

    Consumers that are slower than the camera simply skip frames instead of
    falling further and further behind.
    """

    def __init__(self, source):
        self._source = source
        self._ready = threading.Condition()
        self._frame: np.ndarray | None = None
        self._count = 0
        self._done = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._done:
                frame = self._source.read()
                with self._ready:
                    if frame is None:
                        self._done = True
                    else:
                        self._frame, self._count = frame, self._count + 1
                    self._ready.notify_all()
        except BaseException as error:  # handed to the consumer in next()
            with self._ready:
                self._error, self._done = error, True
                self._ready.notify_all()

    def next(self, after: int) -> tuple[int, np.ndarray] | None:
        """Wait for a frame newer than number `after`; (number, frame), or None at the end."""
        with self._ready:
            self._ready.wait_for(lambda: self._count > after or self._done)
            if self._count > after:
                return self._count, self._frame
            if self._error is not None:
                raise RuntimeError(f"camera failed: {self._error}") from self._error
            return None

    def close(self) -> None:
        with self._ready:
            self._done = True
            self._ready.notify_all()
        self._thread.join(timeout=2)
        self._source.close()
