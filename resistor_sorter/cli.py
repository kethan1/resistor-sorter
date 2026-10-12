"""Command line entry point."""

from __future__ import annotations

import argparse
import itertools
import math
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import FrameGrabber, load_image, open_source
from .colors import COLORS, Palette, load_calibration, save_calibration
from .config import Config, load_config
from .sorter import Chute, Sorter
from .vision import Detection, annotate, detect, read, scene_problem
from .web import Preview


def _describe(detection: Detection | None) -> str:
    if detection is None:
        return "no resistor found"
    return ", ".join(f"{b.color} ({b.candidates[0][1]:.0f})" for b in detection.bands) or "no bands"


def _grab(cfg: Config) -> np.ndarray:
    camera = open_source(cfg.camera)
    try:
        frame = camera.read()
    finally:
        camera.close()
    if frame is None:
        raise RuntimeError("camera returned no frame")
    return frame


def _feed_preview(grabber: FrameGrabber, preview: Preview, overlay, max_fps: float) -> None:
    """Send every camera frame (up to max_fps) to the preview with the newest analysis drawn on."""
    seen, last = 0, 0.0
    while (item := grabber.next(seen)) is not None:
        seen, frame = item
        now = time.monotonic()
        if preview.watched and now - last >= 1 / max_fps:
            last = now
            preview.show(annotate(frame, *overlay()))


def cmd_run(cfg: Config, args) -> int:
    palette = Palette(load_calibration(cfg.calibration))
    sorter = Sorter(cfg.sorter, cfg.bins)
    chute = Chute(cfg.servo) if cfg.servo.enabled else None
    preview = Preview(cfg.web.host, cfg.web.port) if cfg.web.enabled else None
    grabber = FrameGrabber(open_source(cfg.camera))
    # Analysis of the latest processed frame, read by the preview thread.
    overlay = (None, None, None, True)
    if preview:
        threading.Thread(target=_feed_preview, daemon=True,
                         args=(grabber, preview, lambda: overlay, cfg.web.max_fps)).start()
        print(f"preview on http://{cfg.web.host}:{cfg.web.port}/", flush=True)
    print("ready - place a resistor under the camera (Ctrl-C to quit)", flush=True)

    seen, last_hint = 0, None
    try:
        while (item := grabber.next(seen)) is not None:
            seen, frame = item
            detection = detect(frame, palette, cfg.vision)
            reading = read(detection)
            hint = scene_problem(frame, cfg.vision) if detection is None else None
            overlay = (detection, reading, hint, sorter.trusted(reading))
            if hint != last_hint and hint is not None:
                print(f"setup problem: {hint}", flush=True)
            last_hint = hint
            event = sorter.step(detection is not None, reading)
            if event is not None:
                if event.bin is not None:
                    print(f"{event.reading}  [{' '.join(event.reading.colors)}]  -> {event.bin.name}", flush=True)
                elif event.reading is not None:
                    print(f"{event.reading}  [{' '.join(event.reading.colors)}]  -> reject (no bin for this value)", flush=True)
                else:
                    print(f"unreadable ({_describe(detection)})  -> reject", flush=True)
                if chute:
                    chute.point(event.bin.angle if event.bin else cfg.servo.reject_angle)
    except KeyboardInterrupt:
        print()
    finally:
        grabber.close()
        if chute:
            chute.close()
        if preview:
            preview.close()
    return 0


def cmd_read(cfg: Config, args) -> int:
    palette = Palette(load_calibration(cfg.calibration))
    frames = [(path, load_image(path)) for path in args.images] or [("camera", _grab(cfg))]
    if args.debug and len(frames) > 1:
        raise ValueError("--debug takes a single image")
    failed = False
    for name, frame in frames:
        detection = detect(frame, palette, cfg.vision)
        reading = read(detection)
        hint = scene_problem(frame, cfg.vision) if detection is None else None
        if reading is None:
            failed = True
            print(f"{name}: unreadable" + (f" ({hint})" if hint else ""))
        else:
            verdict = "trusted" if Sorter(cfg.sorter, cfg.bins).trusted(reading) else "NOT trusted"
            print(f"{name}: {reading}  [{' '.join(reading.colors)}]  "
                  f"margin {reading.margin:.1f}, worst band {reading.worst:.0f} -> {verdict}")
        print(f"  bands left to right: {_describe(detection)}")
        if args.debug:
            out = annotate(frame, detection, reading, hint, Sorter(cfg.sorter, cfg.bins).trusted(reading))
            if detection is not None:
                # Paste the straightened body, enlarged, along the bottom edge.
                scale = out.shape[1] / detection.strip.shape[1]
                strip = cv2.resize(detection.strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                strip = strip[:out.shape[0] // 3]
                out[-strip.shape[0]:] = strip
            cv2.imwrite(args.debug, out)
    return 1 if failed else 0


def cmd_calibrate(cfg: Config, args) -> int:
    path = Path(cfg.calibration)
    if args.reset:
        path.unlink(missing_ok=True)
        print(f"removed {path}")
        return 0
    if not args.colors:
        raise ValueError("give the band colours of the resistor under the camera, e.g. "
                         "`calibrate yellow violet red gold`")
    unknown = [c for c in args.colors if c not in COLORS]
    if unknown:
        raise ValueError(f"unknown colour(s) {', '.join(unknown)}; choose from {', '.join(COLORS)}")

    samples = load_calibration(path)
    palette = Palette(samples)
    frame = load_image(args.image) if args.image else _grab(cfg)
    detection = detect(frame, palette, cfg.vision)
    if detection is None:
        raise RuntimeError("no resistor found in the picture")
    # Segments at the ends may be end caps, and faint ones glare on bare body,
    # rather than bands; the resistor may lie either way round. Keep whichever
    # pairing of bands and names fits best.
    bands = detection.bands
    droppable = [i for i, band in enumerate(bands) if band.drop_cost < math.inf]
    subsets = [
        ([band for i, band in enumerate(bands) if i not in dropped], sum(bands[i].drop_cost for i in dropped))
        for dropped in itertools.combinations(droppable, len(bands) - len(args.colors))
    ] if len(bands) >= len(args.colors) else []
    if not subsets:
        raise RuntimeError(f"you named {len(args.colors)} bands but {len(bands)} were "
                           f"found ({_describe(detection)})")

    def misfit(option):
        (subset, cost), colors = option
        return cost + sum(dict(band.candidates)[color] for band, color in zip(subset, colors))

    (bands, _), colors = min(((s, c) for s in subsets for c in (args.colors, args.colors[::-1])), key=misfit)
    for band, color in zip(bands, colors):
        lab = [round(float(v), 1) for v in band.lab]
        print(f"  {color:7} Lab {lab}  (was {dict(band.candidates)[color]:.0f} from the old reference)")
        samples.setdefault(color, []).append(lab)
    save_calibration(path, samples)
    print(f"saved to {path}")
    return 0


def cmd_snap(cfg: Config, args) -> int:
    if not cv2.imwrite(args.output, _grab(cfg)):
        raise RuntimeError(f"cannot write {args.output!r}")
    print(f"wrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="resistor-sorter", description=__doc__)
    parser.add_argument("-c", "--config", help="TOML settings file (default: ./config.toml if present)")
    parser.add_argument("--source", help="override the camera source: a /dev/video* path, a V4L2 index, "
                        "picamera, or a file")
    parser.add_argument("--focus", type=int, help="override the manual focus (0-100, -1 = autofocus)")
    parser.add_argument("--zoom", type=int, help="override the zoom (0 = widest - 100)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="watch the camera and sort resistors as they appear")
    run.set_defaults(func=cmd_run)

    read_ = sub.add_parser("read", help="read one resistor from image files or a camera snapshot")
    read_.add_argument("images", nargs="*")
    read_.add_argument("--debug", metavar="FILE", help="save an annotated picture")
    read_.set_defaults(func=cmd_read)

    calibrate = sub.add_parser("calibrate", help="learn band colours from a resistor of known value")
    calibrate.add_argument("colors", nargs="*", metavar="COLOUR")
    calibrate.add_argument("--image", help="use a picture instead of the camera")
    calibrate.add_argument("--reset", action="store_true", help="forget all learned colours")
    calibrate.set_defaults(func=cmd_calibrate)

    snap = sub.add_parser("snap", help="save one camera frame")
    snap.add_argument("output")
    snap.set_defaults(func=cmd_snap)

    args = parser.parse_args(argv)
    # systemd stops the service with SIGTERM; exit through the finally blocks
    # so the camera is put back to automatic.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        cfg = load_config(args.config)
        if args.source:
            cfg.camera.source = args.source
        if args.focus is not None:
            cfg.camera.focus = args.focus
        if args.zoom is not None:
            cfg.camera.zoom = args.zoom
        return args.func(cfg, args)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"resistor-sorter: {error}", file=sys.stderr)
        return 1
