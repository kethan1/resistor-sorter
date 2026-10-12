"""A tiny MJPEG server so the camera view can be checked from a browser on a headless Pi."""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

_PAGE = b"""<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Resistor sorter</title>
<style>body{margin:0;background:#111}img{display:block;max-width:100%;max-height:100vh;margin:auto}</style>
<img src="/stream.mjpg" alt="camera view">
"""

_MAX_WIDTH = 640


class Preview:
    def __init__(self, host: str, port: int):
        self._frame_ready = threading.Condition()
        self._jpeg = b""
        self._viewers = 0
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(_PAGE)))
                    self.end_headers()
                    self.wfile.write(_PAGE)
                elif self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    # A small send buffer means a slow network drops frames
                    # rather than queueing seconds of stale video.
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024)
                    preview._stream(self.wfile)
                else:
                    self.send_error(404)

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def _stream(self, out) -> None:
        with self._frame_ready:
            self._viewers += 1
        try:
            while True:
                with self._frame_ready:
                    self._frame_ready.wait()
                    jpeg = self._jpeg
                out.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                          b"Content-Length: %d\r\n\r\n%b\r\n" % (len(jpeg), jpeg))
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self._frame_ready:
                self._viewers -= 1

    @property
    def watched(self) -> bool:
        """Whether anyone is looking; lets the caller skip drawing and encoding."""
        return self._viewers > 0

    def show(self, frame: np.ndarray) -> None:
        if frame.shape[1] > _MAX_WIDTH:
            scale = _MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._frame_ready:
                self._jpeg = jpeg.tobytes()
                self._frame_ready.notify_all()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
