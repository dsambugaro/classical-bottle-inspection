"""Server-side capture threads (REQ-I4): one thread per active camera keeps the latest frame."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

WSL_USB_HINT: str = (
    "USB cameras in WSL2 need usbipd (https://learn.microsoft.com/windows/wsl/connect-usb)."
)


class CaptureLike(Protocol):
    """The part of cv2.VideoCapture the app uses (fakes implement it in tests)."""

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV name
        """True when the device or stream is open."""
        ...

    def read(self) -> tuple[bool, Any]:
        """Grabs and decodes the next frame."""
        ...

    def release(self) -> None:
        """Closes the device."""
        ...

    def set(self, propId: int, value: float) -> bool:  # noqa: N803 - OpenCV name
        """Sets a capture property."""
        ...


@dataclass(frozen=True)
class CameraConfig:
    """What is needed to open a camera."""

    id: int
    name: str
    kind: str
    device_index: int | None
    url: str | None
    width: int | None
    height: int | None
    fps: float


def open_opencv(config: CameraConfig) -> CaptureLike:
    """Opens a camera with OpenCV (USB by index, RTSP/HTTP by URL)."""
    if config.kind == "usb":
        capture = cv2.VideoCapture(int(config.device_index or 0))
    else:
        capture = cv2.VideoCapture(config.url or "")
    if config.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(config.width))
    if config.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(config.height))
    return capture


CaptureFactory = Callable[[CameraConfig], CaptureLike]


class CameraError(Exception):
    """A camera cannot be opened or stopped delivering frames."""

    def __init__(self, code: str, message: str) -> None:
        """Error with a translatable code."""
        super().__init__(message)
        self.code = code


class CaptureWorker:
    """Reads frames continuously in a thread and keeps only the latest one."""

    def __init__(self, config: CameraConfig, factory: CaptureFactory) -> None:
        """Not started until `start`."""
        self.config = config
        self.factory = factory
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._frame_seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: CameraError | None = None
        self.users = 0

    def start(self, timeout: float = 8.0) -> None:
        """Opens the camera in the thread; raises CameraError when it cannot be opened."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self.error = None
        self._thread = threading.Thread(
            target=self._run, name=f"camera-{self.config.id}", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            self.stop()
            raise CameraError("camera.timeout", f"camera '{self.config.name}' did not answer")
        if self.error is not None:
            raise self.error

    def _run(self) -> None:
        """Capture loop (thread)."""
        try:
            capture = self.factory(self.config)
        except Exception as error:  # noqa: BLE001 - surfaced to the user
            self.error = CameraError("camera.open_failed", str(error))
            self._ready.set()
            return
        if not capture.isOpened():
            hint = f" {WSL_USB_HINT}" if self.config.kind == "usb" else ""
            self.error = CameraError(
                "camera.open_failed", f"cannot open camera '{self.config.name}'.{hint}"
            )
            self._ready.set()
            capture.release()
            return
        min_interval = 1.0 / max(self.config.fps * 2, 1.0)
        failures = 0
        try:
            while not self._stop.is_set():
                start = time.monotonic()
                ok, frame = capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 50:
                        self.error = CameraError(
                            "camera.no_frames",
                            f"camera '{self.config.name}' stopped sending frames",
                        )
                        self._ready.set()
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.time()
                    self._frame_seq += 1
                self._ready.set()
                elapsed = time.monotonic() - start
                if elapsed < min_interval:
                    self._stop.wait(min_interval - elapsed)
        finally:
            capture.release()

    def latest(self) -> tuple[np.ndarray | None, int]:
        """(latest frame or None, sequence number)."""
        with self._lock:
            return (self._frame.copy() if self._frame is not None else None), self._frame_seq

    def wait_new(self, after_seq: int, timeout: float = 2.0) -> tuple[np.ndarray | None, int]:
        """Waits for a frame newer than `after_seq`."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame, seq = self.latest()
            if frame is not None and seq > after_seq:
                return frame, seq
            if self.error is not None:
                raise self.error
            time.sleep(0.01)
        return self.latest()

    @property
    def alive(self) -> bool:
        """True while the capture thread runs."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        """Stops the thread and releases the camera."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None
