"""Threaded camera capture using OpenCV.

Runs frame acquisition in a daemon thread and provides frames
via a thread-safe queue. Designed for low-latency, continuous
capture with graceful start/stop lifecycle.

Thread model:
    - Capture thread (daemon): reads frames from the camera device
      and places them into a bounded queue.
    - Consumer (pipeline thread): calls get_frame() to retrieve
      the latest frame without blocking.

Frame dropping:
    When the queue is full (maxsize=2), the oldest frame is discarded
    so the consumer always gets the most recent frame. This prevents
    processing stale data when the pipeline is slower than capture.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from core.types import FrameData
from core.exceptions import CameraError


class CameraCapture:
    """Threaded camera capture using OpenCV.

    Runs capture in a daemon thread and provides frames via a
    thread-safe queue. Consumers call get_frame() to retrieve
    the latest captured frame without blocking.

    Example:
        camera = CameraCapture(device_index=0)
        camera.start()
        try:
            while True:
                frame = camera.get_frame()
                if frame is not None:
                    process(frame)
        finally:
            camera.stop()
    """

    # Number of initial frames to skip (webcam warm-up)
    _WARMUP_FRAMES: int = 5

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """Initialize camera capture parameters.

        Args:
            device_index: OS camera device index (0 = default camera).
            width: Requested capture width in pixels.
            height: Requested capture height in pixels.
            fps: Requested capture framerate.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._device_index = device_index
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps

        # Actual resolution (set after camera opens)
        self._width = width
        self._height = height

        # Thread management
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Frame queue — maxsize=2 ensures we drop stale frames
        self._frame_queue: queue.Queue[FrameData] = queue.Queue(maxsize=2)
        self._frame_count: int = 0

        # Performance tracking
        self._fps_actual: float = 0.0
        self._last_fps_time: float = 0.0
        self._fps_frame_count: int = 0

        # OpenCV capture object (created in start())
        self._capture: Optional[cv2.VideoCapture] = None

        self._logger.debug(
            "CameraCapture initialized: device=%d, %dx%d @ %dfps",
            device_index, width, height, fps,
        )

    def start(self) -> None:
        """Start the camera capture thread.

        Opens the camera device, configures resolution/FPS, skips
        warm-up frames, then starts the capture thread.

        Raises:
            CameraError: If the camera cannot be opened or produces
                         no readable frames.
        """
        with self._lock:
            if self._running:
                self._logger.warning("Camera capture already running")
                return

        self._logger.info(
            "Opening camera device %d...", self._device_index
        )

        # Open the camera device
        self._capture = cv2.VideoCapture(self._device_index)
        if not self._capture.isOpened():
            raise CameraError(
                f"Failed to open camera device {self._device_index}. "
                "Check that the camera is connected and not in use by "
                "another application."
            )

        # Configure capture properties
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        self._capture.set(cv2.CAP_PROP_FPS, self._requested_fps)

        # Read back actual resolution (camera may not support requested)
        self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

        self._logger.info(
            "Camera opened: actual resolution %dx%d @ %.1f fps",
            self._width, self._height, actual_fps,
        )

        if (self._width != self._requested_width
                or self._height != self._requested_height):
            self._logger.warning(
                "Camera resolution differs from requested: "
                "requested %dx%d, got %dx%d",
                self._requested_width, self._requested_height,
                self._width, self._height,
            )

        # Warm-up: skip initial frames (often black/corrupt)
        self._logger.debug(
            "Skipping %d warm-up frames...", self._WARMUP_FRAMES
        )
        for i in range(self._WARMUP_FRAMES):
            ret, _ = self._capture.read()
            if not ret:
                self._capture.release()
                self._capture = None
                raise CameraError(
                    f"Camera failed to produce frames during warm-up "
                    f"(failed at frame {i + 1}/{self._WARMUP_FRAMES})."
                )

        # Start the capture thread
        self._running = True
        self._frame_count = 0
        self._last_fps_time = time.monotonic()
        self._fps_frame_count = 0

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCapture",
            daemon=True,
        )
        self._thread.start()
        self._logger.info("Camera capture thread started")

    def stop(self) -> None:
        """Stop the capture thread and release the camera device.

        Safe to call multiple times. Blocks until the capture thread
        has terminated.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False

        self._logger.info("Stopping camera capture...")

        # Wait for the capture thread to finish
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                self._logger.warning(
                    "Camera capture thread did not terminate within timeout"
                )
        self._thread = None

        # Release the camera
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        # Clear any remaining frames in the queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self._logger.info(
            "Camera capture stopped. Total frames captured: %d",
            self._frame_count,
        )

    def get_frame(self) -> Optional[FrameData]:
        """Get the latest captured frame (non-blocking).

        If multiple frames are queued, discards older ones and
        returns only the most recent frame.

        Returns:
            The most recent FrameData, or None if no frame is available.
        """
        frame: Optional[FrameData] = None

        # Drain the queue, keeping only the last frame
        try:
            while True:
                frame = self._frame_queue.get_nowait()
        except queue.Empty:
            pass

        return frame

    def is_running(self) -> bool:
        """Check if the capture thread is actively running.

        Returns:
            True if capturing frames, False otherwise.
        """
        return self._running

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the actual capture resolution.

        Returns:
            Tuple of (width, height) in pixels.
        """
        return (self._width, self._height)

    @property
    def fps(self) -> float:
        """Get the measured capture framerate.

        Returns:
            Frames per second, measured over a rolling window.
        """
        return self._fps_actual

    @property
    def frame_count(self) -> int:
        """Get the total number of frames captured.

        Returns:
            Running frame count since start().
        """
        return self._frame_count

    def _capture_loop(self) -> None:
        """Internal capture loop running in the daemon thread.

        Continuously reads frames from the camera and places them
        in the frame queue. Runs until self._running is set to False.

        If a frame read fails, logs a warning and continues. After
        multiple consecutive failures, stops the capture.
        """
        consecutive_failures = 0
        max_consecutive_failures = 30  # ~1 second at 30fps

        self._logger.debug("Capture loop started")

        while self._running:
            if self._capture is None or not self._capture.isOpened():
                self._logger.error("Camera device lost")
                self._running = False
                break

            ret, raw_frame = self._capture.read()

            if not ret or raw_frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    self._logger.error(
                        "Camera failed to produce frames for %d "
                        "consecutive attempts. Stopping capture.",
                        consecutive_failures,
                    )
                    self._running = False
                    break
                self._logger.debug(
                    "Frame read failed (attempt %d/%d)",
                    consecutive_failures, max_consecutive_failures,
                )
                continue

            # Successful read — reset failure counter
            consecutive_failures = 0
            self._frame_count += 1

            # Create the FrameData packet
            timestamp = time.monotonic()
            frame_data = FrameData(
                frame=raw_frame,
                timestamp=timestamp,
                frame_number=self._frame_count,
                width=raw_frame.shape[1],
                height=raw_frame.shape[0],
            )

            # Put frame in queue; if full, discard the oldest frame
            try:
                self._frame_queue.put_nowait(frame_data)
            except queue.Full:
                # Discard the oldest frame to make room
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put_nowait(frame_data)
                except queue.Full:
                    # Extremely unlikely — another thread consumed between ops
                    pass

            # Update FPS measurement (every 1 second)
            self._fps_frame_count += 1
            elapsed = timestamp - self._last_fps_time
            if elapsed >= 1.0:
                self._fps_actual = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._last_fps_time = timestamp

        self._logger.debug("Capture loop ended")

    def __del__(self) -> None:
        """Ensure camera is released on garbage collection."""
        if self._running:
            self.stop()

    def __repr__(self) -> str:
        """Return a string representation of the camera capture state."""
        status = "running" if self._running else "stopped"
        return (
            f"CameraCapture(device={self._device_index}, "
            f"{self._width}x{self._height}, "
            f"status={status}, frames={self._frame_count})"
        )
