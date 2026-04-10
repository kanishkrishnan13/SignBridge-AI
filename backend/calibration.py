import time
import warnings
import numpy as np


class Calibrator:
    """
    30-second hand-size calibration session.

    Collects landmark data from successive frames, then computes a scale
    factor that can be used to normalise landmarks to a reference hand size.
    """

    DURATION_SECONDS = 30

    def __init__(self, detector):
        self.detector = detector
        self.calibration_data: list[np.ndarray] = []
        self.is_calibrating: bool = False
        self.start_time: float | None = None
        self.duration: int = self.DURATION_SECONDS
        self._scale_factor: float | None = None

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def start(self) -> dict:
        """Start (or restart) a 30-second calibration session."""
        self.calibration_data = []
        self.is_calibrating = True
        self.start_time = time.monotonic()
        self._scale_factor = None
        return {'status': 'started', 'duration': self.duration}

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame) -> dict:
        """
        Extract landmarks from *frame* and accumulate calibration data.

        Returns a status dict::

            {
                'status':    'calibrating' | 'complete' | 'not_started',
                'progress':  0.0–1.0,
                'remaining': seconds remaining (float),
                'frames_collected': int,
            }
        """
        if not self.is_calibrating:
            return {
                'status': 'not_started',
                'progress': 0.0,
                'remaining': float(self.duration),
                'frames_collected': 0,
            }

        elapsed = time.monotonic() - self.start_time
        remaining = max(0.0, self.duration - elapsed)
        progress = min(1.0, elapsed / self.duration)

        landmarks = self.detector.detect_landmarks(frame)
        if landmarks is not None:
            self.calibration_data.append(landmarks.copy())

        if elapsed >= self.duration:
            self.is_calibrating = False
            self._scale_factor = self.compute_scale_factor()
            return {
                'status': 'complete',
                'progress': 1.0,
                'remaining': 0.0,
                'frames_collected': len(self.calibration_data),
                'scale_factor': self._scale_factor,
            }

        return {
            'status': 'calibrating',
            'progress': round(progress, 3),
            'remaining': round(remaining, 1),
            'frames_collected': len(self.calibration_data),
        }

    # ------------------------------------------------------------------
    # Scale computation
    # ------------------------------------------------------------------

    def compute_scale_factor(self) -> float:
        """
        Compute an average hand size (palm width) from accumulated frames.

        Uses the Euclidean distance between wrist (0) and middle-finger MCP (9)
        as a proxy for hand size.  Returns 1.0 if no data was collected.
        """
        if not self.calibration_data:
            warnings.warn("No calibration data collected; returning scale_factor=1.0")
            return 1.0

        sizes = []
        for lm in self.calibration_data:
            try:
                lm = np.array(lm).reshape(21, 3)
                dist = float(np.linalg.norm(lm[9, :2] - lm[0, :2]))
                if dist > 0:
                    sizes.append(dist)
            except Exception:
                continue

        if not sizes:
            return 1.0

        avg_size = float(np.mean(sizes))
        # Reference hand size: 0.3 (typical normalised wrist→MCP distance)
        reference = 0.3
        scale_factor = reference / avg_size if avg_size > 0 else 1.0
        return round(scale_factor, 6)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return the current calibration status without processing a frame."""
        if not self.is_calibrating and self.start_time is None:
            return {
                'status': 'idle',
                'progress': 0.0,
                'remaining': float(self.duration),
                'frames_collected': 0,
                'scale_factor': None,
            }

        if self.is_calibrating:
            elapsed = time.monotonic() - self.start_time
            remaining = max(0.0, self.duration - elapsed)
            progress = min(1.0, elapsed / self.duration)
            return {
                'status': 'calibrating',
                'progress': round(progress, 3),
                'remaining': round(remaining, 1),
                'frames_collected': len(self.calibration_data),
                'scale_factor': None,
            }

        return {
            'status': 'complete',
            'progress': 1.0,
            'remaining': 0.0,
            'frames_collected': len(self.calibration_data),
            'scale_factor': self._scale_factor,
        }
