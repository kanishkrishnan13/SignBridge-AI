"""
Calibration module — captures user hand landmarks and computes a
position / scale offset so gesture detection adapts to individual hands.
"""

import json
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CALIBRATION_FILE = os.path.join(_BASE_DIR, "model", "calibration.json")

_CALIBRATION_DURATION_SEC = 30
_MIN_SAMPLES = 10


class CalibrationManager:
    """
    Manages a calibration session that collects hand-landmark samples and
    computes a per-user offset (translation) and scale factor.
    """

    def __init__(self, calibration_file: str = _DEFAULT_CALIBRATION_FILE):
        self._calibration_file = calibration_file
        self._samples: list[np.ndarray] = []
        self._session_start: float | None = None
        self._offset: np.ndarray = np.zeros(3, dtype=np.float32)
        self._scale: float = 1.0
        self._calibrated: bool = False

        self._try_load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def offset(self) -> list[float]:
        return self._offset.tolist()

    @property
    def scale(self) -> float:
        return float(self._scale)

    def start_calibration(self) -> dict:
        """Begin a 30-second calibration capture session."""
        self._samples = []
        self._session_start = time.time()
        self._calibrated = False
        logger.info("Calibration session started")
        return {"status": "started", "duration_sec": _CALIBRATION_DURATION_SEC}

    def add_frame(self, landmarks: list | np.ndarray) -> dict:
        """
        Add a single frame's landmarks to the calibration buffer.

        Parameters
        ----------
        landmarks : array-like of shape (21, 3)

        Returns
        -------
        dict with session progress information
        """
        if self._session_start is None:
            return {"status": "error", "message": "Calibration session not started"}

        elapsed = time.time() - self._session_start
        if elapsed > _CALIBRATION_DURATION_SEC:
            return {"status": "expired", "message": "Calibration window has closed"}

        arr = np.array(landmarks, dtype=np.float32)
        if arr.shape != (21, 3):
            return {"status": "error", "message": f"Expected (21,3) landmarks, got {arr.shape}"}

        self._samples.append(arr)
        remaining = max(0.0, _CALIBRATION_DURATION_SEC - elapsed)
        return {
            "status": "collecting",
            "samples_collected": len(self._samples),
            "remaining_sec": round(remaining, 1),
        }

    def compute_offset(self) -> dict:
        """
        Compute calibration parameters from collected samples.

        The offset is the mean wrist position across all samples.
        The scale is the mean palm size (wrist→middle-finger MCP distance).

        Returns
        -------
        dict with offset, scale, and status
        """
        if len(self._samples) < _MIN_SAMPLES:
            return {
                "status": "error",
                "message": (
                    f"Insufficient samples ({len(self._samples)}). "
                    f"Need at least {_MIN_SAMPLES}."
                ),
            }

        stacked = np.stack(self._samples, axis=0)  # (N, 21, 3)

        # Translation offset: mean wrist position
        wrist_positions = stacked[:, 0, :]  # (N, 3)
        self._offset = wrist_positions.mean(axis=0)

        # Scale: mean palm size
        palm_vecs = stacked[:, 9, :] - stacked[:, 0, :]  # wrist → middle MCP
        palm_sizes = np.linalg.norm(palm_vecs, axis=1)
        mean_palm = float(palm_sizes.mean())
        self._scale = mean_palm if mean_palm > 1e-6 else 1.0

        self._calibrated = True
        self._save()

        logger.info(
            "Calibration complete — offset=%s  scale=%.4f  samples=%d",
            self._offset.tolist(),
            self._scale,
            len(self._samples),
        )
        return {
            "status": "ok",
            "offset": self._offset.tolist(),
            "scale": self._scale,
            "samples_used": len(self._samples),
        }

    def apply_calibration(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Translate and scale *landmarks* using the computed calibration.

        Parameters
        ----------
        landmarks : np.ndarray  shape (21, 3)

        Returns
        -------
        np.ndarray  shape (21, 3) — calibration-adjusted landmarks
        """
        if not self._calibrated:
            return landmarks
        adjusted = (landmarks - self._offset) / self._scale
        return adjusted.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        data = {
            "offset": self._offset.tolist(),
            "scale": float(self._scale),
            "calibrated": self._calibrated,
        }
        try:
            os.makedirs(os.path.dirname(self._calibration_file), exist_ok=True)
            with open(self._calibration_file, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            logger.info("Calibration saved to %s", self._calibration_file)
        except OSError as exc:
            logger.error("Failed to save calibration: %s", exc)

    def _try_load(self):
        if not os.path.exists(self._calibration_file):
            return
        try:
            with open(self._calibration_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._offset = np.array(data.get("offset", [0.0, 0.0, 0.0]), dtype=np.float32)
            self._scale = float(data.get("scale", 1.0))
            self._calibrated = bool(data.get("calibrated", False))
            logger.info(
                "Calibration loaded from %s (calibrated=%s)",
                self._calibration_file,
                self._calibrated,
            )
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not load calibration file: %s", exc)
