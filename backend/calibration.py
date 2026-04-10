import numpy as np
import os
import json
import logging

logger = logging.getLogger(__name__)

_DEFAULT_STATE = {
    'scale_factor': 1.0,
    'offset_x': 0.0,
    'offset_y': 0.0,
    'hand_size': 1.0,
    'samples': 0,
    'calibrated': False,
}

# Minimum number of samples before calibration is considered complete
CALIBRATION_THRESHOLD = 30


class CalibrationModule:
    """Tracks user-specific hand measurements to improve recognition accuracy."""

    def __init__(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(
                os.path.dirname(__file__), '..', 'calibration_data.json'
            )
        self.save_path = save_path
        self.calibration_data = dict(_DEFAULT_STATE)
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    loaded = json.load(f)
                # Merge so any new keys in _DEFAULT_STATE are present
                self.calibration_data = {**_DEFAULT_STATE, **loaded}
                logger.info("Calibration data loaded")
        except Exception as e:
            logger.warning(f"Could not load calibration: {e}")

    def _save(self):
        try:
            with open(self.save_path, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save calibration: {e}")

    def update(self, landmarks):
        """Incorporate a new landmark sample into the running calibration average.

        Expects landmarks as a list of dicts with 'x', 'y', and optionally 'z' keys.
        Requires at least 21 landmarks (a full MediaPipe hand).
        """
        try:
            if len(landmarks) < 21:
                return {'success': False, 'error': 'Expected at least 21 landmarks'}

            points = np.array(
                [[lm['x'], lm['y'], lm.get('z', 0.0)] for lm in landmarks],
                dtype=np.float64,
            )

            # Distance from wrist (0) to middle finger tip (12)
            hand_size = float(np.linalg.norm(points[12] - points[0]))

            n = self.calibration_data['samples']
            self.calibration_data['hand_size'] = (
                self.calibration_data['hand_size'] * n + hand_size
            ) / (n + 1)
            self.calibration_data['samples'] = n + 1

            if self.calibration_data['samples'] >= CALIBRATION_THRESHOLD:
                self.calibration_data['calibrated'] = True
                safe_size = max(self.calibration_data['hand_size'], 1e-6)
                self.calibration_data['scale_factor'] = 1.0 / safe_size

            self._save()

            return {
                'success': True,
                'samples': self.calibration_data['samples'],
                'calibrated': self.calibration_data['calibrated'],
                'hand_size': float(self.calibration_data['hand_size']),
            }
        except Exception as e:
            logger.error(f"Calibration update error: {e}")
            return {'success': False, 'error': str(e)}

    def get_params(self):
        """Return the current calibration parameters."""
        return dict(self.calibration_data)

    def reset(self):
        """Reset all calibration data to defaults."""
        self.calibration_data = dict(_DEFAULT_STATE)
        self._save()
        return {'success': True, 'message': 'Calibration reset'}
