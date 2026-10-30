import numpy as np
from sklearn.ensemble import IsolationForest

from features import FEATURES

ALERT_RISK = 50.0   # risk >= 50 means "flagged"
HIGH_RISK = 75.0


def risk_level(risk):
    if risk >= HIGH_RISK:
        return "HIGH"
    if risk >= ALERT_RISK:
        return "MEDIUM"
    return "NORMAL"


class Detector:
    """Isolation Forest trained on NORMAL windows only.

    The raw anomaly score is turned into a 0-100 risk score:
      risk = 0    -> as ordinary as the typical normal window
      risk = 50   -> the alert threshold (a normal window exceeds it only about
                     target_fpr of the time, measured on held-out normal windows)
      risk = 100  -> far beyond the threshold
    """

    def __init__(self, n_estimators=200, target_fpr=0.01, random_state=42):
        self.n_estimators = n_estimators
        self.target_fpr = target_fpr
        self.random_state = random_state

    def fit(self, fit_windows, calib_windows, known_ids=None, id_share=None):
        """fit_windows trains the forest; calib_windows (held-out normal) sets the threshold."""
        X = fit_windows[FEATURES].to_numpy(dtype=float)

        # robust statistics of normal traffic, used by the explanation layer
        self.median_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.median_), axis=0) * 1.4826
        std = X.std(axis=0)
        self.scale_ = np.where(mad > 1e-9, mad, np.where(std > 1e-9, std, 1.0))

        self.model_ = IsolationForest(n_estimators=self.n_estimators,
                                      random_state=self.random_state)
        self.model_.fit(X)

        calib_raw = self._raw(calib_windows)
        self.low_ = float(np.median(calib_raw))
        self.threshold_ = float(np.quantile(calib_raw, 1.0 - self.target_fpr))
        if self.threshold_ - self.low_ < 1e-6:
            self.threshold_ = self.low_ + 1e-6
        # Isolation Forest scores are bounded (extreme outliers reach only a bit above the
        # threshold), so risk 100 is anchored close to it. This scale only decides
        # MEDIUM vs HIGH; flagged / not flagged is decided by risk >= 50.
        self.cap_ = self.threshold_ + 0.75 * (self.threshold_ - self.low_)

        self.known_ids_ = set(known_ids) if known_ids is not None else set()
        self.id_share_ = dict(id_share) if id_share is not None else {}
        return self

    def _raw(self, windows):
        X = windows[FEATURES].to_numpy(dtype=float)
        return -self.model_.score_samples(X)     # higher = more anomalous

    def risk(self, windows):
        """Risk score 0-100 for every window (numpy array)."""
        raw = self._raw(windows)
        below = ALERT_RISK * np.clip((raw - self.low_) / (self.threshold_ - self.low_), 0, 1)
        above = ALERT_RISK + (100.0 - ALERT_RISK) * np.clip(
            (raw - self.threshold_) / (self.cap_ - self.threshold_), 0, 1)
        return np.where(raw <= self.threshold_, below, above)

    def predict(self, windows):
        """True for windows whose risk reaches the alert threshold."""
        return self.risk(windows) >= ALERT_RISK
