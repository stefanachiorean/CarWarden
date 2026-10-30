import numpy as np
from sklearn.ensemble import RandomForestClassifier

from features import CLASSIFIER_FEATURES


class AttackClassifier:
    """Supervised model that names the attack type of a window.

    Trained on labelled windows ("normal", "DoS", "Fuzzy", ...). It can only
    name attack types it has seen. If it says "normal" for a window that the
    detector flagged, the pipeline reports "unknown anomaly".
    """

    def __init__(self, n_estimators=200, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            class_weight="balanced",
                                            random_state=random_state,
                                            n_jobs=-1)

    def fit(self, windows):
        self.model.fit(windows[CLASSIFIER_FEATURES].to_numpy(dtype=float), windows["attack_type"])
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, windows):
        """Returns (predicted labels, confidence = probability of that label)."""
        X = windows[CLASSIFIER_FEATURES].to_numpy(dtype=float)
        proba = self.model.predict_proba(X)
        best = proba.argmax(axis=1)
        labels = np.array(self.classes_)[best]
        confidence = proba[np.arange(len(X)), best]
        return labels, confidence

    def feature_importance(self):
        return dict(zip(CLASSIFIER_FEATURES, self.model.feature_importances_))