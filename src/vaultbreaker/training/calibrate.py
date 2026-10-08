import numpy as np
from scipy.optimize import minimize
import joblib
from pathlib import Path
from typing import Dict, Tuple, Union, Any

from vaultbreaker.utils.logger import get_logger

logger = get_logger("Calibrator")

class TemperatureScalingCalibrator:
    """
    Calibrates binary prediction probabilities using Temperature Scaling
    and establishes decision thresholds aligned to a target False Positive Rate (FPR).
    """
    def __init__(self, target_fpr: float = 0.05):
        self.temperature: float = 1.0
        self.target_fpr: float = target_fpr
        self.decision_threshold: float = 0.5
        self.risk_thresholds = {
            "clean_max": 0.20,
            "low_risk": 0.40,
            "medium_risk": 0.65,
            "high_risk": 0.85
        }

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> "TemperatureScalingCalibrator":
        """
        Optimize temperature T > 0 on validation set logits to minimize
        negative log-likelihood (binary cross-entropy).
        """
        val_logits = np.asarray(val_logits, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=np.float64)

        def nll_loss(t_param: np.ndarray) -> float:
            t = t_param[0]
            scaled = val_logits / max(1e-4, t)
            # numerically stable binary cross entropy
            log_p = -np.logaddexp(0.0, -scaled)
            log_1_p = -np.logaddexp(0.0, scaled)
            loss = -(val_labels * log_p + (1.0 - val_labels) * log_1_p)
            return float(np.mean(loss))

        res = minimize(nll_loss, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        logger.info(f"Fitted calibration temperature T = {self.temperature:.4f} (NLL: {res.fun:.4f})")

        # Calibrated probabilities on validation set
        val_probs = self.calibrate_logits(val_logits)

        # Calibrate decision threshold to target FPR on clean covers (y == 0)
        clean_probs = val_probs[val_labels == 0]
        if len(clean_probs) > 0:
            # Sort clean probabilities descending to pick threshold at target FPR
            clean_sorted = np.sort(clean_probs)
            # Index where top fraction is target_fpr
            k = int(np.floor((1.0 - self.target_fpr) * len(clean_sorted)))
            k = min(max(k, 0), len(clean_sorted) - 1)
            self.decision_threshold = float(clean_sorted[k])
            logger.info(f"Calibrated decision threshold at target FPR {self.target_fpr:.2%} -> {self.decision_threshold:.4f}")
        else:
            self.decision_threshold = 0.5

        return self

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to raw model logits."""
        scaled = np.asarray(logits, dtype=np.float64) / max(1e-4, self.temperature)
        probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30.0, 30.0)))
        return probs.astype(np.float32)

    def calibrate_prob(self, raw_prob: float) -> float:
        """Given an uncalibrated sigmoid probability in (0, 1), apply temperature."""
        p = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
        logit = np.log(p / (1.0 - p))
        scaled_logit = logit / max(1e-4, self.temperature)
        calibrated_p = float(1.0 / (1.0 + np.exp(-scaled_logit)))
        return calibrated_p

    def get_verdict_and_risk(self, prob: float) -> Tuple[str, str, Dict[str, Any]]:
        """
        Produce human-interpretable verdict, risk level, and metadata details.
        """
        verdict = "SUSPECTED STEGO" if prob >= self.decision_threshold else "CLEAN"

        if prob < self.risk_thresholds["clean_max"]:
            risk = "BENIGN / CLEAN"
        elif prob < self.risk_thresholds["low_risk"]:
            risk = "LOW RISK"
        elif prob < self.risk_thresholds["medium_risk"]:
            risk = "MEDIUM RISK"
        elif prob < self.risk_thresholds["high_risk"]:
            risk = "HIGH RISK"
        else:
            risk = "CRITICAL RISK"

        details = {
            "calibrated_probability": float(prob),
            "decision_threshold": float(self.decision_threshold),
            "temperature": float(self.temperature),
            "target_fpr": float(self.target_fpr)
        }
        return verdict, risk, details

    def save(self, filepath: Union[str, Path]) -> None:
        """Save calibrator parameters to disk."""
        joblib.dump(self, str(filepath))
        logger.info(f"Saved calibrator to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TemperatureScalingCalibrator":
        """Load calibrator from disk."""
        return joblib.load(str(filepath))
