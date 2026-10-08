import os
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.io import ensure_dir

logger = get_logger("Baselines")

class BaselineModelsManager:
    """
    Trains and evaluates classical machine learning baselines
    (Logistic Regression, SVM RBF, Random Forest, Gradient Boosting)
    per format on handcrafted feature vectors.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}

    def get_models(self) -> Dict[str, Any]:
        """Instantiate baseline classifiers with probability support."""
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=self.random_state, class_weight="balanced"
            ),
            "SVM_RBF": SVC(
                kernel="rbf", probability=True, random_state=self.random_state, class_weight="balanced"
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=self.random_state, class_weight="balanced"
            ),
            "GradientBoosting": HistGradientBoostingClassifier(
                max_iter=100, max_depth=6, learning_rate=0.05, random_state=self.random_state
            )
        }

    def train_and_eval(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        media_type: str,
        save_dir: Optional[Path] = None
    ) -> Dict[str, Dict[str, float]]:
        """Train all baseline models for a specific media type and compute validation metrics."""
        results = {}
        classifiers = self.get_models()

        if save_dir:
            ensure_dir(save_dir)

        for name, clf in classifiers.items():
            logger.info(f"Training baseline {name} for format [{media_type}]...")
            clf.fit(X_train, y_train)

            # Predict on val
            y_pred = clf.predict(X_val)
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_val)[:, 1]
            else:
                y_prob = clf.decision_function(X_val)

            acc = float(accuracy_score(y_val, y_pred))
            f1 = float(f1_score(y_val, y_pred, zero_division=0))
            try:
                auc = float(roc_auc_score(y_val, y_prob))
            except Exception:
                auc = 0.5

            results[name] = {
                "accuracy": acc,
                "f1": f1,
                "roc_auc": auc
            }
            logger.info(f"  {name} [{media_type}] -> Acc: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

            model_key = f"{media_type}_{name}"
            self.models[model_key] = clf
            if save_dir:
                joblib.dump(clf, save_dir / f"baseline_{model_key}.joblib")

        return results

    def predict_proba(self, model_key: str, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using trained baseline."""
        clf = self.models[model_key]
        return clf.predict_proba(X)[:, 1]
