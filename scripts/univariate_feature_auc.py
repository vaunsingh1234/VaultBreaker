#!/usr/bin/env python3
"""Univariate Feature AUC Diagnostic Script for VaultBreaker.
Evaluates individual feature discriminative capability (ROC-AUC) against stego labels,
identifying high-signal features and noisy dimensions for each modality.
"""

import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from vaultbreaker.utils.config import load_config
from vaultbreaker.utils.logger import get_logger

logger = get_logger("FeatureDiagnostics")

def analyze_features_univariate(features_dir: Path):
    train_matrix_path = features_dir / "split_train.npz"
    if not train_matrix_path.exists():
        logger.error(f"Train matrix not found at {train_matrix_path}. Run extract_features first.")
        return {}

    data = dict(np.load(train_matrix_path, allow_pickle=True))
    results = {}

    for m_type in ["image", "audio", "video"]:
        x_key = f"X_{m_type}"
        y_key = f"y_{m_type}"
        if x_key not in data:
            continue

        X = data[x_key]
        y = data[y_key]
        n_feats = X.shape[1]
        logger.info(f"\n=======================================================")
        logger.info(f"UNIVARIATE FEATURE AUC RANKING: {m_type.upper()} (N={len(y)}, D={n_feats})")
        logger.info(f"=======================================================")

        feat_scores = []
        for j in range(n_feats):
            feat_vals = X[:, j]
            # Replace NaNs
            feat_vals = np.nan_to_num(feat_vals, nan=0.0)
            if len(np.unique(feat_vals)) <= 1 or len(np.unique(y)) <= 1:
                auc = 0.50
            else:
                try:
                    auc = float(roc_auc_score(y, feat_vals))
                    # Distance from chance: max(auc, 1 - auc)
                    if auc < 0.5:
                        auc = 1.0 - auc
                except Exception:
                    auc = 0.50
            feat_scores.append((j, auc))

        feat_scores.sort(key=lambda x: x[1], reverse=True)
        results[m_type] = feat_scores

        logger.info("Top 10 Most Discriminative Features:")
        for rank, (f_idx, auc) in enumerate(feat_scores[:10], 1):
            logger.info(f"  #{rank:02d} Feature [{f_idx:02d}]: Univariate ROC-AUC = {auc:.4f}")

        # Summary statistics
        aucs_arr = np.array([s[1] for s in feat_scores])
        sig_count = np.sum(aucs_arr >= 0.65)
        logger.info(f"Summary for {m_type.capitalize()}:")
        logger.info(f"  Mean Feature AUC: {np.mean(aucs_arr):.4f}")
        logger.info(f"  Max Feature AUC:  {np.max(aucs_arr):.4f}")
        logger.info(f"  Features with strong signal (AUC >= 0.65): {sig_count}/{n_feats}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    feat_dir = Path(cfg.get("paths", {}).get("features_dir", "data/features"))
    analyze_features_univariate(feat_dir)

if __name__ == "__main__":
    main()
