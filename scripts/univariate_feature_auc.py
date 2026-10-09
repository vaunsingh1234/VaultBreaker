import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from vaultbreaker.utils.config import load_config
from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.io import ensure_dir

logger = get_logger("FeatureDiagnostics")

FEATURE_NAMES = {
    "image": [
        "lsb_mean", "lsb_var", "h_trans", "v_trans", "lsb_entropy", "run_mean", "run_var", "lsb_b1_corr",
        "chi_normalized", "chi_stat_log", "spa_diff", "diag_var", "lag1_corr", "parity_imbalance",
        "mean_pix", "std_pix", "sk_pix", "kt_pix", "hist_diff_mean", "hist_diff_std", "zero_ratio", "sat_ratio",
        "res_h_var", "res_h_kt", "res_h_range", "res_v_var", "res_v_kt", "res_v_range",
        "res_lap_var", "res_lap_kt", "kv_var", "kv_mad", "kv_sk", "kv_kt",
        "spam_diag_0", "spam_diag_1", "spam_diag_2", "spam_diag_3", "spam_diag_4", "spam_diag_5", "spam_diag_6",
        "spam_symm_0", "spam_symm_1", "spam_symm_2", "spam_symm_3", "spam_symm_4", "spam_symm_5", "spam_symm_6",
        "spam_symm_7", "spam_symm_8", "spam_symm_9", "spam_symm_10", "spam_symm_11", "spam_symm_12", "spam_symm_13",
        "spam_symm_14", "spam_symm_15", "spam_symm_16",
        "blockiness_v", "blockiness_h", "blockiness_total",
        "dc_var", "mid_energy", "high_energy", "mid_zero_frac", "mid_skew", "mid_kurt",
        "near_int", "ac_mean_abs", "ac_std", "ac_max", "ac_mad"
    ],
    "audio": [
        "lsb_mean", "lsb_var", "lsb_trans", "lsb_entropy", "run_mean", "run_var",
        "chi_norm", "chi_log", "even_odd_asymmetry", "u_samples_mod4_zero",
        "d1_mean", "d1_var", "d1_sk", "d1_kt", "d2_var", "d2_kt",
        "mad_d1", "ratio_d2_d1", "ent_d1", "ent_d2",
        "amp_mean", "amp_std", "amp_sk", "amp_kt", "zcr_mean", "zcr_std",
        "spec_cent_mean", "spec_cent_std", "spec_roll_mean", "spec_roll_std",
        "spec_flat_mean", "spec_flat_std", "spec_bw_mean", "spec_bw_std",
        "hf_ratio", "hf_energy_log", "flux_mean", "flux_std",
        "spec_cent_median", "spec_cent_iqr",
        "r1", "r2", "r3", "r4",
        "lpc_var", "lpc_sk", "lpc_kt", "lpc_err_ratio",
        "mfcc_mean_0", "mfcc_mean_1", "mfcc_mean_2", "mfcc_mean_3",
        "mfcc_std_0", "mfcc_std_1", "mfcc_std_2", "mfcc_std_3"
    ]
}

def analyze_features_univariate(features_dir: Path, output_dir: Path):
    train_matrix_path = features_dir / "split_train.npz"
    if not train_matrix_path.exists():
        logger.error(f"Train matrix not found at {train_matrix_path}. Run extract_features first.")
        return {}

    ensure_dir(output_dir)
    data = dict(np.load(train_matrix_path, allow_pickle=True))
    results = {}

    md_lines = ["# Univariate Feature AUC Diagnostics (TRAIN Split Only)\n\n"]

    for m_type in ["image", "audio", "video"]:
        x_key = f"X_{m_type}"
        y_key = f"y_{m_type}"
        if x_key not in data:
            continue

        X = data[x_key]
        y = data[y_key]
        n_feats = X.shape[1]
        feat_names = FEATURE_NAMES.get(m_type, [f"feat_{i}" for i in range(n_feats)])

        logger.info(f"\n=======================================================")
        logger.info(f"UNIVARIATE FEATURE AUC RANKING: {m_type.upper()} (TRAIN N={len(y)}, D={n_feats})")
        logger.info(f"=======================================================")

        feat_scores = []
        for j in range(n_feats):
            feat_vals = X[:, j]
            feat_vals = np.nan_to_num(feat_vals, nan=0.0)
            fname = feat_names[j] if j < len(feat_names) else f"feat_{j}"
            if len(np.unique(feat_vals)) <= 1 or len(np.unique(y)) <= 1:
                auc = 0.50
                raw_auc = 0.50
            else:
                try:
                    raw_auc = float(roc_auc_score(y, feat_vals))
                    auc = max(raw_auc, 1.0 - raw_auc)
                except Exception:
                    raw_auc = 0.50
                    auc = 0.50
            feat_scores.append({
                "index": j,
                "name": fname,
                "discriminative_auc": float(round(auc, 4)),
                "raw_auc": float(round(raw_auc, 4))
            })

        feat_scores.sort(key=lambda x: x["discriminative_auc"], reverse=True)
        results[m_type] = {
            "num_samples": int(len(y)),
            "num_features": int(n_feats),
            "rankings": feat_scores
        }

        top_15 = feat_scores[:15]
        bottom_15 = feat_scores[-15:]

        logger.info(f"Top 15 Most Discriminative Features ({m_type}):")
        for rank, item in enumerate(top_15, 1):
            logger.info(f"  #{rank:02d} [{item['index']:02d}] {item['name']:24s} | AUC = {item['discriminative_auc']:.4f} (raw={item['raw_auc']:.4f})")

        logger.info(f"Bottom 15 Least Discriminative Features ({m_type}):")
        for rank, item in enumerate(reversed(bottom_15), 1):
            logger.info(f"  #{rank:02d} [{item['index']:02d}] {item['name']:24s} | AUC = {item['discriminative_auc']:.4f} (raw={item['raw_auc']:.4f})")

        # Markdown section
        md_lines.append(f"## {m_type.upper()} Modality (N={len(y)}, D={n_feats})\n\n")
        md_lines.append("### Top 15 Most Discriminative Features\n\n")
        md_lines.append("| Rank | Index | Feature Name | Discriminative AUC | Raw AUC |\n")
        md_lines.append("| :--- | :--- | :--- | :--- | :--- |\n")
        for rank, item in enumerate(top_15, 1):
            md_lines.append(f"| {rank} | {item['index']} | `{item['name']}` | **{item['discriminative_auc']:.4f}** | {item['raw_auc']:.4f} |\n")

        md_lines.append("\n### Bottom 15 Least Discriminative Features\n\n")
        md_lines.append("| Rank (from bottom) | Index | Feature Name | Discriminative AUC | Raw AUC |\n")
        md_lines.append("| :--- | :--- | :--- | :--- | :--- |\n")
        for rank, item in enumerate(bottom_15, 1):
            md_lines.append(f"| {rank} | {item['index']} | `{item['name']}` | {item['discriminative_auc']:.4f} | {item['raw_auc']:.4f} |\n")
        md_lines.append("\n---\n\n")

    # Save JSON and MD
    json_path = output_dir / "univariate_train_auc.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON diagnostics to {json_path}")

    md_path = output_dir / "univariate_train_auc.md"
    with open(md_path, "w") as f:
        f.writelines(md_lines)
    logger.info(f"Saved Markdown diagnostics to {md_path}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="docs/RESULTS/diagnostics")
    args = parser.parse_args()

    cfg = load_config(args.config)
    feat_dir = Path(cfg.get("paths", {}).get("features_dir", "data/features"))
    out_dir = Path(args.output_dir)
    analyze_features_univariate(feat_dir, out_dir)

if __name__ == "__main__":
    main()
