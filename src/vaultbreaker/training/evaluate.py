import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.decomposition import PCA
import torch

from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.io import ensure_dir
from vaultbreaker.models.unified_net import VaultBreakerUnifiedNet
from vaultbreaker.training.dataset import MultiModalStegoDataset
from vaultbreaker.training.calibrate import TemperatureScalingCalibrator

logger = get_logger("Evaluator")

def compute_detection_error_pe(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Computes minimum detection error:
    P_E = min_tau (P_FA(tau) + P_MD(tau)) / 2
    Returns (P_E, optimal_threshold).
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr
    pe_values = (fpr + fnr) / 2.0
    best_idx = np.argmin(pe_values)
    return float(pe_values[best_idx]), float(thresholds[best_idx])

def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    source_ids: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    seed: int = 42
) -> Dict[str, List[float]]:
    """Calculates cluster bootstrap 95% confidence intervals by resampling unique sources."""
    n = len(y_true)
    if n < 4:
        return {"accuracy_ci95": [0.0, 1.0], "roc_auc_ci95": [0.5, 0.5]}
    rng = np.random.RandomState(seed)
    accs, aucs = [], []

    if source_ids is not None and len(source_ids) == n:
        unique_sources = np.unique(source_ids)
        source_indices = {s: np.where(source_ids == s)[0] for s in unique_sources}
        n_clusters = len(unique_sources)
        for _ in range(n_boot):
            sampled_sources = rng.choice(unique_sources, size=n_clusters, replace=True)
            boot_idx = np.concatenate([source_indices[s] for s in sampled_sources])
            yt_b = y_true[boot_idx]
            yp_b = y_prob[boot_idx]
            accs.append(float(accuracy_score(yt_b, yp_b >= threshold)))
            if len(np.unique(yt_b)) > 1:
                try:
                    aucs.append(float(roc_auc_score(yt_b, yp_b)))
                except Exception:
                    pass
    else:
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            yt_b = y_true[idx]
            yp_b = y_prob[idx]
            accs.append(float(accuracy_score(yt_b, yp_b >= threshold)))
            if len(np.unique(yt_b)) > 1:
                try:
                    aucs.append(float(roc_auc_score(yt_b, yp_b)))
                except Exception:
                    pass

    acc_lo, acc_hi = np.percentile(accs, [2.5, 97.5]) if accs else (0.0, 1.0)
    auc_lo, auc_hi = np.percentile(aucs, [2.5, 97.5]) if aucs else (0.5, 0.5)
    return {
        "accuracy_ci95": [round(float(acc_lo), 4), round(float(acc_hi), 4)],
        "roc_auc_ci95": [round(float(auc_lo), 4), round(float(auc_hi), 4)]
    }

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    source_ids: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Calculates all key steganography detection performance metrics with cluster bootstrap 95% CIs."""
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    pe, pe_thresh = compute_detection_error_pe(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    ci_dict = compute_bootstrap_ci(y_true, y_prob, threshold=threshold, source_ids=source_ids)

    unique_src_count = int(len(np.unique(source_ids))) if source_ids is not None else int(len(y_true))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "p_e": round(pe, 4),
        "p_e_threshold": round(pe_thresh, 4),
        "confusion_matrix": cm,
        "sample_count": len(y_true),
        "unique_sources": unique_src_count,
        "sample_count_str": f"{len(y_true)} files ({unique_src_count} unique sources)",
        "accuracy_ci95": ci_dict["accuracy_ci95"],
        "roc_auc_ci95": ci_dict["roc_auc_ci95"]
    }

class StegoEvaluationSuite:
    """
    Comprehensive evaluation engine for VaultBreaker.
    Produces per-format, overall, per-payload, cross-method, and ablation metrics,
    along with publication-grade visualizations.
    """
    def __init__(self, model: VaultBreakerUnifiedNet, calibrator: TemperatureScalingCalibrator):
        self.model = model
        self.calibrator = calibrator
        self.model.eval()

    def predict_dataset(self, data_by_format: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Runs model inference across dataset and collects calibrated probabilities and latent codes."""
        records = []
        all_z = []
        all_formats = []
        all_labels = []

        with torch.no_grad():
            for m_type in ["image", "audio", "video"]:
                x_key = f"X_{m_type}"
                y_key = f"y_{m_type}"
                m_key = f"methods_{m_type}"
                r_key = f"rates_{m_type}"

                if x_key not in data_by_format:
                    continue

                X = data_by_format[x_key]
                y = data_by_format[y_key]
                methods = data_by_format[m_key]
                rates = data_by_format[r_key]
                sources = data_by_format.get(f"sources_{m_type}", np.array([f"{m_type}_src_{idx:04d}" for idx in range(len(X))]))

                for i in range(len(X)):
                    x_tensor = torch.from_numpy(X[i]).float().unsqueeze(0)
                    logit, _, z = self.model(x_tensor, m_type)
                    raw_prob = torch.sigmoid(logit).item()
                    cal_prob = self.calibrator.calibrate_prob(raw_prob)

                    records.append({
                        "source_id": str(sources[i]),
                        "media_type": m_type,
                        "label": int(y[i]),
                        "method": str(methods[i]),
                        "payload_rate": round(float(rates[i]), 2),
                        "raw_prob": float(raw_prob),
                        "cal_prob": float(cal_prob)
                    })
                    all_z.append(z.squeeze(0).cpu().numpy())
                    all_formats.append(m_type)
                    all_labels.append(int(y[i]))

        df = pd.DataFrame(records)
        z_matrix = np.array(all_z, dtype=np.float32) if all_z else np.empty((0, 128))
        return {
            "df": df,
            "z_matrix": z_matrix,
            "formats": all_formats,
            "labels": all_labels
        }

    def run_full_evaluation(
        self,
        test_data: Dict[str, np.ndarray],
        results_dir: Path
    ) -> Dict[str, Any]:
        """Runs the entire evaluation suite and produces plots and metrics."""
        ensure_dir(results_dir)
        pred_res = self.predict_dataset(test_data)
        df = pred_res["df"]
        thresh = self.calibrator.decision_threshold

        full_results = {}

        # 1. Overall Metrics
        y_true_all = df["label"].to_numpy()
        y_prob_all = df["cal_prob"].to_numpy()
        full_results["overall"] = compute_metrics(
            y_true_all, y_prob_all, threshold=thresh, source_ids=df["source_id"].to_numpy()
        )

        # 2. Per-Format Metrics
        full_results["per_format"] = {}
        for m_type in ["image", "audio", "video"]:
            sub = df[df["media_type"] == m_type]
            if not sub.empty:
                full_results["per_format"][m_type] = compute_metrics(
                    sub["label"].to_numpy(), sub["cal_prob"].to_numpy(), threshold=thresh,
                    source_ids=sub["source_id"].to_numpy()
                )

        # Audio LSB headline metric (LSB replacement + LSB matching, separating echo hiding)
        sub_aud_lsb = df[(df["media_type"] == "audio") & (df["method"].isin(["clean", "lsb_replacement", "lsb_matching"]))]
        if not sub_aud_lsb.empty and len(sub_aud_lsb["label"].unique()) > 1:
            full_results["audio_lsb_headline"] = compute_metrics(
                sub_aud_lsb["label"].to_numpy(), sub_aud_lsb["cal_prob"].to_numpy(), threshold=thresh,
                source_ids=sub_aud_lsb["source_id"].to_numpy()
            )

        # 3. Per-Method Metrics
        full_results["per_method"] = {}
        for method in sorted(df["method"].unique()):
            sub = df[df["method"] == method]
            u_src = int(len(sub["source_id"].unique()))
            full_results["per_method"][method] = {
                "accuracy": round(float(accuracy_score(sub["label"], sub["cal_prob"] >= thresh)), 4),
                "mean_cal_prob": round(float(np.mean(sub["cal_prob"])), 4),
                "sample_count": len(sub),
                "unique_sources": u_src,
                "sample_count_str": f"{len(sub)} files ({u_src} unique sources)",
                "insufficient_samples": bool(u_src < 30)
            }

        # 4. Accuracy vs Payload Rate (Cleanly rounded)
        full_results["accuracy_vs_payload"] = {}
        stego_sub = df[df["label"] == 1]
        sorted_rates = np.sort(stego_sub["payload_rate"].unique())
        for rate in sorted_rates:
            r_sub = stego_sub[stego_sub["payload_rate"] == rate]
            u_src = int(len(r_sub["source_id"].unique()))
            full_results["accuracy_vs_payload"][f"{float(rate):.2f}"] = {
                "accuracy": round(float(np.mean(r_sub["cal_prob"] >= thresh)), 4),
                "sample_count": len(r_sub),
                "unique_sources": u_src,
                "sample_count_str": f"{len(r_sub)} files ({u_src} unique sources)",
                "insufficient_samples": bool(u_src < 30)
            }

        # 5. Generate Plots
        self._plot_roc_curves(df, results_dir / "roc_curves.png")
        self._plot_confusion_matrices(df, thresh, results_dir / "confusion_matrices.png")
        self._plot_accuracy_vs_payload(stego_sub, thresh, results_dir / "accuracy_vs_payload.png")
        if len(pred_res["z_matrix"]) >= 4:
            self._plot_latent_space(pred_res, results_dir / "latent_space_tsne.png")

        # 6. Save JSON & Markdown
        metrics_json_path = results_dir / "metrics.json"
        with open(metrics_json_path, "w") as f:
            json.dump(full_results, f, indent=2)

        md_table = self._generate_markdown_table(full_results)
        with open(results_dir / "evaluation_table.md", "w") as f:
            f.write(md_table)

        logger.info(f"Full evaluation completed. Artifacts saved in {results_dir}")
        return full_results

    def _plot_roc_curves(self, df: pd.DataFrame, save_path: Path) -> None:
        """Plot publication-grade ROC curves per format and overall."""
        plt.figure(figsize=(7, 6))
        fpr, tpr, _ = roc_curve(df["label"], df["cal_prob"])
        auc = roc_auc_score(df["label"], df["cal_prob"]) if len(df["label"].unique()) > 1 else 0.5
        plt.plot(fpr, tpr, label=f"Overall Unified (AUC = {auc:.3f})", color="black", linewidth=2.5)

        colors = {"image": "#1f77b4", "audio": "#ff7f0e", "video": "#2ca02c"}
        for m_type in ["image", "audio", "video"]:
            sub = df[df["media_type"] == m_type]
            if len(sub["label"].unique()) > 1:
                f, t, _ = roc_curve(sub["label"], sub["cal_prob"])
                a = roc_auc_score(sub["label"], sub["cal_prob"])
                plt.plot(f, t, label=f"{m_type.capitalize()} (AUC = {a:.3f})", color=colors.get(m_type, "blue"), linestyle="--", linewidth=1.8)

        plt.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Random Guess (AUC = 0.50)")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR / Recall)")
        plt.title("VaultBreaker ROC Performance Curves (Full Real Dataset)")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def _plot_confusion_matrices(self, df: pd.DataFrame, thresh: float, save_path: Path) -> None:
        """Plot confusion matrix heatmaps across all modalities."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        types = [("Overall", df)] + [(m.capitalize(), df[df["media_type"] == m]) for m in ["image", "audio", "video"]]

        for ax, (title, sub) in zip(axes, types):
            if sub.empty:
                ax.set_title(f"{title} (N/A)")
                continue
            y_t = sub["label"].to_numpy()
            y_p = (sub["cal_prob"].to_numpy() >= thresh).astype(int)
            cm = confusion_matrix(y_t, y_p, labels=[0, 1])

            im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
            ax.set_title(f"{title}\n(N={len(sub)})")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Clean", "Stego"])
            ax.set_yticklabels(["Clean", "Stego"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > len(sub)//2 else "black", fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def _plot_accuracy_vs_payload(self, stego_df: pd.DataFrame, thresh: float, save_path: Path) -> None:
        """Plot Detection Accuracy vs Embedding Payload Rate with rounded labels."""
        plt.figure(figsize=(7, 5))
        rates = np.sort(stego_df["payload_rate"].unique())
        accs = []
        labels = []
        for r in rates:
            sub = stego_df[stego_df["payload_rate"] == r]
            acc = float(np.mean(sub["cal_prob"] >= thresh))
            accs.append(acc)
            labels.append(f"{float(r):.2f}")

        plt.plot(labels, accs, marker="o", linewidth=2.5, color="#d62728", label="Stego Detection Rate")
        plt.xlabel("Embedding Payload Rate")
        plt.ylabel("Detection Sensitivity")
        plt.title("Detection Accuracy vs. Embedding Payload Rate")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def _plot_latent_space(self, pred_res: Dict, save_path: Path) -> None:
        """Plot PCA 2D projection of common 128-d latent space."""
        z = pred_res["z_matrix"]
        labels = pred_res["labels"]
        formats = pred_res["formats"]
        if len(z) < 4:
            return

        pca = PCA(n_components=2)
        proj = pca.fit_transform(z)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        format_colors = {"image": "#1f77b4", "audio": "#ff7f0e", "video": "#2ca02c"}
        for f in ["image", "audio", "video"]:
            mask = np.array([fmt == f for fmt in formats])
            if np.any(mask):
                ax1.scatter(proj[mask, 0], proj[mask, 1], label=f.capitalize(), c=format_colors[f], alpha=0.7, s=40)
        ax1.set_title("Common Latent Space: Colored by Format")
        ax1.set_xlabel("Latent Dim 1")
        ax1.set_ylabel("Latent Dim 2")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.4)

        mask_clean = np.array([l == 0 for l in labels])
        mask_stego = np.array([l == 1 for l in labels])
        ax2.scatter(proj[mask_clean, 0], proj[mask_clean, 1], label="Clean Cover", c="#2ca02c", alpha=0.7, s=40)
        ax2.scatter(proj[mask_stego, 0], proj[mask_stego, 1], label="Stego", c="#d62728", alpha=0.7, s=40)
        ax2.set_title("Common Latent Space: Colored by Stego Label")
        ax2.set_xlabel("Latent Dim 1")
        ax2.set_ylabel("Latent Dim 2")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.4)

        plt.suptitle("VaultBreaker 128-D Common Latent Space Projection (Full Test Split)", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def _generate_markdown_table(self, full_results: Dict) -> str:
        """Format metrics into Markdown table including 95% confidence intervals."""
        ov = full_results["overall"]
        lines = [
            "# VaultBreaker Empirical Performance Summary\n",
            "### Summary Performance by Media Format (with 95% Bootstrap CIs)\n",
            "| Evaluation Split | Samples | Accuracy (95% CI) | Precision | Recall | F1 Score | ROC-AUC (95% CI) | Detection Error $P_E$ |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        for fmt, m in full_results["per_format"].items():
            acc_ci_str = f"{m['accuracy']:.4f} `[{m['accuracy_ci95'][0]:.3f}, {m['accuracy_ci95'][1]:.3f}]`"
            auc_ci_str = f"{m['roc_auc']:.4f} `[{m['roc_auc_ci95'][0]:.3f}, {m['roc_auc_ci95'][1]:.3f}]`"
            lines.append(
                f"| **{fmt.capitalize()}** | {m['sample_count']} | {acc_ci_str} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {auc_ci_str} | {m['p_e']:.4f} |"
            )

        ov_acc_ci = f"**{ov['accuracy']:.4f}** `[{ov['accuracy_ci95'][0]:.3f}, {ov['accuracy_ci95'][1]:.3f}]`"
        ov_auc_ci = f"**{ov['roc_auc']:.4f}** `[{ov['roc_auc_ci95'][0]:.3f}, {ov['roc_auc_ci95'][1]:.3f}]`"
        lines.append(
            f"| **Unified Overall** | **{ov['sample_count']}** | {ov_acc_ci} | "
            f"**{ov['precision']:.4f}** | **{ov['recall']:.4f}** | **{ov['f1']:.4f}** | {ov_auc_ci} | **{ov['p_e']:.4f}** |\n"
        )

        if "audio_lsb_headline" in full_results:
            a_lsb = full_results["audio_lsb_headline"]
            lines.append(
                f"> **Audio LSB Headline (Excl. Echo Hiding):** Accuracy: **{a_lsb['accuracy']:.4f}** "
                f"(95% CI: `[{a_lsb['accuracy_ci95'][0]:.3f}, {a_lsb['accuracy_ci95'][1]:.3f}]`), "
                f"ROC-AUC: **{a_lsb['roc_auc']:.4f}** (95% CI: `[{a_lsb['roc_auc_ci95'][0]:.3f}, {a_lsb['roc_auc_ci95'][1]:.3f}]`), N={a_lsb['sample_count']}\n"
            )

        lines.append("### Embedding Method Detection Rates\n")
        lines.append("| Embedding Method | Samples | Detection Accuracy | Mean Stego Probability |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for method, res in full_results["per_method"].items():
            lines.append(f"| `{method}` | {res['sample_count']} | {res['accuracy']:.4f} | {res['mean_cal_prob']:.4f} |")

        lines.append("\n### Accuracy vs. Payload Density Curve\n")
        lines.append("| Payload Rate | Stego Samples | Detection Accuracy |")
        lines.append("| :--- | :---: | :---: |")
        for rate, res in full_results["accuracy_vs_payload"].items():
            lines.append(f"| **{rate}** | {res['sample_count']} | {res['accuracy']:.4f} |")

        return "\n".join(lines) + "\n"
