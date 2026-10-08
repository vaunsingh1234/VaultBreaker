import json
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
from sklearn.manifold import TSNE
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
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr # Missed detection rate P_MD
    pe_values = (fpr + fnr) / 2.0
    best_idx = np.argmin(pe_values)
    return float(pe_values[best_idx]), float(thresholds[best_idx])

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Calculates all key steganography detection performance metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    pe, pe_thresh = compute_detection_error_pe(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "p_e": pe,
        "p_e_threshold": pe_thresh,
        "confusion_matrix": cm,
        "sample_count": len(y_true)
    }

class StegoEvaluationSuite:
    """
    Comprehensive evaluation engine for VaultBreaker.
    Produces per-format, overall, per-payload, cross-method, and sanity check metrics,
    along with publication-grade visualizations.
    """
    def __init__(self, model: VaultBreakerUnifiedNet, calibrator: TemperatureScalingCalibrator):
        self.model = model
        self.calibrator = calibrator
        self.model.eval()

    def predict_dataset(self, data_by_format: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Runs model inference across test dataset and collects probabilities and latent codes.
        """
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

                for i in range(len(X)):
                    x_tensor = torch.from_numpy(X[i]).float().unsqueeze(0)
                    logit, _, z = self.model(x_tensor, m_type)
                    raw_prob = torch.sigmoid(logit).item()
                    cal_prob = self.calibrator.calibrate_prob(raw_prob)

                    records.append({
                        "media_type": m_type,
                        "label": int(y[i]),
                        "method": str(methods[i]),
                        "payload_rate": float(rates[i]),
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
        full_results["overall"] = compute_metrics(y_true_all, y_prob_all, threshold=thresh)

        # 2. Per-Format Metrics
        full_results["per_format"] = {}
        for m_type in ["image", "audio", "video"]:
            sub = df[df["media_type"] == m_type]
            if not sub.empty:
                full_results["per_format"][m_type] = compute_metrics(
                    sub["label"].to_numpy(), sub["cal_prob"].to_numpy(), threshold=thresh
                )

        # 3. Per-Method Metrics
        full_results["per_method"] = {}
        for method in df["method"].unique():
            sub = df[df["method"] == method]
            full_results["per_method"][method] = {
                "accuracy": float(accuracy_score(sub["label"], sub["cal_prob"] >= thresh)),
                "mean_cal_prob": float(np.mean(sub["cal_prob"])),
                "sample_count": len(sub)
            }

        # 4. Accuracy vs Payload Rate
        full_results["accuracy_vs_payload"] = {}
        stego_sub = df[df["label"] == 1]
        for rate in np.sort(stego_sub["payload_rate"].unique()):
            r_sub = stego_sub[stego_sub["payload_rate"] == rate]
            full_results["accuracy_vs_payload"][str(rate)] = {
                "accuracy": float(np.mean(r_sub["cal_prob"] >= thresh)),
                "sample_count": len(r_sub)
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
        """Plot ROC curves per format and overall."""
        plt.figure(figsize=(7, 6))
        # Overall
        fpr, tpr, _ = roc_curve(df["label"], df["cal_prob"])
        auc = roc_auc_score(df["label"], df["cal_prob"]) if len(df["label"].unique()) > 1 else 0.5
        plt.plot(fpr, tpr, label=f"Overall (AUC = {auc:.3f})", color="black", linewidth=2.5)

        colors = {"image": "#1f77b4", "audio": "#ff7f0e", "video": "#2ca02c"}
        for m_type in ["image", "audio", "video"]:
            sub = df[df["media_type"] == m_type]
            if len(sub["label"].unique()) > 1:
                f, t, _ = roc_curve(sub["label"], sub["cal_prob"])
                a = roc_auc_score(sub["label"], sub["cal_prob"])
                plt.plot(f, t, label=f"{m_type.capitalize()} (AUC = {a:.3f})", color=colors.get(m_type, "blue"), linestyle="--")

        plt.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Random Guess (AUC = 0.50)")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR / Recall)")
        plt.title("VaultBreaker ROC Performance Curves")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def _plot_confusion_matrices(self, df: pd.DataFrame, thresh: float, save_path: Path) -> None:
        """Plot confusion matrix heatmaps."""
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
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red" if cm[i, j] > len(sub)//2 else "black", fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def _plot_accuracy_vs_payload(self, stego_df: pd.DataFrame, thresh: float, save_path: Path) -> None:
        """Plot Detection Accuracy vs Embedding Payload Rate."""
        plt.figure(figsize=(7, 5))
        rates = np.sort(stego_df["payload_rate"].unique())
        accs = []
        for r in rates:
            sub = stego_df[stego_df["payload_rate"] == r]
            acc = float(np.mean(sub["cal_prob"] >= thresh))
            accs.append(acc)

        plt.plot(rates, accs, marker="o", linewidth=2.5, color="#d62728", label="Stego Detection Rate")
        plt.xlabel("Embedding Payload Rate (bpp / bps)")
        plt.ylabel("Detection Accuracy (Sensitivity)")
        plt.title("Detection Accuracy vs. Payload Density Curve")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def _plot_latent_space(self, pred_res: Dict, save_path: Path) -> None:
        """Plot 2D projection (PCA/t-SNE) of the shared 128-d latent space."""
        z = pred_res["z_matrix"]
        labels = pred_res["labels"]
        formats = pred_res["formats"]

        n_samples = len(z)
        if n_samples < 4:
            return

        # Use PCA for reliable projection regardless of sample size
        pca = PCA(n_components=2)
        proj = pca.fit_transform(z)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Colored by Format
        format_colors = {"image": "#1f77b4", "audio": "#ff7f0e", "video": "#2ca02c"}
        for f in ["image", "audio", "video"]:
            mask = np.array([fmt == f for fmt in formats])
            if np.any(mask):
                ax1.scatter(proj[mask, 0], proj[mask, 1], label=f.capitalize(), c=format_colors[f], alpha=0.7, s=40)
        ax1.set_title("Shared Latent Space: Colored by Format")
        ax1.set_xlabel("Latent Dim 1")
        ax1.set_ylabel("Latent Dim 2")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.4)

        # 2. Colored by Stego Label
        mask_clean = np.array([l == 0 for l in labels])
        mask_stego = np.array([l == 1 for l in labels])
        ax2.scatter(proj[mask_clean, 0], proj[mask_clean, 1], label="Clean Cover", c="#2ca02c", alpha=0.7, s=40)
        ax2.scatter(proj[mask_stego, 0], proj[mask_stego, 1], label="Stego", c="#d62728", alpha=0.7, s=40)
        ax2.set_title("Shared Latent Space: Colored by Stego Label")
        ax2.set_xlabel("Latent Dim 1")
        ax2.set_ylabel("Latent Dim 2")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.4)

        plt.suptitle("VaultBreaker 128-D Shared Latent Space Projection", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def _generate_markdown_table(self, full_results: Dict) -> str:
        """Format metrics into a clean Markdown table."""
        ov = full_results["overall"]
        lines = [
            "# VaultBreaker Evaluation Report\n",
            "### Summary Performance by Media Format\n",
            "| Evaluation Split | Samples | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Detection Error $P_E$ |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        for fmt, m in full_results["per_format"].items():
            lines.append(
                f"| **{fmt.capitalize()}** | {m['sample_count']} | {m['accuracy']:.4f} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['p_e']:.4f} |"
            )

        lines.append(
            f"| **Unified Overall** | **{ov['sample_count']}** | **{ov['accuracy']:.4f}** | "
            f"**{ov['precision']:.4f}** | **{ov['recall']:.4f}** | **{ov['f1']:.4f}** | **{ov['roc_auc']:.4f}** | **{ov['p_e']:.4f}** |\n"
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
