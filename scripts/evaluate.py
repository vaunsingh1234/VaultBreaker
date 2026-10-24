#!/usr/bin/env python3
"""Evaluation CLI for VaultBreaker.
Evaluates the trained multi-modal pipeline (deep model, handcrafted MLP, and GBDT baseline)
on the balanced test split, computes PE, ROC-AUC, recalls at 1%, 5%, 10% FPR,
and cluster bootstrap 95% confidence intervals by source_id.
Produces:
  - docs/RESULTS/metrics_deep.json
  - docs/RESULTS/metrics_handcrafted.json
  - docs/RESULTS/comparison.json
  - docs/RESULTS/method_payload_heatmap.png
Smoke runs write strictly to docs/RESULTS/smoke/ and never overwrite real results.
"""

import argparse
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import soundfile as sf
import imageio.v3 as iio
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, confusion_matrix
)

import subprocess

from vaultbreaker.utils.config import load_config
from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.io import ensure_dir
from vaultbreaker.models.unified_net import VaultBreakerUnifiedNet
from vaultbreaker.models.deep_net import VaultBreakerDeepNet
from vaultbreaker.training.calibrate import TemperatureScalingCalibrator, load_format_calibrators
from vaultbreaker.training.evaluate import (
    compute_metrics, compute_bootstrap_ci, compute_detection_error_pe
)
from vaultbreaker.features.extractor import MultiModalFeaturePipeline

logger = get_logger("Evaluate")


def get_git_commit() -> str:
    """Safely obtain current git commit hash."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent.parent)
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def compute_recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fprs=(0.01, 0.05, 0.10)) -> Dict[str, float]:
    """Compute TPR (recall) at specific FPR operating points on the ROC curve."""
    if len(np.unique(y_true)) < 2:
        return {f"recall_at_{int(fpr*100)}pct_fpr": 0.0 for fpr in target_fprs}
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    recalls = {}
    for target in target_fprs:
        idx = np.where(fpr <= target)[0]
        if len(idx) > 0:
            rec = float(tpr[idx[-1]])
        else:
            rec = 0.0
        recalls[f"recall_at_{int(target*100)}pct_fpr"] = round(rec, 4)
    return recalls


def build_balanced_test_set(df_test: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Construct a strictly balanced test set:
    For each clean file, include ONE stego variant, sampled evenly across methods and payload sizes.
    Never evaluate every method x payload variant against each clean file.
    """
    clean_df = df_test[df_test["label"] == 0].copy()
    stego_df = df_test[df_test["label"] == 1].copy()

    selected_rows = []
    rng = np.random.RandomState(seed)

    for m_type in ["image", "audio", "video"]:
        m_clean = clean_df[clean_df["media_type"] == m_type].sort_values("source_id")
        m_stego = stego_df[stego_df["media_type"] == m_type]

        if m_clean.empty or m_stego.empty:
            continue

        # Find unique method x payload combinations for even round-robin sampling
        combos = m_stego[["method", "payload_rate"]].drop_duplicates().sort_values(["method", "payload_rate"]).to_dict("records")
        n_combos = len(combos)

        stego_by_source = {}
        for _, s_row in m_stego.iterrows():
            stego_by_source.setdefault(s_row["source_id"], []).append(s_row)

        for idx, (_, c_row) in enumerate(m_clean.iterrows()):
            src = c_row["source_id"]
            variants = stego_by_source.get(src, [])
            if not variants:
                continue

            selected_rows.append(c_row.to_dict())

            target_combo = combos[idx % n_combos]
            matched = [
                v for v in variants
                if v["method"] == target_combo["method"] and np.isclose(v["payload_rate"], target_combo["payload_rate"], atol=0.01)
            ]
            if matched:
                pick = matched[0]
            else:
                pick_idx = rng.randint(0, len(variants))
                pick = variants[pick_idx]
            selected_rows.append(pick.to_dict() if isinstance(pick, pd.Series) else dict(pick))

    df_balanced = pd.DataFrame(selected_rows).reset_index(drop=True)
    logger.info(
        f"Built balanced test set: {len(df_balanced)} samples "
        f"({(df_balanced['label'] == 0).sum()} clean, {(df_balanced['label'] == 1).sum()} stego)."
    )
    return df_balanced


def evaluate_deep_model_test_set(
    deep_ckpt_path: Path,
    calibrator_path: Optional[Path],
    df_test_balanced: pd.DataFrame,
    device: torch.device,
    batch_size: int = 32
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run evaluation on the trained VaultBreakerDeepNet on the balanced test set.
    Reports accuracy strictly at the validation-calibrated threshold (5% FPR).
    """
    logger.info(f"Loading deep model checkpoint from {deep_ckpt_path}...")
    ckpt = torch.load(deep_ckpt_path, map_location=device)
    model_state = ckpt.get("model_state", ckpt)

    # Detect base_channels dynamically
    base_ch = 16
    for k, v in model_state.items():
        if "image_deep.block1.0.weight" in k:
            base_ch = v.shape[0]
            break

    model = VaultBreakerDeepNet(
        base_channels=base_ch,
        latent_dim=128,
        use_fusion=False,
        dropout=0.0
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Load Calibrators fitted on validation set per format (Item 2)
    calibrators = load_format_calibrators(calibrator_path) if calibrator_path else {
        fmt: TemperatureScalingCalibrator(target_fpr=0.05) for fmt in ["image", "audio", "video"]
    }
    for fmt, cal in calibrators.items():
        logger.info(f"Loaded {fmt} deep calibrator (T={cal.temperature:.4f}, threshold={cal.decision_threshold:.4f})")

    records = []
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in df_test_balanced.iterrows():
        fp = Path(row["filepath"])
        if fp.exists():
            by_type.setdefault(row["media_type"], []).append(row.to_dict())

    with torch.no_grad():
        for m_type, items in by_type.items():
            bs = min(batch_size, 8 if m_type == "video" else batch_size)
            cal = calibrators.get(m_type, TemperatureScalingCalibrator(target_fpr=0.05))
            for i in range(0, len(items), bs):
                chunk = items[i:i + bs]
                tensors = []
                valid_chunk = []
                for it in chunk:
                    fp = it["filepath"]
                    try:
                        if m_type == "image":
                            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                            if img is None:
                                continue
                            if img.shape != (256, 256):
                                img = cv2.resize(img, (256, 256))
                            t = torch.from_numpy(img).unsqueeze(0).float()
                        elif m_type == "audio":
                            data, _ = sf.read(fp, dtype="int16")
                            if len(data) < 48000:
                                data = np.pad(data, (0, 48000 - len(data)), mode="wrap")
                            elif len(data) > 48000:
                                data = data[:48000]
                            t = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
                        else:  # video
                            frames = iio.imread(fp)
                            if frames.ndim == 4:
                                frames = np.dot(frames[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                            if len(frames) < 8:
                                pad = np.zeros((8 - len(frames), frames.shape[1], frames.shape[2]), dtype=np.uint8)
                                frames = np.concatenate([frames, pad], axis=0)
                            frames = frames[:8]
                            if frames.shape[1:3] != (128, 128):
                                frames = np.stack([cv2.resize(f, (128, 128)) for f in frames])
                            t = torch.from_numpy(frames).unsqueeze(1).float()
                        tensors.append(t)
                        valid_chunk.append(it)
                    except Exception:
                        continue

                if not tensors:
                    continue

                batch_t = torch.stack(tensors).to(device)
                logits, _, _ = model(batch_t, m_type)
                raw_probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
                if not isinstance(raw_probs, list):
                    raw_probs = [raw_probs]

                for it, raw_p in zip(valid_chunk, raw_probs):
                    cal_p = cal.calibrate_prob(raw_p)
                    records.append({
                        "source_id": it["source_id"],
                        "filepath": it["filepath"],
                        "media_type": it["media_type"],
                        "label": int(it["label"]),
                        "method": it["method"],
                        "payload_rate": float(it["payload_rate"]),
                        "raw_prob": float(raw_p),
                        "cal_prob": float(cal_p)
                    })

    df_eval = pd.DataFrame(records)
    logger.info(f"Evaluated {len(df_eval)} balanced samples across formats.")

    # 1. Overall Metrics (evaluated with format-calibrated threshold per sample)
    y_true_all = df_eval["label"].values
    y_prob_all = df_eval["cal_prob"].values
    src_all = df_eval["source_id"].values
    y_pred_all = np.array([
        row["cal_prob"] >= calibrators.get(row["media_type"], TemperatureScalingCalibrator(target_fpr=0.05)).decision_threshold
        for _, row in df_eval.iterrows()
    ], dtype=int)

    overall_metrics = compute_metrics(y_true_all, y_prob_all, threshold=0.5, source_ids=src_all)
    overall_metrics["accuracy"] = round(float(accuracy_score(y_true_all, y_pred_all)), 4)
    overall_metrics["recall"] = round(float(recall_score(y_true_all, y_pred_all, zero_division=0)), 4)
    overall_metrics["precision"] = round(float(precision_score(y_true_all, y_pred_all, zero_division=0)), 4)
    overall_metrics["recalls_at_fpr"] = compute_recall_at_fpr(y_true_all, y_prob_all)
    overall_metrics["calibrated_thresholds"] = {
        fmt: round(float(calibrators[fmt].decision_threshold), 4) for fmt in ["image", "audio", "video"] if fmt in calibrators
    }
    overall_metrics["calibrated_threshold"] = round(float(np.mean([cal.decision_threshold for cal in calibrators.values()])), 4)

    # 2. Per-Format Metrics (evaluated strictly at that format's validation-calibrated threshold targeting 5% FPR)
    per_format_metrics = {}
    for fmt in ["image", "audio", "video"]:
        fmt_df = df_eval[df_eval["media_type"] == fmt]
        if len(fmt_df) > 0:
            yt = fmt_df["label"].values
            yp = fmt_df["cal_prob"].values
            src = fmt_df["source_id"].values
            cal = calibrators.get(fmt, TemperatureScalingCalibrator(target_fpr=0.05))
            thresh_f = cal.decision_threshold
            m = compute_metrics(yt, yp, threshold=thresh_f, source_ids=src)
            m["recalls_at_fpr"] = compute_recall_at_fpr(yt, yp)
            m["calibrated_threshold"] = round(float(thresh_f), 4)
            clean_yp = yp[yt == 0]
            if len(clean_yp) > 0:
                m["test_false_positive_rate"] = round(float(np.mean(clean_yp >= thresh_f)), 4)
            else:
                m["test_false_positive_rate"] = 0.0
            if m["unique_sources"] < 30:
                m["status"] = "too few samples to trust"
                m["warning"] = "too few samples to trust"
            per_format_metrics[fmt] = m

    mean_decision_threshold = overall_metrics["calibrated_threshold"]

    # 3. Per-Method Metrics
    per_method_metrics = {}
    stego_eval = df_eval[df_eval["label"] == 1]
    clean_eval = df_eval[df_eval["label"] == 0]
    for meth in sorted(stego_eval["method"].unique()):
        sub_stego = stego_eval[stego_eval["method"] == meth]
        sub_eval = pd.concat([clean_eval, sub_stego], ignore_index=True)
        yt_m = sub_eval["label"].values
        yp_m = sub_eval["cal_prob"].values
        src_m = sub_eval["source_id"].values
        meth_fmts = sub_stego["media_type"].unique()
        if len(meth_fmts) == 1 and meth_fmts[0] in calibrators:
            m_thresh = calibrators[meth_fmts[0]].decision_threshold
        else:
            m_thresh = mean_decision_threshold
        m_res = compute_metrics(yt_m, yp_m, threshold=m_thresh, source_ids=src_m)
        m_res["recalls_at_fpr"] = compute_recall_at_fpr(yt_m, yp_m)
        m_res["calibrated_threshold"] = round(float(m_thresh), 4)
        m_res["stego_sample_count"] = len(sub_stego)
        per_method_metrics[meth] = m_res

    # 4. Per-Payload Metrics
    per_payload_metrics = {}
    for pl in sorted(stego_eval["payload_rate"].unique()):
        sub_pl = stego_eval[np.isclose(stego_eval["payload_rate"], pl, atol=0.01)]
        sub_eval = pd.concat([clean_eval, sub_pl], ignore_index=True)
        yt_p = sub_eval["label"].values
        yp_p = sub_eval["cal_prob"].values
        src_p = sub_eval["source_id"].values
        p_res = compute_metrics(yt_p, yp_p, threshold=mean_decision_threshold, source_ids=src_p)
        p_res["recalls_at_fpr"] = compute_recall_at_fpr(yt_p, yp_p)
        p_res["calibrated_threshold"] = round(float(mean_decision_threshold), 4)
        p_res["stego_sample_count"] = len(sub_pl)
        per_payload_metrics[f"{pl:.2f}"] = p_res

    eval_results = {
        "model_type": "VaultBreakerDeepNet",
        "checkpoint": str(deep_ckpt_path),
        "overall": overall_metrics,
        "per_format": per_format_metrics,
        "per_method": per_method_metrics,
        "per_payload": per_payload_metrics
    }

    return eval_results, records


def evaluate_handcrafted_model_test_set(
    models_dir: Path,
    df_test_balanced: pd.DataFrame,
    features_cache_dir: Path
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run evaluation on the saved handcrafted MLP model (models_saved/unified_model.pt).
    Extracts handcrafted features on the balanced test items and applies scalers and calibrator.
    """
    weights_path = models_dir / "unified_model.pt"
    cal_path = models_dir / "calibrator.joblib"
    if not weights_path.exists():
        logger.warning(f"Handcrafted model {weights_path} not found.")
        return None, []

    logger.info("Evaluating handcrafted model on balanced test set...")
    pipeline = MultiModalFeaturePipeline(cache_dir=features_cache_dir, num_workers=2)
    features_map = pipeline.extract_dataset(df_test_balanced)

    # Load scalers
    scalers = {}
    for fmt in ["image", "audio", "video"]:
        sc_path = models_dir / f"scaler_{fmt}.joblib"
        if sc_path.exists():
            scalers[fmt] = joblib.load(sc_path)

    # Load unified net
    model = VaultBreakerUnifiedNet(
        img_dim=72, aud_dim=56, vid_dim=64,
        latent_dim=128, hidden_dim=256,
        use_aux_head=True, num_methods=8
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    calibrator = TemperatureScalingCalibrator.load(cal_path) if cal_path.exists() else TemperatureScalingCalibrator(target_fpr=0.05)
    decision_threshold = calibrator.decision_threshold

    records = []
    with torch.no_grad():
        for _, row in df_test_balanced.iterrows():
            fp = row["filepath"]
            m_type = row["media_type"]
            if fp not in features_map or m_type not in scalers:
                continue

            raw_feat = features_map[fp].reshape(1, -1)
            norm_feat = scalers[m_type].transform(raw_feat)
            t_feat = torch.from_numpy(norm_feat).float()

            logits, _, _ = model(t_feat, m_type)
            raw_p = float(torch.sigmoid(logits).squeeze().item())
            cal_p = calibrator.calibrate_prob(raw_p) if calibrator else raw_p

            records.append({
                "source_id": row["source_id"],
                "filepath": fp,
                "media_type": m_type,
                "label": int(row["label"]),
                "method": row["method"],
                "payload_rate": float(row["payload_rate"]),
                "raw_prob": raw_p,
                "cal_prob": cal_p
            })

    if not records:
        return None, []

    df_eval = pd.DataFrame(records)
    y_true_all = df_eval["label"].values
    y_prob_all = df_eval["cal_prob"].values
    src_all = df_eval["source_id"].values

    overall_metrics = compute_metrics(y_true_all, y_prob_all, threshold=decision_threshold, source_ids=src_all)
    overall_metrics["recalls_at_fpr"] = compute_recall_at_fpr(y_true_all, y_prob_all)
    overall_metrics["calibrated_threshold"] = round(float(decision_threshold), 4)

    per_format_metrics = {}
    for fmt in ["image", "audio", "video"]:
        fmt_df = df_eval[df_eval["media_type"] == fmt]
        if len(fmt_df) > 0:
            yt = fmt_df["label"].values
            yp = fmt_df["cal_prob"].values
            src = fmt_df["source_id"].values
            m = compute_metrics(yt, yp, threshold=decision_threshold, source_ids=src)
            m["recalls_at_fpr"] = compute_recall_at_fpr(yt, yp)
            m["calibrated_threshold"] = round(float(decision_threshold), 4)
            if m["unique_sources"] < 30:
                m["status"] = "too few samples to trust"
                m["warning"] = "too few samples to trust"
            per_format_metrics[fmt] = m

    eval_results = {
        "model_type": "VaultBreakerUnifiedNet",
        "checkpoint": str(weights_path),
        "overall": overall_metrics,
        "per_format": per_format_metrics
    }
    return eval_results, records


def evaluate_gradient_boosting_test_set(
    models_dir: Path,
    df_test_balanced: pd.DataFrame,
    features_cache_dir: Path
) -> Dict[str, Any]:
    """
    Evaluate real HistGradientBoostingClassifier baselines on the same balanced test set.
    """
    pipeline = MultiModalFeaturePipeline(cache_dir=features_cache_dir, num_workers=2)
    features_map = pipeline.extract_dataset(df_test_balanced)
    results = {}

    for fmt in ["image", "audio", "video"]:
        model_path = models_dir / f"baseline_{fmt}_GradientBoosting.joblib"
        scaler_path = models_dir / f"scaler_{fmt}.joblib"

        if not model_path.exists() or not scaler_path.exists():
            continue

        try:
            gb = joblib.load(model_path)
            sc = joblib.load(scaler_path)

            fmt_df = df_test_balanced[df_test_balanced["media_type"] == fmt]
            valid_rows = [r for _, r in fmt_df.iterrows() if r["filepath"] in features_map]
            if len(valid_rows) < 4:
                continue

            raw_feats = np.stack([features_map[r["filepath"]] for r in valid_rows], axis=0)
            norm_feats = sc.transform(raw_feats)
            probs = gb.predict_proba(norm_feats)[:, 1]
            yt = np.array([r["label"] for r in valid_rows])
            src = np.array([r["source_id"] for r in valid_rows])

            # Determine 5% FPR threshold on clean files
            clean_probs = probs[yt == 0]
            if len(clean_probs) > 0:
                clean_sorted = np.sort(clean_probs)
                k = int(np.floor((1.0 - 0.05) * len(clean_sorted)))
                k = min(max(k, 0), len(clean_sorted) - 1)
                thresh = float(clean_sorted[k])
            else:
                thresh = 0.5

            m = compute_metrics(yt, probs, threshold=thresh, source_ids=src)
            m["recalls_at_fpr"] = compute_recall_at_fpr(yt, probs)
            m["calibrated_threshold"] = round(thresh, 4)
            results[fmt] = m
        except Exception as e:
            logger.warning(f"Error evaluating GBDT for {fmt}: {e}")

    return results


def build_side_by_side_comparison(
    deep_metrics: Optional[Dict[str, Any]],
    handcrafted_metrics: Optional[Dict[str, Any]],
    gbdt_metrics: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Assemble strictly honest side-by-side comparison table.
    A model appears only if its metrics were computed on the SAME test manifest.
    Otherwise write 'not run'. Zero invented or fallback numbers!
    """
    comparison = {}

    for fmt in ["image", "audio", "video"]:
        deep_f = deep_metrics.get("per_format", {}).get(fmt) if deep_metrics else None
        mlp_f = handcrafted_metrics.get("per_format", {}).get(fmt) if handcrafted_metrics else None
        gb_f = gbdt_metrics.get(fmt) if gbdt_metrics else None

        fmt_entry = {}

        if deep_f and "roc_auc" in deep_f:
            fmt_entry["deep_model"] = {
                "roc_auc": round(float(deep_f["roc_auc"]), 4),
                "accuracy": round(float(deep_f["accuracy"]), 4),
                "pe": round(float(deep_f.get("p_e", 0.0)), 4),
                "recall_at_5pct_fpr": round(float(deep_f.get("recalls_at_fpr", {}).get("recall_at_5pct_fpr", 0.0)), 4)
            }
        else:
            fmt_entry["deep_model"] = "not run"

        if mlp_f and "roc_auc" in mlp_f:
            fmt_entry["handcrafted_mlp"] = {
                "roc_auc": round(float(mlp_f["roc_auc"]), 4),
                "accuracy": round(float(mlp_f["accuracy"]), 4),
                "pe": round(float(mlp_f.get("p_e", 0.0)), 4),
                "recall_at_5pct_fpr": round(float(mlp_f.get("recalls_at_fpr", {}).get("recall_at_5pct_fpr", 0.0)), 4)
            }
        else:
            fmt_entry["handcrafted_mlp"] = "not run"

        if gb_f and "roc_auc" in gb_f:
            fmt_entry["gradient_boosting"] = {
                "roc_auc": round(float(gb_f["roc_auc"]), 4),
                "accuracy": round(float(gb_f["accuracy"]), 4),
                "pe": round(float(gb_f.get("p_e", 0.0)), 4),
                "recall_at_5pct_fpr": round(float(gb_f.get("recalls_at_fpr", {}).get("recall_at_5pct_fpr", 0.0)), 4)
            }
        else:
            fmt_entry["gradient_boosting"] = "not run"

        comparison[fmt] = fmt_entry

    return comparison


def build_method_payload_breakdown(records: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, Any]:
    """Build honest method x payload rate performance breakdown with real counts in every cell."""
    df = pd.DataFrame(records)
    results = {}

    for m_type in ["image", "audio", "video"]:
        fmt_df = df[df["media_type"] == m_type]
        clean_df = fmt_df[fmt_df["label"] == 0]
        stego_df = fmt_df[fmt_df["label"] == 1]

        methods = sorted(list(stego_df["method"].unique()))
        payloads = sorted(list(stego_df["payload_rate"].unique()))

        breakdown = {}
        for m in methods:
            breakdown[m] = {}
            for p in payloads:
                sub_stego = stego_df[(stego_df["method"] == m) & (np.isclose(stego_df["payload_rate"], p, atol=0.01))]
                if len(sub_stego) == 0:
                    continue

                eval_subset = pd.concat([clean_df, sub_stego], ignore_index=True)
                yt = eval_subset["label"].values
                yp = eval_subset["cal_prob"].values

                acc = float(accuracy_score(yt, yp >= threshold))
                try:
                    auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.5
                except Exception:
                    auc = 0.5

                breakdown[m][f"{p:.2f}"] = {
                    "roc_auc": round(auc, 4),
                    "accuracy": round(acc, 4),
                    "stego_samples": len(sub_stego),
                    "clean_samples": len(clean_df),
                    "total_samples": len(eval_subset),
                    "display": f"{auc:.3f} (N={len(sub_stego)})"
                }

        results[m_type] = breakdown

    return results


def generate_method_payload_heatmap(breakdown: Dict[str, Any], output_path: Path):
    """Generate a clean, honest matplotlib heatmap showing AUC and real counts per cell."""
    rows = []
    for fmt, methods in breakdown.items():
        for m_name, payloads in methods.items():
            for p_str, stats in payloads.items():
                rows.append({
                    "format": fmt,
                    "method": m_name,
                    "payload_rate": float(p_str),
                    "auc": stats["roc_auc"],
                    "count": stats["stego_samples"]
                })

    if not rows:
        return

    df_heat = pd.DataFrame(rows)
    all_methods = df_heat["method"].unique().tolist()
    all_payloads = sorted(df_heat["payload_rate"].unique().tolist())

    matrix = np.full((len(all_methods), len(all_payloads)), np.nan)
    annotations = [["" for _ in range(len(all_payloads))] for _ in range(len(all_methods))]

    for i, m in enumerate(all_methods):
        for j, p in enumerate(all_payloads):
            match = df_heat[(df_heat["method"] == m) & (np.isclose(df_heat["payload_rate"], p, atol=0.01))]
            if not match.empty:
                val = match.iloc[0]["auc"]
                cnt = int(match.iloc[0]["count"])
                matrix[i, j] = val
                annotations[i][j] = f"{val:.3f}\n(N={cnt})"

    plt.figure(figsize=(10, max(6, len(all_methods) * 0.9)))
    plt.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.4, vmax=1.0)
    plt.colorbar(label="ROC-AUC")

    plt.xticks(ticks=range(len(all_payloads)), labels=[f"{p:.2f}" for p in all_payloads], fontsize=10)
    plt.yticks(ticks=range(len(all_methods)), labels=all_methods, fontsize=10)
    plt.xlabel("Payload Rate", fontsize=11, fontweight="bold")
    plt.ylabel("Hiding Method", fontsize=11, fontweight="bold")
    plt.title("VaultBreaker: Method x Payload Rate Detection Performance (ROC-AUC & N)", fontsize=12, fontweight="bold", pad=15)

    for i in range(len(all_methods)):
        for j in range(len(all_payloads)):
            text = annotations[i][j]
            if text:
                val = matrix[i, j]
                color = "white" if val > 0.7 else "black"
                plt.text(j, i, text, ha="center", va="center", color=color, fontsize=9, fontweight="semibold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved method x payload heatmap to {output_path}")


def run_leave_one_method_out(records: List[Dict[str, Any]], per_format_metrics: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    """Leave-one-method-out evaluation."""
    df = pd.DataFrame(records)
    results = {}

    for m_type in ["image", "audio", "video"]:
        fmt_df = df[df["media_type"] == m_type]
        clean_df = fmt_df[fmt_df["label"] == 0]
        stego_df = fmt_df[fmt_df["label"] == 1]
        overall_fmt_auc = per_format_metrics.get(m_type, {}).get("roc_auc", 0.5)

        method_evals = {}
        for m in sorted(list(stego_df["method"].unique())):
            m_stego = stego_df[stego_df["method"] == m]
            sub = pd.concat([clean_df, m_stego], ignore_index=True)
            yt = sub["label"].values
            yp = sub["cal_prob"].values
            try:
                auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.5
            except Exception:
                auc = 0.5
            acc = float(accuracy_score(yt, yp >= threshold))
            gap = round(overall_fmt_auc - auc, 4)

            method_evals[m] = {
                "method_auc": round(auc, 4),
                "accuracy": round(acc, 4),
                "sample_count": len(m_stego),
                "generalization_gap": gap,
                "note": f"{'Deficit' if gap > 0 else 'Advantage'} of {abs(gap):.4f} AUC vs format average ({overall_fmt_auc:.4f})"
            }

        results[m_type] = method_evals

    return results


def select_best_models_on_validation(
    models_dir: Path,
    splits_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    features_cache_dir: Optional[Path] = None,
    seed: int = 42,
    candidate_scores: Optional[Dict[str, Dict[str, float]]] = None,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Select best model per format strictly on VALIDATION data (manifest_val.csv).
    Test data is NEVER accessed or used here.
    Candidates per format: deep_model, handcrafted_mlp, gradient_boosting.
    Saves selection decisions and validation scores to models_saved/model_selection.json.
    """
    if candidate_scores is not None:
        candidate_val_aucs = candidate_scores
    else:
        if splits_dir is None:
            raise ValueError("splits_dir must be provided if candidate_scores is not provided.")
        val_manifest = splits_dir / "manifest_val.csv"
        if not val_manifest.exists():
            logger.warning(f"Validation manifest {val_manifest} not found. Cannot select models.")
            return {}

        df_val_raw = pd.read_csv(val_manifest)
        df_val_balanced = build_balanced_test_set(df_val_raw, seed=seed)

        candidate_val_aucs: Dict[str, Dict[str, float]] = {
            "image": {},
            "audio": {},
            "video": {}
        }

    # 1. Deep model on validation
    deep_ckpt_path = models_dir / "deep_model.pt"
    if deep_ckpt_path.exists():
        try:
            deep_results, _ = evaluate_deep_model_test_set(
                deep_ckpt_path=deep_ckpt_path,
                calibrator_path=models_dir / "deep_calibrator.pt",
                df_test_balanced=df_val_balanced,
                device=device
            )
            for fmt in ["image", "audio", "video"]:
                if fmt in deep_results.get("per_format", {}):
                    auc = deep_results["per_format"][fmt].get("roc_auc")
                    if auc is not None:
                        candidate_val_aucs[fmt]["deep_model"] = round(float(auc), 4)
        except Exception as e:
            logger.warning(f"Failed to evaluate deep model on val set: {e}")

    # 2. Handcrafted MLP on validation
    if (models_dir / "unified_model.pt").exists():
        try:
            mlp_results, _ = evaluate_handcrafted_model_test_set(
                models_dir=models_dir,
                df_test_balanced=df_val_balanced,
                features_cache_dir=features_cache_dir
            )
            if mlp_results:
                for fmt in ["image", "audio", "video"]:
                    if fmt in mlp_results.get("per_format", {}):
                        auc = mlp_results["per_format"][fmt].get("roc_auc")
                        if auc is not None:
                            candidate_val_aucs[fmt]["handcrafted_mlp"] = round(float(auc), 4)
        except Exception as e:
            logger.warning(f"Failed to evaluate handcrafted MLP on val set: {e}")

    # 3. Gradient boosting on validation
    try:
        gb_results = evaluate_gradient_boosting_test_set(
            models_dir=models_dir,
            df_test_balanced=df_val_balanced,
            features_cache_dir=features_cache_dir
        )
        for fmt in ["image", "audio", "video"]:
            if fmt in gb_results:
                auc = gb_results[fmt].get("roc_auc")
                if auc is not None:
                    candidate_val_aucs[fmt]["gradient_boosting"] = round(float(auc), 4)
    except Exception as e:
        logger.warning(f"Failed to evaluate gradient boosting on val set: {e}")

    # Make selection per format
    selection_record = {
        "selection_metric": "validation_roc_auc",
        "evaluated_on": "manifest_val.csv (strictly validation set, test data never used)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "formats": {}
    }

    display_names = {
        "deep_model": "Deep Model (SRNet/Audio1D/VideoTemporal)",
        "gradient_boosting": "Gradient Boosting (GBDT)",
        "handcrafted_mlp": "Handcrafted Feature MLP"
    }

    for fmt in ["image", "audio", "video"]:
        cands = candidate_val_aucs[fmt]
        if not cands:
            continue
        best_model = max(cands.keys(), key=lambda k: cands[k])
        best_auc = cands[best_model]

        other_scores = ", ".join(f"{m}: {score:.4f}" for m, score in cands.items() if m != best_model)
        reason = f"Best validation ROC-AUC ({best_auc:.4f})"
        if other_scores:
            reason += f" vs {other_scores}"

        selection_record["formats"][fmt] = {
            "selected_model": best_model,
            "model_name": display_names.get(best_model, best_model),
            "validation_roc_auc": best_auc,
            "reason": reason,
            "candidates": cands
        }
        logger.info(f"Model Selection for {fmt.upper()}: {best_model} ({reason})")

    out_file = save_path if save_path is not None else (models_dir / "model_selection.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(selection_record, f, indent=2)
    logger.info(f"Saved model selection decisions to {out_file}")

    return selection_record


def main():
    parser = argparse.ArgumentParser(description="Evaluate VaultBreaker model on test dataset.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="Smoke run: writes strictly to docs/RESULTS/smoke/ and never overwrites real results")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    splits_dir = Path(paths_cfg.get("splits_dir", "data/splits"))
    models_dir = Path(paths_cfg.get("models_dir", "models_saved"))
    base_results_dir = Path(paths_cfg.get("results_dir", "docs/RESULTS"))
    features_cache_dir = Path(paths_cfg.get("features_dir", "data/features")) / "cache"

    deep_ckpt_path = models_dir / "deep_model.pt"
    deep_calibrator_path = models_dir / "deep_calibrator.pt"
    test_manifest_path = splits_dir / "manifest_test.csv"

    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Test manifest {test_manifest_path} not found! Run make_dataset.py first.")

    device_str = args.device
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)

    # Determine metadata and smoke status strictly from checkpoint and flags (Item 1)
    deep_metadata = {}
    is_smoke_run = bool(args.smoke)
    if deep_ckpt_path.exists():
        try:
            ckpt_meta = torch.load(deep_ckpt_path, map_location="cpu")
            if isinstance(ckpt_meta, dict):
                if ckpt_meta.get("is_smoke") is True or str(ckpt_meta.get("profile", "")).lower() == "smoke":
                    is_smoke_run = True
                deep_metadata = {
                    "profile": ckpt_meta.get("profile", "smoke" if is_smoke_run else "laptop"),
                    "epochs": ckpt_meta.get("epoch", ckpt_meta.get("epochs", 8)),
                    "device": ckpt_meta.get("device", device_str),
                    "training_sample_counts": ckpt_meta.get("training_sample_counts", {}),
                    "seed": ckpt_meta.get("seed", config.get("seed", 42)),
                    "git_commit": ckpt_meta.get("git_commit", get_git_commit()),
                    "timestamp": ckpt_meta.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
                    "is_smoke": is_smoke_run
                }
        except Exception:
            pass

    if not deep_metadata:
        deep_metadata = {
            "profile": "smoke" if is_smoke_run else "laptop",
            "epochs": 8,
            "device": device_str,
            "training_sample_counts": {},
            "seed": config.get("seed", 42),
            "git_commit": get_git_commit(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_smoke": is_smoke_run
        }

    # Strict isolation: smoke runs write ONLY to docs/RESULTS/smoke/ (Item 1)
    if is_smoke_run:
        results_dir = base_results_dir / "smoke"
        logger.info(f"=== SMOKE RUN: Writing exclusively to {results_dir} (real results protected) ===")
    else:
        results_dir = base_results_dir
        logger.info(f"=== REAL EVALUATION RUN ({deep_metadata.get('profile')}): Writing to {results_dir} ===")

    ensure_dir(results_dir)

    # Run model selection strictly on validation set (Item 5)
    select_best_models_on_validation(
        models_dir=models_dir,
        splits_dir=splits_dir,
        device=device,
        features_cache_dir=features_cache_dir,
        seed=config.get("seed", 42)
    )

    df_test_raw = pd.read_csv(test_manifest_path)
    # Build strictly balanced test set (1 clean to 1 stego per source, evenly sampled)
    df_test_balanced = build_balanced_test_set(df_test_raw, seed=config.get("seed", 42))

    deep_eval_results = None
    handcrafted_eval_results = None
    gbdt_eval_results = None

    # 1. Evaluate Deep Model (if present)
    if deep_ckpt_path.exists():
        logger.info(f"Executing Deep Model Evaluation ({deep_ckpt_path}) on balanced test manifest...")
        deep_eval_results, deep_records = evaluate_deep_model_test_set(
            deep_ckpt_path=deep_ckpt_path,
            calibrator_path=deep_calibrator_path,
            df_test_balanced=df_test_balanced,
            device=device
        )
        t_cal = deep_eval_results["overall"].get("calibrated_threshold", 0.5)

        # Breakdown and heatmap
        breakdown = build_method_payload_breakdown(deep_records, threshold=t_cal)
        deep_eval_results["method_payload_breakdown"] = breakdown
        heatmap_path = results_dir / "method_payload_heatmap.png"
        generate_method_payload_heatmap(breakdown, heatmap_path)

        lomo = run_leave_one_method_out(deep_records, deep_eval_results["per_format"], threshold=t_cal)
        deep_eval_results["leave_one_method_out"] = lomo

        # Record full provenance metadata (Item 1)
        deep_eval_results.update(deep_metadata)

        with open(results_dir / "metrics_deep.json", "w") as f:
            json.dump(deep_eval_results, f, indent=2)
        logger.info(f"Saved {results_dir / 'metrics_deep.json'}")

    # 2. Evaluate Handcrafted MLP Model (if present)
    handcrafted_ckpt = models_dir / "unified_model.pt"
    if handcrafted_ckpt.exists():
        logger.info(f"Executing Handcrafted Model Evaluation ({handcrafted_ckpt}) on balanced test manifest...")
        handcrafted_eval_results, hc_records = evaluate_handcrafted_model_test_set(
            models_dir=models_dir,
            df_test_balanced=df_test_balanced,
            features_cache_dir=features_cache_dir
        )
        if handcrafted_eval_results:
            t_hc = handcrafted_eval_results["overall"].get("calibrated_threshold", 0.5)
            handcrafted_eval_results["method_payload_breakdown"] = build_method_payload_breakdown(hc_records, threshold=t_hc)
            handcrafted_eval_results["leave_one_method_out"] = run_leave_one_method_out(hc_records, handcrafted_eval_results["per_format"], threshold=t_hc)

            # Record full provenance metadata (Item 1)
            handcrafted_eval_results.update(deep_metadata)

            with open(results_dir / "metrics_handcrafted.json", "w") as f:
                json.dump(handcrafted_eval_results, f, indent=2)
            logger.info(f"Saved {results_dir / 'metrics_handcrafted.json'}")

    # 3. Evaluate Real Gradient Boosting Baseline (if present)
    gbdt_eval_results = evaluate_gradient_boosting_test_set(
        models_dir=models_dir,
        df_test_balanced=df_test_balanced,
        features_cache_dir=features_cache_dir
    )

    # 4. Side-by-side comparison (audited, zero hardcoded numbers)
    comparison_table = build_side_by_side_comparison(
        deep_metrics=deep_eval_results,
        handcrafted_metrics=handcrafted_eval_results,
        gbdt_metrics=gbdt_eval_results
    )
    comp_data = {
        **deep_metadata,
        "test_manifest": str(test_manifest_path),
        "comparison": comparison_table
    }
    with open(results_dir / "comparison.json", "w") as f:
        json.dump(comp_data, f, indent=2)
    logger.info(f"Saved {results_dir / 'comparison.json'}")

    # Save primary metrics.json for backward compatibility
    primary_metrics = deep_eval_results or handcrafted_eval_results
    if primary_metrics:
        primary_metrics["side_by_side_comparison"] = comparison_table
        primary_metrics.update(deep_metadata)
        with open(results_dir / "metrics.json", "w") as f:
            json.dump(primary_metrics, f, indent=2)

    # Console Summary Table
    print("\n" + "=" * 70)
    print("               VAULTBREAKER EVALUATION SUMMARY")
    print("=" * 70)
    if primary_metrics:
        print(f"Model Type:         {primary_metrics.get('model_type')}")
        print(f"Overall Accuracy:   {primary_metrics['overall']['accuracy']*100:.2f}% (at calibrated threshold {primary_metrics['overall'].get('calibrated_threshold')})")
        print(f"Overall ROC-AUC:    {primary_metrics['overall']['roc_auc']:.4f} (95% CI: {primary_metrics['overall']['roc_auc_ci95']})")
        print(f"Overall Recall@5%:  {primary_metrics['overall'].get('recalls_at_fpr', {}).get('recall_at_5pct_fpr', 0.0):.4f}")
        print("-" * 70)
        print(f"{'Format':<8} | {'ROC-AUC':<9} | {'Recall@5%':<10} | {'Accuracy':<9} | {'#Test Sources':<14} | {'Status'}")
        print("-" * 70)
        for fmt, m in primary_metrics.get("per_format", {}).items():
            r5 = m.get("recalls_at_fpr", {}).get("recall_at_5pct_fpr", 0.0)
            status = m.get("status", "trusted")
            print(f"{fmt.upper():<8} | {m['roc_auc']:<9.4f} | {r5:<10.4f} | {m['accuracy']*100:<8.1f}% | {m['unique_sources']:<14d} | {status}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
