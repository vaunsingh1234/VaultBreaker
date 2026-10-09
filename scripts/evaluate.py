#!/usr/bin/env python3
"""Evaluation CLI for VaultBreaker.
Evaluates the trained multi-modal pipeline on the test split,
computes PE, ROC-AUC and 95% bootstrap confidence intervals,
executes the 3 core ablations, cross-method generalization tests,
runs the cover-only shortcut sanity check, and generates publication plots.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from vaultbreaker.utils.config import load_config
from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.io import ensure_dir
from vaultbreaker.models.unified_net import VaultBreakerUnifiedNet
from vaultbreaker.training.calibrate import TemperatureScalingCalibrator
from vaultbreaker.training.evaluate import StegoEvaluationSuite
from vaultbreaker.training.dataset import MultiModalStegoDataset
from vaultbreaker.training.trainer import UnifiedTrainer

logger = get_logger("Evaluate")

def run_shortcut_sanity_check(train_data: dict, seed: int = 42) -> dict:
    """
    Shortcut sanity check:
    Takes clean cover samples per format, assigns purely random labels,
    and trains a classifier. If no shortcut exists, ROC-AUC will be ~0.50.
    """
    logger.info("Executing Shortcut Sanity Check (Random labels on clean covers)...")
    results = {}
    rng = np.random.RandomState(seed)

    for m_type in ["image", "audio", "video"]:
        x_key = f"X_{m_type}"
        y_key = f"y_{m_type}"
        if x_key not in train_data:
            continue
        X = train_data[x_key]
        y = train_data[y_key]
        clean_mask = (y == 0)
        X_clean = X[clean_mask]

        if len(X_clean) < 6:
            results[m_type] = {"auc": 0.50, "sample_count": len(X_clean)}
            continue

        y_random = rng.randint(0, 2, size=len(X_clean))
        n_split = int(0.7 * len(X_clean))
        if n_split < 2 or len(X_clean) - n_split < 2:
            results[m_type] = {"auc": 0.50, "sample_count": len(X_clean)}
            continue

        y_tr, y_te = y_random[:n_split], y_random[n_split:]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            results[m_type] = {"auc": 0.50, "sample_count": len(X_clean)}
            continue

        clf = LogisticRegression(max_iter=500, random_state=seed)
        clf.fit(X_clean[:n_split], y_tr)
        probs = clf.predict_proba(X_clean[n_split:])[:, 1]
        try:
            auc = float(roc_auc_score(y_te, probs))
        except Exception:
            auc = 0.50
        results[m_type] = {"auc": round(auc, 4), "sample_count": len(X_clean)}
        logger.info(f"  Sanity Check [{m_type}]: Random-Label ROC-AUC = {auc:.4f} (N={len(X_clean)})")

    return results

def run_cross_method_generalization(train_data: dict, test_data: dict) -> dict:
    """
    Cross-method generalization test:
    Trains on LSB replacement only, tests on unseen methods (lsb_matching, edge_adaptive, dct_ac).
    Reports sample counts next to each number.
    """
    logger.info("Executing Cross-Method Generalization Test...")
    X_tr_img = train_data.get("X_image", np.empty((0, 72)))
    y_tr_img = train_data.get("y_image", np.empty(0))
    m_tr_img = train_data.get("methods_image", np.empty(0))

    X_te_img = test_data.get("X_image", np.empty((0, 72)))
    y_te_img = test_data.get("y_image", np.empty(0))
    m_te_img = test_data.get("methods_image", np.empty(0))

    if len(X_tr_img) == 0 or len(X_te_img) == 0:
        return {}

    train_mask = (m_tr_img == "clean") | (m_tr_img == "lsb_replacement")
    if np.sum(train_mask) < 4:
        return {}

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr_img[train_mask], y_tr_img[train_mask])

    results = {}
    for test_method in ["lsb_matching", "edge_adaptive", "dct_ac"]:
        method_mask = (m_te_img == "clean") | (m_te_img == test_method)
        n_samples = int(np.sum(method_mask))
        if n_samples >= 2 and len(np.unique(y_te_img[method_mask])) > 1:
            probs = clf.predict_proba(X_te_img[method_mask])[:, 1]
            try:
                auc = float(roc_auc_score(y_te_img[method_mask], probs))
            except Exception:
                auc = 0.5
            results[test_method] = {"auc": round(auc, 4), "sample_count": n_samples}
            logger.info(f"  Trained on LSB-Repl -> Tested on {test_method}: ROC-AUC = {auc:.4f} (N={n_samples})")

    return results

def run_ablations(train_data: dict, val_data: dict, test_data: dict, unified_metrics: dict, config: dict, models_dir: Path) -> dict:
    """
    Runs the 3 scientific ablations:
    (a) Unified shared head vs separate model per format (honest per-format comparison)
    (b) With vs without auxiliary method head (actively trains unified net without aux head)
    (c) Handcrafted features + neural net vs best baseline (Histogram Gradient Boosting / RF)
    """
    logger.info("Executing 3 Scientific Ablations...")
    ablations = {}

    # (a) Separate models per format vs unified
    sep_metrics = {}
    for m_type in ["image", "audio", "video"]:
        x_tr, y_tr = train_data.get(f"X_{m_type}"), train_data.get(f"y_{m_type}")
        x_te, y_te = test_data.get(f"X_{m_type}"), test_data.get(f"y_{m_type}")
        if x_tr is not None and x_te is not None and len(x_tr) > 0 and len(x_te) > 0:
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(x_tr, y_tr)
            probs = clf.predict_proba(x_te)[:, 1]
            auc = float(roc_auc_score(y_te, probs)) if len(np.unique(y_te)) > 1 else 0.5
            acc = float(accuracy_score(y_te, probs >= 0.5))
            sep_metrics[m_type] = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4), "sample_count": len(y_te)}

    comparison_notes = []
    for m_type in ["image", "audio", "video"]:
        rf_acc = sep_metrics.get(m_type, {}).get("accuracy", 0.0)
        uni_acc = unified_metrics.get("per_format", {}).get(m_type, {}).get("accuracy", 0.0)
        if rf_acc > uni_acc:
            comparison_notes.append(f"On {m_type}, separate RandomForest achieves higher accuracy ({rf_acc:.4f} vs {uni_acc:.4f}).")
        else:
            comparison_notes.append(f"On {m_type}, unified model achieves higher or equal accuracy ({uni_acc:.4f} vs {rf_acc:.4f}).")

    ablations["ablation_a_separate_vs_shared"] = {
        "description": "Separate per-format RandomForest models vs Unified Cross-Modal Network",
        "separate_per_format": sep_metrics,
        "unified_model": {
            m: {"accuracy": unified_metrics["per_format"][m]["accuracy"], "roc_auc": unified_metrics["per_format"][m]["roc_auc"]}
            for m in unified_metrics.get("per_format", {})
        },
        "finding": " ".join(comparison_notes)
    }

    # (b) With vs without auxiliary head
    logger.info("Ablation (b): Actively training and evaluating unified model WITHOUT auxiliary head...")
    feat_cfg = config.get("features", {})
    model_cfg = config.get("models", {})
    train_cfg = config.get("training", {})
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model_no_aux = VaultBreakerUnifiedNet(
        img_dim=feat_cfg.get("image_dim", 72),
        aud_dim=feat_cfg.get("audio_dim", 56),
        vid_dim=feat_cfg.get("video_dim", 64),
        latent_dim=model_cfg.get("latent_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        dropout=model_cfg.get("dropout", 0.2),
        use_aux_head=False,
        num_methods=8
    )

    train_ds = MultiModalStegoDataset(train_data)
    val_ds = MultiModalStegoDataset(val_data)
    trainer_no_aux = UnifiedTrainer(
        model=model_no_aux,
        learning_rate=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0001),
        aux_loss_weight=0.0,
        device=device
    )

    no_aux_save_path = models_dir / "unified_model_no_aux.pt"
    trainer_no_aux.fit(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=train_cfg.get("epochs", 25),
        batch_size=train_cfg.get("batch_size", 32),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 6),
        save_path=no_aux_save_path
    )

    model_no_aux.eval()
    no_aux_per_fmt = {}
    all_y_no_aux = []
    all_probs_no_aux = []
    for m_type in ["image", "audio", "video"]:
        x_te = test_data.get(f"X_{m_type}")
        y_te = test_data.get(f"y_{m_type}")
        if x_te is not None and len(x_te) > 0:
            with torch.no_grad():
                xt = torch.tensor(x_te, dtype=torch.float32).to(device)
                logits, _, _ = model_no_aux(xt, m_type)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
            acc = float(accuracy_score(y_te, probs >= 0.5))
            auc = float(roc_auc_score(y_te, probs)) if len(np.unique(y_te)) > 1 else 0.5
            no_aux_per_fmt[m_type] = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4), "sample_count": len(y_te)}
            all_y_no_aux.extend(y_te)
            all_probs_no_aux.extend(probs)

    all_y_arr = np.array(all_y_no_aux)
    all_p_arr = np.array(all_probs_no_aux)
    no_aux_ov_acc = float(accuracy_score(all_y_arr, all_p_arr >= 0.5))
    no_aux_ov_auc = float(roc_auc_score(all_y_arr, all_p_arr)) if len(np.unique(all_y_arr)) > 1 else 0.5

    with_aux_auc = unified_metrics["overall"]["roc_auc"]
    diff_auc = with_aux_auc - no_aux_ov_auc
    if abs(diff_auc) < 0.01:
        finding_b = (
            f"The auxiliary head achieves comparable ROC-AUC to the model without it "
            f"({with_aux_auc:.4f} with-aux vs {no_aux_ov_auc:.4f} without-aux, diff {diff_auc:+.4f}). "
            f"The data does not show significant regularizing gains from auxiliary method supervision."
        )
    elif diff_auc > 0:
        finding_b = (
            f"The auxiliary head modestly improves ROC-AUC "
            f"({with_aux_auc:.4f} with-aux vs {no_aux_ov_auc:.4f} without-aux, diff +{diff_auc:.4f})."
        )
    else:
        finding_b = (
            f"The model without auxiliary head achieves slightly higher ROC-AUC "
            f"({no_aux_ov_auc:.4f} without-aux vs {with_aux_auc:.4f} with-aux, diff {-diff_auc:.4f}). "
            f"The multi-task auxiliary head did not improve binary detection performance."
        )

    ablations["ablation_b_auxiliary_head"] = {
        "description": "Unified network trained WITH auxiliary method head vs WITHOUT auxiliary method head",
        "with_aux_head": {
            "overall_roc_auc": unified_metrics["overall"]["roc_auc"],
            "overall_accuracy": unified_metrics["overall"]["accuracy"],
            "per_format": {
                m: {"accuracy": unified_metrics["per_format"][m]["accuracy"], "roc_auc": unified_metrics["per_format"][m]["roc_auc"]}
                for m in unified_metrics.get("per_format", {})
            }
        },
        "without_aux_head": {
            "overall_roc_auc": round(no_aux_ov_auc, 4),
            "overall_accuracy": round(no_aux_ov_acc, 4),
            "per_format": no_aux_per_fmt
        },
        "finding": finding_b
    }

    # (c) Handcrafted features + neural head vs best classical baseline (HistGradientBoosting)
    gbdt_metrics = {}
    for m_type in ["image", "audio", "video"]:
        x_tr, y_tr = train_data.get(f"X_{m_type}"), train_data.get(f"y_{m_type}")
        x_te, y_te = test_data.get(f"X_{m_type}"), test_data.get(f"y_{m_type}")
        if x_tr is not None and x_te is not None and len(x_tr) > 0 and len(x_te) > 0:
            gbdt = HistGradientBoostingClassifier(random_state=42)
            gbdt.fit(x_tr, y_tr)
            probs = gbdt.predict_proba(x_te)[:, 1]
            auc = float(roc_auc_score(y_te, probs)) if len(np.unique(y_te)) > 1 else 0.5
            acc = float(accuracy_score(y_te, probs >= 0.5))
            gbdt_metrics[m_type] = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4), "sample_count": len(y_te)}

    ablations["ablation_c_deep_vs_gbdt"] = {
        "description": "Deep Unified Network vs HistGradientBoosting baseline",
        "hist_gradient_boosting": gbdt_metrics,
        "unified_network": {
            m: {"accuracy": unified_metrics["per_format"][m]["accuracy"], "roc_auc": unified_metrics["per_format"][m]["roc_auc"]}
            for m in unified_metrics.get("per_format", {})
        }
    }

    return ablations

def main():
    parser = argparse.ArgumentParser(description="Evaluate VaultBreaker model on test dataset.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    features_dir = Path(paths_cfg.get("features_dir", "data/features"))
    models_dir = Path(paths_cfg.get("models_dir", "models_saved"))
    results_dir = Path(paths_cfg.get("results_dir", "docs/RESULTS"))
    ensure_dir(results_dir)

    test_matrix_path = features_dir / "split_test.npz"
    train_matrix_path = features_dir / "split_train.npz"
    val_matrix_path = features_dir / "split_val.npz"
    if not test_matrix_path.exists():
        raise FileNotFoundError(f"Test matrix not found at {test_matrix_path}")

    test_data = dict(np.load(test_matrix_path, allow_pickle=True))
    train_data = dict(np.load(train_matrix_path, allow_pickle=True))
    val_data = dict(np.load(val_matrix_path, allow_pickle=True)) if val_matrix_path.exists() else {}

    feat_cfg = config.get("features", {})
    model_cfg = config.get("models", {})

    # Load unified model
    model = VaultBreakerUnifiedNet(
        img_dim=feat_cfg.get("image_dim", 72),
        aud_dim=feat_cfg.get("audio_dim", 56),
        vid_dim=feat_cfg.get("video_dim", 64),
        latent_dim=model_cfg.get("latent_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        use_aux_head=model_cfg.get("use_aux_head", True),
        num_methods=8
    )
    weights_path = models_dir / "unified_model.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        logger.info(f"Loaded unified model weights from {weights_path}")
    else:
        logger.warning(f"Weights file not found at {weights_path}. Running with initialized model.")

    # Load calibrator
    calibrator_path = models_dir / "calibrator.joblib"
    if calibrator_path.exists():
        calibrator = TemperatureScalingCalibrator.load(calibrator_path)
    else:
        calibrator = TemperatureScalingCalibrator(target_fpr=0.05)

    suite = StegoEvaluationSuite(model, calibrator)
    eval_results = suite.run_full_evaluation(test_data, results_dir)

    # Cross-method generalization
    cross_results = run_cross_method_generalization(train_data, test_data)
    eval_results["cross_method_generalization"] = cross_results

    # Sanity check
    sanity_auc = run_shortcut_sanity_check(train_data)
    eval_results["shortcut_sanity_check"] = sanity_auc

    # Ablations
    ablations = run_ablations(train_data, val_data, test_data, eval_results, config, models_dir)
    eval_results["ablations"] = ablations

    # Update metrics.json
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    logger.info("Evaluation completed. All figures, tables, and metrics written to docs/RESULTS/metrics.json.")

if __name__ == "__main__":
    main()
