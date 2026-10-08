#!/usr/bin/env python3
"""Unified Model Training and Calibration CLI for VaultBreaker.
Trains per-format classical baselines and the unified multi-modal neural network,
fits temperature scaling calibration, and sets target-FPR thresholds.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import torch

from vaultbreaker.utils.config import load_config
from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.seed import seed_everything
from vaultbreaker.utils.io import ensure_dir
from vaultbreaker.models.baselines import BaselineModelsManager
from vaultbreaker.models.unified_net import VaultBreakerUnifiedNet
from vaultbreaker.training.dataset import MultiModalStegoDataset
from vaultbreaker.training.trainer import UnifiedTrainer
from vaultbreaker.training.calibrate import TemperatureScalingCalibrator

logger = get_logger("Train")

def main():
    parser = argparse.ArgumentParser(description="Train VaultBreaker baselines and unified model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--fast", action="store_true", help="Fast training mode (fewer epochs)")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    seed_everything(seed)

    paths_cfg = config.get("paths", {})
    features_dir = Path(paths_cfg.get("features_dir", "data/features"))
    models_dir = Path(paths_cfg.get("models_dir", "models_saved"))
    ensure_dir(models_dir)

    train_matrix_path = features_dir / "split_train.npz"
    val_matrix_path = features_dir / "split_val.npz"

    if not train_matrix_path.exists() or not val_matrix_path.exists():
        raise FileNotFoundError(
            f"Feature matrices not found in {features_dir}. Please run scripts/extract_features.py first."
        )

    train_data = dict(np.load(train_matrix_path, allow_pickle=True))
    val_data = dict(np.load(val_matrix_path, allow_pickle=True))

    # --- 1. TRAIN CLASSICAL BASELINES ---
    logger.info("=== STEP 1: TRAINING PER-FORMAT BASELINE MODELS ===")
    baseline_manager = BaselineModelsManager(random_state=seed)
    baseline_metrics = {}

    for m_type in ["image", "audio", "video"]:
        x_key = f"X_{m_type}"
        y_key = f"y_{m_type}"
        if x_key in train_data and x_key in val_data:
            X_tr, y_tr = train_data[x_key], train_data[y_key]
            X_vl, y_vl = val_data[x_key], val_data[y_key]
            if len(X_tr) > 0 and len(X_vl) > 0:
                logger.info(f"Training baselines on {m_type} (Train: {len(X_tr)}, Val: {len(X_vl)})...")
                m_res = baseline_manager.train_and_eval(
                    X_tr, y_tr, X_vl, y_vl, media_type=m_type, save_dir=models_dir
                )
                baseline_metrics[m_type] = m_res

    # --- 2. TRAIN UNIFIED PYTORCH MODEL ---
    logger.info("=== STEP 2: TRAINING MAIN UNIFIED PYTORCH MODEL ===")
    train_ds = MultiModalStegoDataset(train_data)
    val_ds = MultiModalStegoDataset(val_data)

    model_cfg = config.get("models", {})
    feat_cfg = config.get("features", {})
    train_cfg = config.get("training", {})

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    unified_net = VaultBreakerUnifiedNet(
        img_dim=feat_cfg.get("image_dim", 72),
        aud_dim=feat_cfg.get("audio_dim", 56),
        vid_dim=feat_cfg.get("video_dim", 64),
        latent_dim=model_cfg.get("latent_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        dropout=model_cfg.get("dropout", 0.2),
        use_aux_head=model_cfg.get("use_aux_head", True),
        num_methods=8
    )

    epochs = train_cfg.get("fast_epochs", 5) if args.fast else train_cfg.get("epochs", 30)
    batch_size = train_cfg.get("batch_size", 32)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    patience = train_cfg.get("early_stopping_patience", 6)

    trainer = UnifiedTrainer(
        model=unified_net,
        learning_rate=lr,
        weight_decay=weight_decay,
        aux_loss_weight=model_cfg.get("aux_loss_weight", 0.3),
        device=device
    )

    model_save_path = models_dir / "unified_model.pt"
    train_res = trainer.fit(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=patience,
        save_path=model_save_path
    )

    logger.info(f"Unified Model Best Val Metrics: {train_res['best_metrics']}")

    # --- 3. FIT TEMPERATURE SCALING CALIBRATOR ---
    logger.info("=== STEP 3: FITTING PROBABILITY CALIBRATION ===")
    target_fpr = model_cfg.get("target_fpr", 0.05)
    calibrator = TemperatureScalingCalibrator(target_fpr=target_fpr)
    calibrator.fit(train_res["val_logits"], train_res["val_labels"])
    calibrator.save(models_dir / "calibrator.joblib")

    # Save summary
    summary = {
        "baselines": baseline_metrics,
        "unified_model": {
            "metrics": train_res["best_metrics"],
            "calibrated_threshold": calibrator.decision_threshold,
            "temperature": calibrator.temperature
        }
    }
    with open(models_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
