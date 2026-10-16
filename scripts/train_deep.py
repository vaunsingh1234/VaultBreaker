"""Deep Steganography Model Training Pipeline.

Trains VaultBreakerDeepNet across modalities with:
- Auto device selection (CUDA / MPS / CPU)
- Mixed precision on GPU
- AdamW optimizer and CosineAnnealingLR schedule
- Payload curriculum (high payloads first, then full spectrum)
- On-the-fly paired cover/stego mini-batch sampling
- Source-disjoint validation evaluation and early stopping on ROC-AUC
- Temperature scaling calibration on validation split
- Size guardrails (< 90 MB checkpoint)
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score

from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.config import load_config, get_default_config_path
from vaultbreaker.models.deep_net import VaultBreakerDeepNet
from vaultbreaker.training.on_the_fly_dataset import OnTheFlyPairedStegoDataset, collate_paired_batch
from vaultbreaker.training.calibrate import TemperatureScalingCalibrator

logger = get_logger("DeepTrainer")


def get_auto_device() -> torch.device:
    """Select best available acceleration device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def load_val_samples(manifest_path: Path, max_samples_per_fmt: int = 500) -> List[Dict[str, Any]]:
    """Load pre-generated validation samples for honest evaluation."""
    if not manifest_path.exists():
        logger.warning(f"Validation manifest {manifest_path} not found.")
        return []

    df = pd.read_csv(manifest_path)
    val_samples = []

    for _, row in df.iterrows():
        fp = Path(row["filepath"])
        if not fp.exists():
            continue
        val_samples.append({
            "filepath": str(fp),
            "media_type": row["media_type"],
            "label": float(row["label"]),
            "source_id": row["source_id"]
        })

    # Balance and cap
    by_fmt: Dict[str, List[Dict]] = {}
    for s in val_samples:
        by_fmt.setdefault(s["media_type"], []).append(s)

    selected = []
    for fmt, items in by_fmt.items():
        selected.extend(items[:max_samples_per_fmt])
    return selected


def evaluate_val_set(
    model: VaultBreakerDeepNet,
    val_samples: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 32
) -> Tuple[float, float, Dict[str, float]]:
    """Compute validation ROC-AUC and Accuracy across active modalities using mini-batching."""
    model.eval()
    all_labels = []
    all_probs = []
    by_format_preds: Dict[str, List[Tuple[float, float]]] = {}

    import cv2
    import soundfile as sf
    import imageio.v3 as iio

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for item in val_samples:
        by_type.setdefault(item["media_type"], []).append(item)

    with torch.no_grad():
        for m_type, items in by_type.items():
            bs = min(batch_size, 8 if m_type == "video" else batch_size)
            for i in range(0, len(items), bs):
                chunk = items[i:i + bs]
                tensors = []
                chunk_labels = []
                for it in chunk:
                    fp = it["filepath"]
                    lbl = it["label"]
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
                            t = torch.from_numpy(data.astype(np.float32) / 32768.0).unsqueeze(0)
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
                        chunk_labels.append(lbl)
                    except Exception:
                        continue

                if not tensors:
                    continue

                batch_t = torch.stack(tensors).to(device)
                logits, _, _ = model(batch_t, m_type)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
                if not isinstance(probs, list):
                    probs = [probs]

                for lbl, prob in zip(chunk_labels, probs):
                    all_labels.append(lbl)
                    all_probs.append(prob)
                    by_format_preds.setdefault(m_type, []).append((lbl, prob))

    if len(all_labels) == 0:
        return 0.5, 0.5, {}

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    overall_acc = float(accuracy_score(y_true, y_pred))
    try:
        overall_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        overall_auc = 0.5

    fmt_metrics = {}
    for fmt, pairs in by_format_preds.items():
        f_true = np.array([p[0] for p in pairs])
        f_prob = np.array([p[1] for p in pairs])
        try:
            f_auc = float(roc_auc_score(f_true, f_prob)) if len(np.unique(f_true)) > 1 else 0.5
        except Exception:
            f_auc = 0.5
        fmt_metrics[f"{fmt}_auc"] = f_auc

    return overall_auc, overall_acc, fmt_metrics


def train_deep_pipeline(
    profile_name: str = "laptop",
    config_path: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    resume_path: Optional[str] = None,
    seed: int = 42,
    device_override: Optional[str] = None
) -> Dict[str, Any]:
    """Execute deep model training pipeline."""
    # Seed all pseudo-random generators for reproducibility (T3.2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = load_config(config_path or get_default_config_path())
    profiles = cfg.get("profiles", {})
    profile = profiles.get(profile_name, profiles.get("laptop", {}))

    epochs = epochs_override or profile.get("epochs", 8)
    batch_size = batch_size_override or profile.get("batch_size", 16)
    lr = float(profile.get("learning_rate", 0.0005))
    weight_decay = float(profile.get("weight_decay", 0.0001))
    target_covers_img = profile.get("image_covers", 2000)
    target_covers_aud = profile.get("audio_windows", 1000)
    target_covers_vid = profile.get("video_clips", 350)

    if device_override and device_override != "auto":
        device = torch.device(device_override)
    else:
        device = get_auto_device()

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 1. Gather Training Clean Covers
    splits_dir = Path(cfg.get("paths", {}).get("splits_dir", "data/splits"))
    train_manifest = splits_dir / "manifest_train.csv"

    cover_records = []
    if train_manifest.exists():
        df_train = pd.read_csv(train_manifest)
        clean_df = df_train[df_train["label"] == 0]
        for _, row in clean_df.iterrows():
            fp = Path(row["filepath"])
            if fp.exists():
                cover_records.append({
                    "media_type": row["media_type"],
                    "filepath": str(fp),
                    "source_id": row["source_id"]
                })

    # Group and cap according to profile targets
    img_covers = [r for r in cover_records if r["media_type"] == "image"][:target_covers_img]
    aud_covers = [r for r in cover_records if r["media_type"] == "audio"][:target_covers_aud]
    vid_covers = [r for r in cover_records if r["media_type"] == "video"][:target_covers_vid]

    all_covers = img_covers + aud_covers + vid_covers
    random.shuffle(all_covers)

    print("=" * 65)
    print("      VAULTBREAKER DEEP TRAINING RUN SPECIFICATION")
    print("=" * 65)
    print(f"Profile:            {profile_name.upper()}")
    print(f"Device:             {device} (AMP={use_amp})")
    print(f"Seed:               {seed}")
    print(f"Epochs:             {epochs}")
    print(f"Batch Size (pairs): {batch_size} ({batch_size * 2} samples/step)")
    print(f"Clean Covers Found: Images={len(img_covers)}, Audio={len(aud_covers)}, Video={len(vid_covers)}")
    print(f"Total Covers:       {len(all_covers)}")
    print("=" * 65)

    if len(all_covers) == 0:
        raise RuntimeError("No clean cover files found to train on! Run make data first.")

    # 2. Validation Samples
    val_manifest = splits_dir / "manifest_val.csv"
    val_samples = load_val_samples(val_manifest, max_samples_per_fmt=30 if profile_name == "smoke" else 500)
    logger.info(f"Loaded {len(val_samples)} validation samples for evaluation.")

    # 3. Model Architecture
    base_ch = 16 if profile_name in ("smoke", "laptop") else 32
    model = VaultBreakerDeepNet(
        base_channels=base_ch,
        latent_dim=128,
        use_fusion=False,  # Deep visual/audio features directly into shared latent
        dropout=0.1
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    best_auc = -1.0
    start_epoch = 1

    # Resume model + optimizer + scheduler + scaler states (T3.3, T3.4)
    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming model and optimizer from checkpoint: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state.get("model_state", state))
        if "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
        if "scheduler_state" in state:
            scheduler.load_state_dict(state["scheduler_state"])
        if use_amp and scaler is not None and state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])
        if "epoch" in state:
            start_epoch = int(state["epoch"]) + 1
        if "val_auc" in state:
            best_auc = float(state["val_auc"])
        logger.info(f"Resumed from epoch {start_epoch - 1} with best Val AUC: {best_auc:.4f}")

    train_ds = OnTheFlyPairedStegoDataset(
        all_covers,
        curriculum_epoch=0,
        high_payload_epochs=cfg.get("curriculum", {}).get("high_payload_epochs", 2),
        augment=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_paired_batch,
        num_workers=0  # Stable across macOS and Linux without multiprocessing fork issues
    )

    models_dir = Path(cfg.get("paths", {}).get("models_dir", "models_saved"))
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = models_dir / "deep_model.pt"
    log_csv_path = models_dir / "deep_training_log.csv"

    patience = 6
    patience_counter = 0
    history = []

    start_time = time.time()
    total_steps = len(train_loader)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        train_ds.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        for step_idx, batch_items in enumerate(train_loader):
            optimizer.zero_grad()
            by_fmt: Dict[str, List[Dict]] = {}
            for item in batch_items:
                by_fmt.setdefault(item["media_type"], []).append(item)

            step_loss = 0.0

            for m_type, items in by_fmt.items():
                x = torch.stack([it["x"] for it in items]).to(device)
                y = torch.stack([it["y"] for it in items]).to(device).unsqueeze(1)
                m_idx = torch.stack([it["method_idx"] for it in items]).to(device)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits, aux_logits, _ = model(x, m_type)
                        loss_bin = bce_loss(logits, y)
                        loss_aux = ce_loss(aux_logits, m_idx) if aux_logits is not None else 0.0
                        loss = loss_bin + 0.3 * loss_aux
                    scaler.scale(loss).backward()
                else:
                    logits, aux_logits, _ = model(x, m_type)
                    loss_bin = bce_loss(logits, y)
                    loss_aux = ce_loss(aux_logits, m_idx) if aux_logits is not None else 0.0
                    loss = loss_bin + 0.3 * loss_aux
                    loss.backward()

                step_loss += loss.item() * (len(items) / len(batch_items))

            if use_amp:
                # Gradient clipping under AMP (T3.5)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            epoch_loss += step_loss
            n_steps += 1

            if (step_idx + 1) % 5 == 0 or (step_idx + 1) == total_steps:
                logger.info(f"Epoch {epoch:02d} [{step_idx + 1:02d}/{total_steps:02d}] Step Loss: {step_loss:.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_steps)
        epoch_duration = time.time() - epoch_start

        # Evaluate on validation
        val_auc, val_acc, fmt_aucs = evaluate_val_set(model, val_samples, device)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.1f}s] - "
            f"Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc*100:.1f}% | "
            f"{fmt_aucs}"
        )

        row_log = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_auc": val_auc,
            "val_acc": val_acc,
            "epoch_duration_sec": epoch_duration
        }
        row_log.update(fmt_aucs)
        history.append(row_log)

        # Early stopping and checkpointing (T3.3, T3.4)
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if use_amp and scaler is not None else None,
                "epoch": epoch,
                "val_auc": val_auc,
                "val_acc": val_acc,
                "profile": profile_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, best_ckpt_path)
            logger.info(f" Saved new best deep model checkpoint to {best_ckpt_path} (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without AUC improvement.")
                break

    total_time = time.time() - start_time
    pd.DataFrame(history).to_csv(log_csv_path, index=False)

    # 4. Fit Validation Calibrator on best model
    if best_ckpt_path.exists():
        best_state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_state.get("model_state", best_state))
        model.eval()

        val_logits = []
        val_y = []
        by_type_calib: Dict[str, List[Dict[str, Any]]] = {}
        for item in val_samples:
            by_type_calib.setdefault(item["media_type"], []).append(item)

        with torch.no_grad():
            for m_type, items in by_type_calib.items():
                bs = min(32, 8 if m_type == "video" else 32)
                for i in range(0, len(items), bs):
                    chunk = items[i:i + bs]
                    tensors = []
                    chunk_labels = []
                    for it in chunk:
                        fp = it["filepath"]
                        lbl = it["label"]
                        try:
                            if m_type == "image":
                                import cv2
                                img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                                if img is None:
                                    continue
                                if img.shape != (256, 256):
                                    img = cv2.resize(img, (256, 256))
                                t = torch.from_numpy(img).unsqueeze(0).float()
                            elif m_type == "audio":
                                import soundfile as sf
                                d, _ = sf.read(fp, dtype="int16")
                                if len(d) < 48000:
                                    d = np.pad(d, (0, 48000 - len(d)), mode="wrap")
                                t = torch.from_numpy(d[:48000].astype(np.float32) / 32768.0).unsqueeze(0)
                            else:
                                import imageio.v3 as iio
                                import cv2
                                f = iio.imread(fp)
                                if f.ndim == 4:
                                    f = np.dot(f[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                                if len(f) < 8:
                                    pad = np.zeros((8 - len(f), f.shape[1], f.shape[2]), dtype=np.uint8)
                                    f = np.concatenate([f, pad], axis=0)
                                f = f[:8]
                                if f.shape[1:3] != (128, 128):
                                    f = np.stack([cv2.resize(fr, (128, 128)) for fr in f])
                                t = torch.from_numpy(f).unsqueeze(1).float()
                            tensors.append(t)
                            chunk_labels.append(lbl)
                        except Exception:
                            continue

                    if not tensors:
                        continue

                    batch_t = torch.stack(tensors).to(device)
                    lg, _, _ = model(batch_t, m_type)
                    val_logits.extend(lg.squeeze(-1).cpu().numpy().tolist())
                    val_y.extend(chunk_labels)

        if len(val_logits) > 0 and len(np.unique(val_y)) > 1:
            calibrator = TemperatureScalingCalibrator(target_fpr=0.05)
            calibrator.fit(np.array(val_logits), np.array(val_y))
            calibrator.save(models_dir / "deep_calibrator.pt")
            logger.info("Fitted and saved validation temperature calibrator.")

    # Checkpoint size audit and warning (T3.7)
    ckpt_size_mb = best_ckpt_path.stat().st_size / (1024 * 1024) if best_ckpt_path.exists() else 0.0
    if ckpt_size_mb > 90.0:
        logger.warning(f"CRITICAL WARNING: Checkpoint size {ckpt_size_mb:.2f} MB exceeds 90 MB limit!")
    else:
        logger.info(f"Checkpoint size constraint verified: {ckpt_size_mb:.2f} MB (< 90 MB limit).")

    print("=" * 65)
    print("      DEEP MODEL TRAINING COMPLETED")
    print("=" * 65)
    print(f"Total Time:         {total_time:.1f} s ({total_time / 60:.2f} min)")
    print(f"Best Val ROC-AUC:   {best_auc:.4f}")
    print(f"Checkpoint Path:    {best_ckpt_path}")
    print(f"Checkpoint Size:    {ckpt_size_mb:.2f} MB (Constraint: < 90 MB)")
    print("=" * 65)

    return {
        "best_auc": best_auc,
        "total_time_sec": total_time,
        "checkpoint_path": str(best_ckpt_path),
        "checkpoint_size_mb": ckpt_size_mb,
        "history": history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VaultBreaker Deep Steganalysis Model")
    parser.add_argument("--profile", type=str, default="laptop", choices=["smoke", "laptop", "gpu"])
    parser.add_argument("--config", type=str, default=None, help="Path to configuration YAML file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Computation device")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_deep_pipeline(
        profile_name=args.profile,
        config_path=args.config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        resume_path=args.resume,
        seed=args.seed,
        device_override=args.device
    )
