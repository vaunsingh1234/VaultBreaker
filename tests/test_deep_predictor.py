import pytest
import time
import tempfile
from pathlib import Path
import numpy as np
import cv2
import torch

from vaultbreaker.inference.predict import VaultBreakerPredictor
from vaultbreaker.models.deep_net import VaultBreakerDeepNet


def test_predictor_fallback():
    """Verify predictor falls back to UnifiedMLP_Fallback when deep model is absent (T5.4)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        predictor = VaultBreakerPredictor(models_dir=tmp_path)
        assert predictor.is_deep_model is False
        assert predictor.model_name == "UnifiedMLP_Fallback"
        assert predictor.checkpoint_date is None

        # Test prediction on a synthetic test image
        img_p = tmp_path / "test_clean.png"
        cv2.imwrite(str(img_p), np.ones((64, 64), dtype=np.uint8) * 128)

        report = predictor.predict_file(img_p)
        assert report["model_name"] == "UnifiedMLP_Fallback"
        assert "verdict" in report


def test_predictor_deep_model_load_and_predict():
    """Verify predictor loads VaultBreakerDeepNet and exposes metadata (T5.2, T5.5)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create a mock deep model checkpoint
        net = VaultBreakerDeepNet(base_channels=16, latent_dim=128, use_fusion=False)
        ckpt_path = tmp_path / "deep_model.pt"
        torch.save({
            "model_state": net.state_dict(),
            "epoch": 2,
            "val_auc": 0.985,
            "timestamp": "2026-10-16 12:00:00"
        }, ckpt_path)

        predictor = VaultBreakerPredictor(models_dir=tmp_path)
        assert predictor.is_deep_model is True
        assert predictor.model_name == "VaultBreakerDeepNet"
        assert predictor.checkpoint_date == "2026-10-16 12:00:00"

        # Test deep prediction on a 256x256 image
        img_p = tmp_path / "test_deep_img.png"
        cv2.imwrite(str(img_p), np.random.randint(0, 256, (256, 256), dtype=np.uint8))

        report = predictor.predict_file(img_p)
        assert report["model_name"] == "VaultBreakerDeepNet"
        assert report["checkpoint_date"] == "2026-10-16 12:00:00"
        assert "calibrated_probability" in report
        assert "raw_logit" in report


def test_single_image_cpu_scan_speed():
    """Benchmark: Single image CPU scan must execute in < 5 seconds (T5.3)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        net = VaultBreakerDeepNet(base_channels=16, latent_dim=128, use_fusion=False)
        ckpt_path = tmp_path / "deep_model.pt"
        torch.save({
            "model_state": net.state_dict(),
            "epoch": 1,
            "val_auc": 0.95,
            "timestamp": "2026-10-16 12:00:00"
        }, ckpt_path)

        predictor = VaultBreakerPredictor(models_dir=tmp_path, device=torch.device("cpu"))
        img_p = tmp_path / "benchmark_img.png"
        cv2.imwrite(str(img_p), np.random.randint(0, 256, (256, 256), dtype=np.uint8))

        t0 = time.time()
        report = predictor.predict_file(img_p, generate_explanation=False)
        scan_time = time.time() - t0

        assert scan_time < 5.0, f"Single-image CPU scan took {scan_time:.2f}s, exceeding 5.0s requirement!"
        assert report["model_name"] == "VaultBreakerDeepNet"
