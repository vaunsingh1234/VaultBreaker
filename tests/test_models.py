import pytest
import torch
import numpy as np

from vaultbreaker.models.unified_net import VaultBreakerUnifiedNet
from vaultbreaker.models.baselines import BaselineModelsManager
from vaultbreaker.training.calibrate import TemperatureScalingCalibrator

def test_unified_net_forward_all_modalities():
    net = VaultBreakerUnifiedNet(img_dim=72, aud_dim=56, vid_dim=64, latent_dim=128, use_aux_head=True)
    net.eval()

    # Image
    x_img = torch.randn(4, 72)
    logits_img, aux_img, z_img = net(x_img, "image")
    assert logits_img.shape == (4, 1)
    assert aux_img.shape[0] == 4
    assert z_img.shape == (4, 128)

    # Audio
    x_aud = torch.randn(3, 56)
    logits_aud, aux_aud, z_aud = net(x_aud, "audio")
    assert logits_aud.shape == (3, 1)
    assert z_aud.shape == (3, 128)

    # Video
    x_vid = torch.randn(2, 64)
    logits_vid, aux_vid, z_vid = net(x_vid, "video")
    assert logits_vid.shape == (2, 1)
    assert z_vid.shape == (2, 128)

def test_calibrator_optimization():
    calibrator = TemperatureScalingCalibrator(target_fpr=0.05)
    # Synthetic uncalibrated overconfident logits
    logits = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
    labels = np.array([0, 0, 0, 1, 1, 1])

    calibrator.fit(logits, labels)
    assert calibrator.temperature > 0.0
    assert 0.0 <= calibrator.decision_threshold <= 1.0

    verdict, risk, details = calibrator.get_verdict_and_risk(0.85)
    assert "STEGO" in verdict or "CLEAN" in verdict
    assert "calibrated_probability" in details

def test_baseline_manager():
    mgr = BaselineModelsManager(random_state=42)
    models = mgr.get_models()
    assert "LogisticRegression" in models
    assert "GradientBoosting" in models
    assert "RandomForest" in models
    assert "SVM_RBF" in models
