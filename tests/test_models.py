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


def test_deep_net_output_shapes():
    """Verify VaultBreakerDeepNet output shapes across image, audio, and video modalities (T2.4, T8.1)."""
    from vaultbreaker.models.deep_net import VaultBreakerDeepNet
    net = VaultBreakerDeepNet(base_channels=16, latent_dim=128)
    net.eval()

    # Image: (B=2, C=1, H=256, W=256)
    x_img = torch.randn(2, 1, 256, 256)
    logits_img, aux_img, z_img = net(x_img, "image")
    assert logits_img.shape == (2, 1)
    assert z_img.shape == (2, 128)

    # Audio: (B=2, C=1, L=48000)
    x_aud = torch.randn(2, 1, 48000)
    logits_aud, aux_aud, z_aud = net(x_aud, "audio")
    assert logits_aud.shape == (2, 1)
    assert z_aud.shape == (2, 128)

    # Video: (B=2, K=8, C=1, H=128, W=128)
    x_vid = torch.randn(2, 8, 1, 128, 128)
    logits_vid, aux_vid, z_vid = net(x_vid, "video")
    assert logits_vid.shape == (2, 1)
    assert z_vid.shape == (2, 128)


def test_srm_frozen():
    """Verify SRM filter kernels are frozen and unaffected by backward pass (T2.1, T2.3, T8.2)."""
    from vaultbreaker.models.deep_net import VaultBreakerDeepNet
    net = VaultBreakerDeepNet(base_channels=16)
    assert not net.image_deep.srm.conv.weight.requires_grad

    initial_srm_weights = net.image_deep.srm.conv.weight.clone()

    # Run forward and backward
    net.train()
    x = torch.randn(2, 1, 64, 64)
    logits, _, _ = net(x, "image")
    loss = logits.sum()
    loss.backward()

    # Weight must not have gradient and must be unchanged
    assert net.image_deep.srm.conv.weight.grad is None
    assert torch.allclose(net.image_deep.srm.conv.weight, initial_srm_weights)


def test_audio_diff_frozen():
    """Verify Audio difference filter kernels are frozen and unaffected by backward pass (T2.2, T2.3, T8.3)."""
    from vaultbreaker.models.deep_net import VaultBreakerDeepNet
    net = VaultBreakerDeepNet(base_channels=16)
    assert not net.audio_deep.diff_layer.conv.weight.requires_grad

    initial_diff_weights = net.audio_deep.diff_layer.conv.weight.clone()

    # Run forward and backward
    net.train()
    x = torch.randn(2, 1, 16000)
    logits, _, _ = net(x, "audio")
    loss = logits.sum()
    loss.backward()

    # Weight must not have gradient and must be unchanged
    assert net.audio_deep.diff_layer.conv.weight.grad is None
    assert torch.allclose(net.audio_deep.diff_layer.conv.weight, initial_diff_weights)


def test_deep_net_determinism():
    """Verify identical seed yields identical logits across forward passes (T2.5, T8.4)."""
    from vaultbreaker.models.deep_net import VaultBreakerDeepNet
    torch.manual_seed(42)
    net1 = VaultBreakerDeepNet(base_channels=16)
    net1.eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        logits1, _, z1 = net1(x, "image")

    torch.manual_seed(42)
    net2 = VaultBreakerDeepNet(base_channels=16)
    net2.eval()
    with torch.no_grad():
        logits2, _, z2 = net2(x, "image")

    assert torch.allclose(logits1, logits2, atol=1e-6)
    assert torch.allclose(z1, z2, atol=1e-6)


def test_video_shared_frame_backbone():
    """Verify VideoTemporalBackbone reuses the identical SRNet image backbone (T2.6)."""
    from vaultbreaker.models.deep_net import VaultBreakerDeepNet
    net = VaultBreakerDeepNet(base_channels=16)
    assert net.video_deep.frame_backbone is net.image_deep

