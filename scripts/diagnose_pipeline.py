#!/usr/bin/env python3
"""Diagnostic script to investigate Audio and Image steganography detection issues.
Phase 1: Diagnose Audio and Image First.
"""

import sys
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import cv2
from sklearn.metrics import roc_auc_score, roc_curve

from vaultbreaker.utils.config import load_config
from vaultbreaker.utils.io import compute_sha256
from vaultbreaker.embedders.audio_lsb import (
    embed_audio_lsb_replacement,
    extract_audio_lsb_replacement,
    embed_audio_lsb_matching,
    extract_audio_lsb_matching
)
from vaultbreaker.embedders.audio_echo import embed_audio_echo, extract_audio_echo
from vaultbreaker.embedders.image_lsb import (
    embed_image_lsb_replacement,
    extract_image_lsb_replacement,
    embed_image_lsb_matching
)
from vaultbreaker.features.audio_features import AudioFeatureExtractor
from vaultbreaker.features.image_features import ImageFeatureExtractor

def test_audio_embedders():
    print("==================================================================")
    print("TEST 1: AUDIO EMBEDDER TESTS (Fraction Changed vs Payload Rate)")
    print("==================================================================")
    sr = 16000
    duration = 3.0
    n_samples = int(sr * duration)
    
    # 1. Sine wave signal (non-silent)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    sine_samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    # 2. ESC-50 actual sample if available, or procedural
    audio_files = list(Path("data/raw/audio").glob("*.wav")) + list(Path("data/dataset/audio").glob("*.wav"))
    if audio_files:
        esc_samples, _ = sf.read(str(audio_files[0]), dtype="int16")
        if len(esc_samples) > n_samples:
            esc_samples = esc_samples[:n_samples]
        elif len(esc_samples) < n_samples:
            esc_samples = np.pad(esc_samples, (0, n_samples - len(esc_samples)))
    else:
        esc_samples = sine_samples.copy()

    # 3. Digital silence (all zeros)
    silence_samples = np.zeros(n_samples, dtype=np.int16)

    rates = [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]
    results = {}

    for name, sample_source in [("sine_440hz", sine_samples), ("esc_sample", esc_samples), ("digital_silence", silence_samples)]:
        results[name] = {}
        print(f"\n--- Testing on Signal: {name} (N={len(sample_source)}) ---")
        
        # Check silence fraction in raw sample
        zero_fraction = float(np.mean(sample_source == 0))
        abs_mean = float(np.mean(np.abs(sample_source)))
        print(f"    Raw stats: Zero sample fraction = {zero_fraction:.4f}, Mean absolute amplitude = {abs_mean:.2f}")

        for method in ["lsb_replacement", "lsb_matching", "echo_hiding"]:
            results[name][method] = []
            print(f"  Method: {method}")
            for rate in rates:
                rng = np.random.RandomState(42)
                if method == "echo_hiding":
                    n_bits = max(8, int(len(sample_source) / 2048 * min(rate, 0.8)))
                    payload = rng.randint(0, 2, size=n_bits, dtype=np.uint8)
                    stego, meta = embed_audio_echo(sample_source, payload)
                    extracted = extract_audio_echo(stego, meta)
                    bit_err = float(np.mean(extracted != payload)) if len(extracted) == len(payload) else 1.0
                elif method == "lsb_matching":
                    n_bits = max(32, int(len(sample_source) * rate))
                    payload = rng.randint(0, 2, size=n_bits, dtype=np.uint8)
                    stego, indices = embed_audio_lsb_matching(sample_source, payload, seed=42)
                    extracted = extract_audio_lsb_matching(stego, n_bits, seed=42)
                    bit_err = float(np.mean(extracted != payload))
                else: # lsb_replacement
                    n_bits = max(32, int(len(sample_source) * rate))
                    payload = rng.randint(0, 2, size=n_bits, dtype=np.uint8)
                    stego, indices = embed_audio_lsb_replacement(sample_source, payload, seed=42)
                    extracted = extract_audio_lsb_replacement(stego, n_bits, seed=42)
                    bit_err = float(np.mean(extracted != payload))

                # Measure actual fraction of changed samples
                changed_samples = np.sum(stego != sample_source)
                changed_fraction = float(changed_samples / len(sample_source))

                # Check clipping/overflow
                min_val, max_val = int(np.min(stego)), int(np.max(stego))
                overflow_risk = min_val < -32768 or max_val > 32767

                results[name][method].append({
                    "rate": rate,
                    "bits": int(n_bits),
                    "changed_samples": int(changed_samples),
                    "changed_fraction": changed_fraction,
                    "bit_err_rate": bit_err,
                    "min_val": min_val,
                    "max_val": max_val
                })
                print(f"    rate={rate:4.2f} | bits={n_bits:6d} | changed={changed_fraction:6.4f} (req={rate:.2f}) | BER={bit_err:6.4f} | range=[{min_val}, {max_val}]")

    return results

def test_audio_read_write_symmetry():
    print("\n==================================================================")
    print("TEST 2: AUDIO READ/WRITE RE-ENCODING SYMMETRY")
    print("==================================================================")
    temp_dir = Path("docs/RESULTS/diagnostics/tmp_audio")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    sr = 16000
    samples = np.random.randint(-10000, 10000, size=16000, dtype=np.int16)
    clean_path = temp_dir / "test_clean.wav"
    sf.write(str(clean_path), samples, sr, subtype="PCM_16")

    # Read back
    read_samples, read_sr = sf.read(str(clean_path), dtype="int16")
    diff = np.max(np.abs(read_samples - samples))
    print(f"Clean WAV read/write exact match max absolute diff: {diff} (0 means exact bit preservation)")

    # Test stego
    stego_samples, _ = embed_audio_lsb_replacement(samples, np.random.randint(0, 2, size=5000, dtype=np.uint8))
    stego_path = temp_dir / "test_stego.wav"
    sf.write(str(stego_path), stego_samples, sr, subtype="PCM_16")
    read_stego, _ = sf.read(str(stego_path), dtype="int16")
    diff_stego = np.max(np.abs(read_stego - stego_samples))
    print(f"Stego WAV read/write exact match max absolute diff: {diff_stego} (0 means exact bit preservation)")

def test_esc50_silence_and_features():
    print("\n==================================================================")
    print("TEST 3: ESC-50 SILENCE FRACTIONS & FEATURE SENSITIVITY")
    print("==================================================================")
    audio_files = list(Path("data/raw/audio").glob("*.wav")) + list(Path("data/dataset/audio").glob("*.wav"))
    print(f"Found {len(audio_files)} audio files on disk.")
    
    extractor = AudioFeatureExtractor(sample_rate=16000)

    silence_fractions = []
    rms_values = []
    for p in audio_files[:30]:
        data, sr = sf.read(str(p), dtype="int16")
        if data.ndim > 1:
            data = data[:, 0]
        # Frame-wise energy (1024 sample frames)
        frames = data[: len(data) - len(data) % 1024].reshape(-1, 1024)
        frame_rms = np.sqrt(np.mean(frames.astype(float) ** 2, axis=1))
        # silent frames RMS < 50
        silent_frames = np.mean(frame_rms < 50)
        silence_fractions.append(silent_frames)
        rms_values.append(np.mean(frame_rms))

    if silence_fractions:
        print(f"ESC-50 silent frame fraction: mean = {np.mean(silence_fractions):.4f}, max = {np.max(silence_fractions):.4f}, min = {np.min(silence_fractions):.4f}")
        print(f"ESC-50 frame RMS: mean = {np.mean(rms_values):.2f}, min = {np.min(rms_values):.2f}")

    # Now let's test feature changes on a real audio sample for clean vs LSB replacement at various rates!
    if audio_files:
        test_file = audio_files[0]
        clean_samples, sr = sf.read(str(test_file), dtype="int16")
        if clean_samples.ndim > 1:
            clean_samples = clean_samples[:, 0]
        if len(clean_samples) > 48000:
            clean_samples = clean_samples[:48000]

        feat_clean = extractor.extract(clean_samples, sr=sr)
        print("\nFeature sensitivity on sample audio:")
        print(f"{'Feature Name':20s} | {'Clean':10s} | {'LSB 0.2':10s} | {'LSB 0.7':10s} | {'LSB 1.0':10s}")
        print("-" * 70)
        
        stego_02, _ = embed_audio_lsb_replacement(clean_samples, np.random.randint(0, 2, size=int(0.2 * len(clean_samples)), dtype=np.uint8), seed=1)
        stego_07, _ = embed_audio_lsb_replacement(clean_samples, np.random.randint(0, 2, size=int(0.7 * len(clean_samples)), dtype=np.uint8), seed=2)
        stego_10, _ = embed_audio_lsb_replacement(clean_samples, np.random.randint(0, 2, size=len(clean_samples), dtype=np.uint8), seed=3)

        feat_02 = extractor.extract(stego_02, sr=sr)
        feat_07 = extractor.extract(stego_07, sr=sr)
        feat_10 = extractor.extract(stego_10, sr=sr)

        test_indices = [
            (0, "lsb_mean"),
            (1, "lsb_var"),
            (2, "lsb_trans"),
            (3, "lsb_entropy"),
            (4, "run_mean"),
            (6, "chi_norm"),
            (7, "chi_log"),
            (11, "d1_var"),
            (14, "d2_var"),
            (16, "mad_d1"),
            (18, "ent_d1")
        ]
        for idx, name in test_indices:
            print(f"{name:20s} | {feat_clean[idx]:10.4f} | {feat_02[idx]:10.4f} | {feat_07[idx]:10.4f} | {feat_10[idx]:10.4f}")

def test_image_feature_sensitivity():
    print("\n==================================================================")
    print("TEST 4: IMAGE FEATURE SENSITIVITY (Clean vs LSB Replacement vs Matching)")
    print("==================================================================")
    img_files = list(Path("data/raw/images").glob("*.jpg")) + list(Path("data/raw/images").glob("*.png")) + list(Path("data/dataset/image").glob("*.png"))
    if not img_files:
        print("No images found, creating synthetic test image.")
        test_img = np.random.randint(40, 220, size=(256, 256), dtype=np.uint8)
    else:
        test_img = cv2.imread(str(img_files[0]), cv2.IMREAD_GRAYSCALE)
        if test_img is None or test_img.shape != (256, 256):
            test_img = cv2.resize(test_img if test_img is not None else np.zeros((256, 256), dtype=np.uint8), (256, 256))

    img_extractor = ImageFeatureExtractor()
    feat_clean = img_extractor.extract(test_img)

    # Embed LSB replacement at 0.1, 0.4, 1.0 bpp
    stego_rep_01, _ = embed_image_lsb_replacement(test_img, np.random.randint(0, 2, size=int(0.1 * 256 * 256), dtype=np.uint8), seed=1)
    stego_rep_04, _ = embed_image_lsb_replacement(test_img, np.random.randint(0, 2, size=int(0.4 * 256 * 256), dtype=np.uint8), seed=2)
    stego_rep_10, _ = embed_image_lsb_replacement(test_img, np.random.randint(0, 2, size=256 * 256, dtype=np.uint8), seed=3)

    feat_rep_01 = img_extractor.extract(stego_rep_01)
    feat_rep_04 = img_extractor.extract(stego_rep_04)
    feat_rep_10 = img_extractor.extract(stego_rep_10)

    # Embed LSB matching at 0.4 bpp
    stego_mat_04, _ = embed_image_lsb_matching(test_img, np.random.randint(0, 2, size=int(0.4 * 256 * 256), dtype=np.uint8), seed=4)
    feat_mat_04 = img_extractor.extract(stego_mat_04)

    test_indices = [
        (0, "lsb_mean"),
        (2, "h_trans"),
        (4, "lsb_entropy"),
        (8, "chi_normalized"),
        (9, "chi_stat_log"),
        (10, "spa_diff"),
        (12, "lag1_corr"),
        (22, "res_h_var"),
        (30, "kv_var"),
        (31, "kv_mad")
    ]
    print(f"{'Feature Name':20s} | {'Clean':10s} | {'Rep 0.1':10s} | {'Rep 0.4':10s} | {'Rep 1.0':10s} | {'Match 0.4':10s}")
    print("-" * 80)
    for idx, name in test_indices:
        print(f"{name:20s} | {feat_clean[idx]:10.4f} | {feat_rep_01[idx]:10.4f} | {feat_rep_04[idx]:10.4f} | {feat_rep_10[idx]:10.4f} | {feat_mat_04[idx]:10.4f}")

def test_validation_threshold_conservatism():
    print("\n==================================================================")
    print("TEST 5: VALIDATION & TEST PREDICTIONS, THRESHOLD & OPERATING POINTS")
    print("==================================================================")
    import torch
    from vaultbreaker.models.unified_net import VaultBreakerUnifiedNet
    from vaultbreaker.training.calibrate import TemperatureScalingCalibrator
    
    ckpt_path = Path("models_saved/unified_model.pt")
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found.")
        return

    model = VaultBreakerUnifiedNet(
        img_dim=72,
        aud_dim=56,
        vid_dim=64,
        latent_dim=128,
        hidden_dim=256,
        use_aux_head=True,
        num_methods=8
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    calibrator_path = Path("models_saved/calibrator.joblib")
    if calibrator_path.exists():
        calibrator = TemperatureScalingCalibrator.load(calibrator_path)
    else:
        calibrator = TemperatureScalingCalibrator(target_fpr=0.05)

    print(f"Calibrator temperature: {calibrator.temperature:.4f}, Decision threshold: {calibrator.decision_threshold:.4f}, Target FPR: {calibrator.target_fpr}")

    for split_name in ["val", "test"]:
        split_file = Path(f"data/features/split_{split_name}.npz")
        if not split_file.exists():
            continue
        split_data = dict(np.load(split_file, allow_pickle=True))
        print(f"\n==================== SPLIT: {split_name.upper()} ====================")

        for m_type in ["image", "audio", "video"]:
            X = split_data[f"X_{m_type}"]
            y = split_data[f"y_{m_type}"]
            methods = split_data[f"methods_{m_type}"]
            rates = split_data[f"rates_{m_type}"]

            with torch.no_grad():
                x_tensor = torch.tensor(X, dtype=torch.float32)
                logits, _, _ = model(x_tensor, m_type)
                raw_probs = torch.sigmoid(logits).numpy().ravel()
                cal_probs = np.array([calibrator.calibrate_prob(p) for p in raw_probs])

            thresh = calibrator.decision_threshold
            preds = cal_probs >= thresh

            fpr, tpr, thresholds = roc_curve(y, cal_probs)
            auc = roc_auc_score(y, cal_probs) if len(np.unique(y)) > 1 else 0.5

            print(f"\n--- {m_type.upper()} ({split_name.upper()} N={len(y)}, AUC={auc:.4f}, Overall Acc@thresh={np.mean(preds == y):.4f}) ---")
            
            # FPR Operating Points
            for target_f in [0.01, 0.05, 0.10]:
                mask = fpr <= target_f
                if np.any(mask):
                    idx = np.where(mask)[0][-1]
                    print(f"  FPR <= {target_f:4.2f}: Actual FPR={fpr[idx]:.4f}, Recall={tpr[idx]:.4f}, Threshold={thresholds[idx]:.4f}")

            # Breakdown by method
            print(f"  {'Method':20s} | {'N':4s} | {'Mean Raw P':10s} | {'Mean Cal P':10s} | {'Acc@thresh':10s} | {'Recall':10s}")
            for m in sorted(np.unique(methods)):
                sub_mask = (methods == m)
                sub_y = y[sub_mask]
                sub_raw = raw_probs[sub_mask]
                sub_cal = cal_probs[sub_mask]
                sub_preds = preds[sub_mask]
                sub_acc = np.mean(sub_preds == sub_y)
                sub_rec = np.mean(sub_preds == 1) if m != "clean" else np.mean(sub_preds == 0)
                print(f"  {m:20s} | {len(sub_y):4d} | {np.mean(sub_raw):10.4f} | {np.mean(sub_cal):10.4f} | {sub_acc:10.4f} | {sub_rec:10.4f}")

            # Breakdown by payload rate (for stego only)
            stego_mask = (y == 1)
            print(f"  Payload Rates (Stego only):")
            for r in sorted(np.unique(rates[stego_mask])):
                r_mask = (rates == r) & stego_mask
                if np.sum(r_mask) > 0:
                    r_cal = cal_probs[r_mask]
                    r_rec = np.mean(preds[r_mask])
                    print(f"    rate={r:.2f} (N={np.sum(r_mask):2d}) | Mean Cal P={np.mean(r_cal):.4f} | Recall={r_rec:.4f}")

if __name__ == "__main__":
    t1_res = test_audio_embedders()
    test_audio_read_write_symmetry()
    test_esc50_silence_and_features()
    test_image_feature_sensitivity()
    test_validation_threshold_conservatism()
