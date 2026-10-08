import pytest
import numpy as np
import tempfile
from pathlib import Path
from vaultbreaker.features.image_features import ImageFeatureExtractor
from vaultbreaker.features.audio_features import AudioFeatureExtractor
from vaultbreaker.features.video_features import VideoFeatureExtractor
from vaultbreaker.data.generator import (
    generate_procedural_image, generate_procedural_audio, generate_procedural_video
)
from vaultbreaker.embedders.image_lsb import embed_image_lsb_replacement
from vaultbreaker.embedders.audio_lsb import embed_audio_lsb_replacement
from vaultbreaker.embedders.video_lsb import embed_video_frame_lsb

def test_image_features_shape_and_finite():
    extractor = ImageFeatureExtractor()
    assert extractor.FEATURE_DIM == 72

    img = generate_procedural_image(64, 64, seed=42)
    vec_clean = extractor.extract(img)
    assert vec_clean.shape == (72,)
    assert np.all(np.isfinite(vec_clean))
    assert not np.any(np.isnan(vec_clean))

    payload = np.random.randint(0, 2, size=200, dtype=np.uint8)
    stego, _ = embed_image_lsb_replacement(img, payload, seed=42)
    vec_stego = extractor.extract(stego)
    assert vec_stego.shape == (72,)
    assert np.all(np.isfinite(vec_stego))
    assert not np.array_equal(vec_clean, vec_stego)

def test_audio_features_shape_and_finite():
    extractor = AudioFeatureExtractor(sample_rate=16000)
    assert extractor.FEATURE_DIM == 56

    samples = generate_procedural_audio(sample_rate=16000, duration=0.5, seed=42)
    vec_clean = extractor.extract(samples, sr=16000)
    assert vec_clean.shape == (56,)
    assert np.all(np.isfinite(vec_clean))
    assert not np.any(np.isnan(vec_clean))

    payload = np.random.randint(0, 2, size=400, dtype=np.uint8)
    stego, _ = embed_audio_lsb_replacement(samples, payload, seed=42)
    vec_stego = extractor.extract(stego, sr=16000)
    assert vec_stego.shape == (56,)
    assert np.all(np.isfinite(vec_stego))
    assert not np.array_equal(vec_clean, vec_stego)

def test_video_features_shape_and_finite():
    extractor = VideoFeatureExtractor(sample_frames=4)
    assert extractor.FEATURE_DIM == 64

    frames, audio = generate_procedural_video(height=64, width=64, fps=10, duration=0.5, seed=42)
    vec_clean = extractor.extract(frames, audio_samples=audio, fps=10)
    assert vec_clean.shape == (64,)
    assert np.all(np.isfinite(vec_clean))
    assert not np.any(np.isnan(vec_clean))

    payload = np.random.randint(0, 2, size=300, dtype=np.uint8)
    stego_frames, _ = embed_video_frame_lsb(frames, payload, seed=42)
    vec_stego = extractor.extract(stego_frames, audio_samples=audio, fps=10)
    assert vec_stego.shape == (64,)
    assert np.all(np.isfinite(vec_stego))
    assert not np.array_equal(vec_clean, vec_stego)
