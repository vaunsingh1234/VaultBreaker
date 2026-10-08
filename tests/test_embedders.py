import pytest
import numpy as np
from vaultbreaker.embedders.image_lsb import (
    embed_image_lsb_replacement, extract_image_lsb_replacement,
    embed_image_lsb_matching, extract_image_lsb_matching,
    embed_image_edge_adaptive, extract_image_edge_adaptive
)
from vaultbreaker.embedders.image_dct import embed_image_dct, extract_image_dct
from vaultbreaker.embedders.audio_lsb import (
    embed_audio_lsb_replacement, extract_audio_lsb_replacement,
    embed_audio_lsb_matching, extract_audio_lsb_matching
)
from vaultbreaker.embedders.audio_echo import embed_audio_echo, extract_audio_echo
from vaultbreaker.embedders.video_lsb import (
    embed_video_frame_lsb, extract_video_frame_lsb,
    embed_video_audio_track, extract_video_audio_track
)

def test_image_lsb_replacement_roundtrip():
    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, size=(64, 64), dtype=np.uint8)
    payload = rng.randint(0, 2, size=200, dtype=np.uint8)
    
    stego, _ = embed_image_lsb_replacement(img, payload, seed=123)
    extracted = extract_image_lsb_replacement(stego, len(payload), seed=123)
    
    np.testing.assert_array_equal(payload, extracted)
    assert not np.array_equal(img, stego)

def test_image_lsb_matching_roundtrip():
    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, size=(64, 64), dtype=np.uint8)
    payload = rng.randint(0, 2, size=200, dtype=np.uint8)
    
    stego, _ = embed_image_lsb_matching(img, payload, seed=123)
    extracted = extract_image_lsb_matching(stego, len(payload), seed=123)
    
    np.testing.assert_array_equal(payload, extracted)
    diff = np.abs(stego.astype(int) - img.astype(int))
    assert np.max(diff) <= 1

def test_image_edge_adaptive_roundtrip():
    rng = np.random.RandomState(42)
    # Create image with textured box in center
    img = rng.randint(100, 150, size=(64, 64), dtype=np.uint8)
    img[20:44, 20:44] = rng.randint(0, 256, size=(24, 24), dtype=np.uint8)
    payload = rng.randint(0, 2, size=100, dtype=np.uint8)
    
    stego, indices = embed_image_edge_adaptive(img, payload, seed=123)
    extracted = extract_image_edge_adaptive(stego, indices)
    
    np.testing.assert_array_equal(payload, extracted)

def test_image_dct_roundtrip():
    rng = np.random.RandomState(42)
    img = rng.randint(40, 220, size=(64, 64), dtype=np.uint8)
    payload = rng.randint(0, 2, size=80, dtype=np.uint8)
    
    stego, meta = embed_image_dct(img, payload, seed=123)
    extracted = extract_image_dct(stego, meta)
    
    np.testing.assert_array_equal(payload, extracted)

def test_audio_lsb_replacement_roundtrip():
    rng = np.random.RandomState(42)
    samples = rng.randint(-20000, 20000, size=8000, dtype=np.int16)
    payload = rng.randint(0, 2, size=400, dtype=np.uint8)
    
    stego, _ = embed_audio_lsb_replacement(samples, payload, seed=123)
    extracted = extract_audio_lsb_replacement(stego, len(payload), seed=123)
    
    np.testing.assert_array_equal(payload, extracted)

def test_audio_lsb_matching_roundtrip():
    rng = np.random.RandomState(42)
    samples = rng.randint(-20000, 20000, size=8000, dtype=np.int16)
    payload = rng.randint(0, 2, size=400, dtype=np.uint8)
    
    stego, _ = embed_audio_lsb_matching(samples, payload, seed=123)
    extracted = extract_audio_lsb_matching(stego, len(payload), seed=123)
    
    np.testing.assert_array_equal(payload, extracted)
    diff = np.abs(stego.astype(int) - samples.astype(int))
    assert np.max(diff) <= 1

def test_audio_echo_roundtrip():
    rng = np.random.RandomState(42)
    # White noise signal has flat cepstrum, suitable for echo hiding
    samples = (rng.randn(16000) * 10000).astype(np.int16)
    payload = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
    
    stego, meta = embed_audio_echo(samples, payload, delay_0=40, delay_1=70, alpha=0.15, frame_len=2048)
    extracted = extract_audio_echo(stego, meta)
    
    # Verify echo is detected
    np.testing.assert_array_equal(payload, extracted)

def test_video_frame_lsb_roundtrip():
    rng = np.random.RandomState(42)
    frames = [rng.randint(0, 256, size=(32, 32), dtype=np.uint8) for _ in range(6)]
    payload = rng.randint(0, 2, size=150, dtype=np.uint8)
    
    stego_frames, meta = embed_video_frame_lsb(frames, payload, seed=123)
    extracted = extract_video_frame_lsb(stego_frames, meta)
    
    np.testing.assert_array_equal(payload, extracted)

def test_video_audio_track_roundtrip():
    rng = np.random.RandomState(42)
    audio = rng.randint(-15000, 15000, size=4000, dtype=np.int16)
    payload = rng.randint(0, 2, size=120, dtype=np.uint8)
    
    stego_audio, meta = embed_video_audio_track(audio, payload, seed=123)
    extracted = extract_video_audio_track(stego_audio, meta)
    
    np.testing.assert_array_equal(payload, extracted)
