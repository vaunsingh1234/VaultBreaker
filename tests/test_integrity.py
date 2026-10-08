import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import cv2
import soundfile as sf

from vaultbreaker.data.generator import (
    generate_procedural_image, generate_procedural_audio, generate_procedural_video,
    write_image, write_audio, write_video
)
from vaultbreaker.data.split import split_dataset_by_source
from vaultbreaker.data.integrity import assert_no_leakage, assert_reencoding_identity
from vaultbreaker.embedders.image_lsb import embed_image_lsb_replacement
from vaultbreaker.embedders.audio_lsb import embed_audio_lsb_replacement
from vaultbreaker.embedders.video_lsb import embed_video_frame_lsb

def test_source_disjoint_splitting_no_leakage():
    # Simulate records with paired clean/stego
    records = []
    for i in range(50):
        src_id = f"src_{i:03d}"
        records.append({"source_id": src_id, "label": 0, "media_type": "image"})
        records.append({"source_id": src_id, "label": 1, "media_type": "image"})
        
    df = split_dataset_by_source(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    
    # Assert no leakage
    assert assert_no_leakage(df) is True
    
    # Assert cover and stego share the same split
    for src_id in df["source_id"].unique():
        splits_for_src = df[df["source_id"] == src_id]["split"].unique()
        assert len(splits_for_src) == 1, f"Source {src_id} was split across {splits_for_src}!"

def test_reencoding_identity_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw = generate_procedural_image(64, 64, seed=42)
        payload = np.random.randint(0, 2, size=50, dtype=np.uint8)
        stego, _ = embed_image_lsb_replacement(raw, payload, seed=42)
        
        c_path = tmp_path / "c.png"
        s_path = tmp_path / "s.png"
        write_image(c_path, raw)
        write_image(s_path, stego)
        
        assert assert_reencoding_identity(c_path, s_path) is True

def test_reencoding_identity_audio():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw = generate_procedural_audio(sample_rate=16000, duration=0.5, seed=42)
        payload = np.random.randint(0, 2, size=100, dtype=np.uint8)
        stego, _ = embed_audio_lsb_replacement(raw, payload, seed=42)
        
        c_path = tmp_path / "c.wav"
        s_path = tmp_path / "s.wav"
        write_audio(c_path, raw, sample_rate=16000)
        write_audio(s_path, stego, sample_rate=16000)
        
        assert assert_reencoding_identity(c_path, s_path) is True

def test_reencoding_identity_video():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        frames, _ = generate_procedural_video(height=64, width=64, fps=10, duration=0.5, seed=42)
        payload = np.random.randint(0, 2, size=100, dtype=np.uint8)
        stego_frames, _ = embed_video_frame_lsb(frames, payload, seed=42)
        
        c_path = tmp_path / "c.mkv"
        s_path = tmp_path / "s.mkv"
        write_video(c_path, frames, fps=10)
        write_video(s_path, stego_frames, fps=10)
        
        assert assert_reencoding_identity(c_path, s_path) is True
