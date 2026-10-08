import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import soundfile as sf
import imageio.v3 as iio

from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.io import compute_sha256, ensure_dir
from vaultbreaker.embedders.image_lsb import (
    embed_image_lsb_replacement,
    embed_image_lsb_matching,
    embed_image_edge_adaptive
)
from vaultbreaker.embedders.image_dct import embed_image_dct
from vaultbreaker.embedders.audio_lsb import (
    embed_audio_lsb_replacement,
    embed_audio_lsb_matching
)
from vaultbreaker.embedders.audio_echo import embed_audio_echo
from vaultbreaker.embedders.video_lsb import (
    embed_video_frame_lsb,
    embed_video_audio_track
)

logger = get_logger("DataGenerator")

# --- PROCEDURAL COVER GENERATORS ---

def generate_procedural_image(
    height: int = 256,
    width: int = 256,
    seed: int = 42,
    color: bool = False
) -> np.ndarray:
    """
    Synthesize naturalistic cover images with realistic textures,
    gradients, edges, and multi-octave Perlin-style noise.
    """
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:height, 0:width]
    
    # Base gradient
    angle = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(angle) * x / width + np.sin(angle) * y / height) * 80.0
    
    # Multi-frequency sinusoidal textures
    f1 = rng.uniform(0.02, 0.08)
    f2 = rng.uniform(0.04, 0.12)
    tex = np.sin(f1 * x + rng.uniform(0, np.pi)) * np.cos(f2 * y + rng.uniform(0, np.pi)) * 40.0
    
    # Multi-scale smoothed noise
    noise_raw = rng.randn(height, width) * 20.0
    noise_smooth = cv2.GaussianBlur(noise_raw.astype(np.float32), (15, 15), 4.0)
    
    # Sharp edge structure (geometric partition)
    split_x = rng.randint(width // 4, 3 * width // 4)
    edge_map = np.where(x > split_x, 30.0, -30.0)
    edge_smooth = cv2.GaussianBlur(edge_map.astype(np.float32), (7, 7), 2.0)
    
    img = 128.0 + grad + tex + noise_smooth + edge_smooth
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    if color:
        # Create correlated RGB channels
        r = img
        g = np.clip(img.astype(float) * rng.uniform(0.8, 1.1) + rng.randn(height, width) * 5, 0, 255).astype(np.uint8)
        b = np.clip(img.astype(float) * rng.uniform(0.8, 1.1) + rng.randn(height, width) * 5, 0, 255).astype(np.uint8)
        return np.stack([b, g, r], axis=2) # BGR for cv2
    return img

def generate_procedural_audio(
    sample_rate: int = 16000,
    duration: float = 3.0,
    seed: int = 42
) -> np.ndarray:
    """
    Synthesize acoustic speech-like audio cover with harmonic formants,
    AM/FM modulation, pitch variation, and realistic noise floor.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # Pitch contour with natural vibrato / speech prosody
    f0_base = rng.uniform(130, 220)
    f0 = f0_base + 15 * np.sin(2 * np.pi * 3.5 * t) + rng.randn(n_samples) * 0.5
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    
    # Harmonics with formant resonant peaks (vowel-like)
    signal = np.zeros(n_samples, dtype=np.float32)
    for harmonic in range(1, 10):
        amp = 1.0 / (harmonic ** 1.2)
        # Formant boost near 800 Hz and 2200 Hz
        center_freq = f0_base * harmonic
        boost = 1.0 + 2.0 * np.exp(-((center_freq - 800) ** 2) / (2 * (250 ** 2))) + \
                      1.5 * np.exp(-((center_freq - 2200) ** 2) / (2 * (400 ** 2)))
        signal += (amp * boost * np.sin(harmonic * phase)).astype(np.float32)
        
    # Amplitude envelope
    envelope = (np.sin(2 * np.pi * 1.2 * t) ** 2 + 0.2)
    signal = signal * envelope
    
    # Normalize to 16-bit range
    max_val = np.max(np.abs(signal)) + 1e-6
    signal = (signal / max_val) * 26000.0
    
    # Add subtle realistic acoustic room noise
    ambient = rng.randn(n_samples) * 300.0
    samples = np.clip(signal + ambient, -32768, 32767).astype(np.int16)
    return samples

def generate_procedural_video(
    height: int = 128,
    width: int = 128,
    fps: int = 15,
    duration: float = 2.0,
    seed: int = 42
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Synthesize realistic video by panning and zooming across a larger
    high-resolution textured canvas, creating continuous natural motion.
    Also returns synchronous audio samples.
    """
    rng = np.random.RandomState(seed)
    n_frames = int(fps * duration)
    canvas_h, canvas_w = height * 2, width * 2
    canvas = generate_procedural_image(canvas_h, canvas_w, seed=seed, color=True)
    
    frames = []
    # Smooth panning path
    max_dx = canvas_w - width
    max_dy = canvas_h - height
    start_x = rng.randint(0, max_dx // 2)
    start_y = rng.randint(0, max_dy // 2)
    
    for i in range(n_frames):
        alpha = i / max(1, n_frames - 1)
        curr_x = int(start_x + alpha * (max_dx // 2))
        curr_y = int(start_y + alpha * (max_dy // 2))
        frame = canvas[curr_y : curr_y + height, curr_x : curr_x + width]
        frames.append(frame)
        
    audio = generate_procedural_audio(sample_rate=16000, duration=duration, seed=seed)
    return frames, audio

# --- RE-ENCODING INVARIANT WRITERS ---

def write_image(path: Path, image: np.ndarray) -> None:
    """Save image using OpenCV with deterministic parameters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in [".png"]:
        # Compression level 6 (standard PNG)
        cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    elif ext in [".jpg", ".jpeg"]:
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(path), image)

def write_audio(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    """Save audio using soundfile in 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16", format="WAV")

def write_video(path: Path, frames: List[np.ndarray], fps: int = 15) -> None:
    """
    Save video frames in lossless container (FFV1 in MKV or AVI)
    so compression artifacts do not alter or destroy bit payloads.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # imageio with lossless ffv1 codec in mkv
    iio.imwrite(
        str(path),
        np.stack(frames, axis=0),
        extension=".mkv",
        codec="ffv1",
        fps=fps
    )

# --- PIPELINE BUILDERS ---

def generate_image_dataset(
    output_dir: Path,
    num_samples: int,
    target_size: Tuple[int, int] = (256, 256),
    payload_rates: List[float] = [0.05, 0.1, 0.2, 0.4],
    methods: List[str] = ["lsb_replacement", "lsb_matching", "edge_adaptive", "dct_ac"],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate paired CLEAN and STEGO images enforcing identical re-encoding.
    Each cover yields 1 clean file and 1 stego file for balanced 50/50 ratio.
    """
    records = []
    rng = np.random.RandomState(seed)
    H, W = target_size
    img_dir = output_dir / "images"
    ensure_dir(img_dir)
    
    num_pairs = num_samples // 2
    logger.info(f"Generating {num_pairs} image pairs (Total: {num_pairs * 2} images)...")

    for i in range(num_pairs):
        source_id = f"img_src_{i:04d}"
        cover_seed = seed + i * 10
        raw_cover = generate_procedural_image(H, W, seed=cover_seed, color=False)
        
        # 1. Clean Cover File
        clean_name = f"{source_id}_clean.png"
        clean_path = img_dir / clean_name
        write_image(clean_path, raw_cover)
        
        records.append({
            "source_id": source_id,
            "filename": clean_name,
            "filepath": str(clean_path),
            "media_type": "image",
            "label": 0,
            "method": "clean",
            "payload_rate": 0.0,
            "sha256": compute_sha256(clean_path)
        })
        
        # 2. Stego File
        method = methods[i % len(methods)]
        rate = payload_rates[i % len(payload_rates)]
        stego_seed = seed + i * 10 + 1
        
        if method == "dct_ac":
            # Capacity in 8x8 blocks with mid-frequencies
            n_bits = max(16, int((H // 8) * (W // 8) * 15 * rate))
            p_rng = np.random.RandomState(stego_seed)
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_img, _ = embed_image_dct(raw_cover, payload, seed=stego_seed)
        elif method == "edge_adaptive":
            n_bits = max(16, int(H * W * rate))
            p_rng = np.random.RandomState(stego_seed)
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_img, _ = embed_image_edge_adaptive(raw_cover, payload, seed=stego_seed)
        elif method == "lsb_matching":
            n_bits = max(16, int(H * W * rate))
            p_rng = np.random.RandomState(stego_seed)
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_img, _ = embed_image_lsb_matching(raw_cover, payload, seed=stego_seed)
        else: # lsb_replacement
            n_bits = max(16, int(H * W * rate))
            p_rng = np.random.RandomState(stego_seed)
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_img, _ = embed_image_lsb_replacement(raw_cover, payload, seed=stego_seed)
            
        stego_name = f"{source_id}_stego_{method}.png"
        stego_path = img_dir / stego_name
        write_image(stego_path, stego_img)
        
        records.append({
            "source_id": source_id,
            "filename": stego_name,
            "filepath": str(stego_path),
            "media_type": "image",
            "label": 1,
            "method": method,
            "payload_rate": float(rate),
            "sha256": compute_sha256(stego_path)
        })
        
    return records

def generate_audio_dataset(
    output_dir: Path,
    num_samples: int,
    sample_rate: int = 16000,
    duration: float = 3.0,
    payload_rates: List[float] = [0.05, 0.1, 0.3, 0.5],
    methods: List[str] = ["lsb_replacement", "lsb_matching", "echo_hiding"],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Generate paired CLEAN and STEGO audio files enforcing identical re-encoding."""
    records = []
    aud_dir = output_dir / "audio"
    ensure_dir(aud_dir)
    
    num_pairs = num_samples // 2
    logger.info(f"Generating {num_pairs} audio pairs (Total: {num_pairs * 2} files)...")
    
    for i in range(num_pairs):
        source_id = f"aud_src_{i:04d}"
        cover_seed = seed + i * 10
        raw_samples = generate_procedural_audio(sample_rate, duration, seed=cover_seed)
        
        # 1. Clean File
        clean_name = f"{source_id}_clean.wav"
        clean_path = aud_dir / clean_name
        write_audio(clean_path, raw_samples, sample_rate)
        
        records.append({
            "source_id": source_id,
            "filename": clean_name,
            "filepath": str(clean_path),
            "media_type": "audio",
            "label": 0,
            "method": "clean",
            "payload_rate": 0.0,
            "sha256": compute_sha256(clean_path)
        })
        
        # 2. Stego File
        method = methods[i % len(methods)]
        rate = payload_rates[i % len(payload_rates)]
        stego_seed = seed + i * 10 + 1
        
        if method == "echo_hiding":
            n_bits = max(4, int(len(raw_samples) / 2048 * min(rate, 0.8)))
            p_rng = np.random.RandomState(stego_seed)
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_samples, _ = embed_audio_echo(raw_samples, payload)
        elif method == "lsb_matching":
            n_bits = max(16, int(len(raw_samples) * rate))
            p_rng = np.random.RandomState(stego_seed)
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_samples, _ = embed_audio_lsb_matching(raw_samples, payload, seed=stego_seed)
        else: # lsb_replacement
            n_bits = max(16, int(len(raw_samples) * rate))
            p_rng = np.random.RandomState(stego_seed)
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_samples, _ = embed_audio_lsb_replacement(raw_samples, payload, seed=stego_seed)
            
        stego_name = f"{source_id}_stego_{method}.wav"
        stego_path = aud_dir / stego_name
        write_audio(stego_path, stego_samples, sample_rate)
        
        records.append({
            "source_id": source_id,
            "filename": stego_name,
            "filepath": str(stego_path),
            "media_type": "audio",
            "label": 1,
            "method": method,
            "payload_rate": float(rate),
            "sha256": compute_sha256(stego_path)
        })
        
    return records

def generate_video_dataset(
    output_dir: Path,
    num_samples: int,
    target_size: Tuple[int, int] = (128, 128),
    fps: int = 15,
    duration: float = 2.0,
    payload_rates: List[float] = [0.05, 0.1, 0.2],
    methods: List[str] = ["frame_lsb", "audio_lsb"],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Generate paired CLEAN and STEGO video files in lossless FFV1 container."""
    records = []
    vid_dir = output_dir / "video"
    ensure_dir(vid_dir)
    
    num_pairs = num_samples // 2
    logger.info(f"Generating {num_pairs} video pairs (Total: {num_pairs * 2} files)...")
    H, W = target_size
    
    for i in range(num_pairs):
        source_id = f"vid_src_{i:04d}"
        cover_seed = seed + i * 10
        raw_frames, raw_audio = generate_procedural_video(H, W, fps, duration, seed=cover_seed)
        
        # 1. Clean File
        clean_name = f"{source_id}_clean.mkv"
        clean_path = vid_dir / clean_name
        write_video(clean_path, raw_frames, fps=fps)
        
        records.append({
            "source_id": source_id,
            "filename": clean_name,
            "filepath": str(clean_path),
            "media_type": "video",
            "label": 0,
            "method": "clean",
            "payload_rate": 0.0,
            "sha256": compute_sha256(clean_path)
        })
        
        # 2. Stego File
        method = methods[i % len(methods)]
        rate = payload_rates[i % len(payload_rates)]
        stego_seed = seed + i * 10 + 1
        
        if method == "audio_lsb":
            # Stego in audio track
            n_bits = max(16, int(len(raw_audio) * rate))
            p_rng = np.random.RandomState(stego_seed)
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_audio, _ = embed_video_audio_track(raw_audio, payload, seed=stego_seed)
            # Re-save video frames identically
            stego_name = f"{source_id}_stego_{method}.mkv"
            stego_path = vid_dir / stego_name
            write_video(stego_path, raw_frames, fps=fps)
        else: # frame_lsb
            total_pixels = len(raw_frames) * H * W
            n_bits = max(16, int(total_pixels * rate))
            p_rng = np.random.RandomState(stego_seed)
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_frames, _ = embed_video_frame_lsb(raw_frames, payload, seed=stego_seed)
            stego_name = f"{source_id}_stego_{method}.mkv"
            stego_path = vid_dir / stego_name
            write_video(stego_path, stego_frames, fps=fps)
            
        records.append({
            "source_id": source_id,
            "filename": stego_name,
            "filepath": str(stego_path),
            "media_type": "video",
            "label": 1,
            "method": method,
            "payload_rate": float(rate),
            "sha256": compute_sha256(stego_path)
        })
        
    return records
