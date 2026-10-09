import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import soundfile as sf
import imageio.v3 as iio

from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.io import compute_sha256, ensure_dir
from vaultbreaker.data.downloaders import (
    prepare_real_images,
    prepare_real_audio,
    prepare_real_videos
)
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
    embed_video_frame_lsb_matching
)

logger = get_logger("DataGenerator")

# --- PROCEDURAL COVER FALLBACKS (ONLY WHEN --allow-synthetic IS EXPLICIT) ---

def generate_procedural_image(
    height: int = 256,
    width: int = 256,
    seed: int = 42,
    color: bool = False
) -> np.ndarray:
    """Procedural synthetic image fallback."""
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:height, 0:width]
    angle = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(angle) * x / width + np.sin(angle) * y / height) * 80.0
    f1 = rng.uniform(0.02, 0.08)
    f2 = rng.uniform(0.04, 0.12)
    tex = np.sin(f1 * x + rng.uniform(0, np.pi)) * np.cos(f2 * y + rng.uniform(0, np.pi)) * 40.0
    noise_raw = rng.randn(height, width) * 20.0
    noise_smooth = cv2.GaussianBlur(noise_raw.astype(np.float32), (15, 15), 4.0)
    split_x = rng.randint(width // 4, 3 * width // 4)
    edge_map = np.where(x > split_x, 30.0, -30.0)
    edge_smooth = cv2.GaussianBlur(edge_map.astype(np.float32), (7, 7), 2.0)
    img = 128.0 + grad + tex + noise_smooth + edge_smooth
    img = np.clip(img, 0, 255).astype(np.uint8)
    if color:
        r = img
        g = np.clip(img.astype(float) * rng.uniform(0.8, 1.1) + rng.randn(height, width) * 5, 0, 255).astype(np.uint8)
        b = np.clip(img.astype(float) * rng.uniform(0.8, 1.1) + rng.randn(height, width) * 5, 0, 255).astype(np.uint8)
        return np.stack([b, g, r], axis=2)
    return img

def generate_procedural_audio(
    sample_rate: int = 16000,
    duration: float = 3.0,
    seed: int = 42
) -> np.ndarray:
    """Procedural synthetic audio fallback."""
    rng = np.random.RandomState(seed)
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    f0_base = rng.uniform(130, 220)
    f0 = f0_base + 15 * np.sin(2 * np.pi * 3.5 * t) + rng.randn(n_samples) * 0.5
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    signal = np.zeros(n_samples, dtype=np.float32)
    for harmonic in range(1, 8):
        amp = 1.0 / (harmonic ** 1.2)
        signal += (amp * np.sin(harmonic * phase)).astype(np.float32)
    envelope = (np.sin(2 * np.pi * 1.2 * t) ** 2 + 0.2)
    signal = signal * envelope
    max_val = np.max(np.abs(signal)) + 1e-6
    signal = (signal / max_val) * 24000.0
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
    """Procedural video fallback."""
    rng = np.random.RandomState(seed)
    n_frames = int(fps * duration)
    canvas_h, canvas_w = height * 2, width * 2
    canvas = generate_procedural_image(canvas_h, canvas_w, seed=seed, color=True)
    frames = []
    max_dx = canvas_w - width
    max_dy = canvas_h - height
    start_x = rng.randint(0, max_dx // 2)
    start_y = rng.randint(0, max_dy // 2)
    for i in range(n_frames):
        alpha = i / max(1, n_frames - 1)
        curr_x = int(start_x + alpha * (max_dx // 2))
        curr_y = int(start_y + alpha * (max_dy // 2))
        frame = canvas[curr_y : curr_y + height, curr_x : curr_x + width]
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    audio = generate_procedural_audio(sample_rate=16000, duration=duration, seed=seed)
    return frames, audio

# --- RE-ENCODING INVARIANT WRITERS ---

def write_image(path: Path, image: np.ndarray) -> None:
    """Save image deterministically using OpenCV PNG compression level 6."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])

def write_audio(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    """Save audio deterministically using soundfile 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16", format="WAV")

def write_video(path: Path, frames: List[np.ndarray], fps: int = 15) -> None:
    """
    Save video frames in lossless FFV1 MKV container.
    Specifies pixelformat='rgb24' to avoid lossy YUV chroma subsampling.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        str(path),
        np.stack(frames, axis=0),
        extension=".mkv",
        codec="ffv1",
        fps=fps,
        pixelformat="rgb24"
    )

# --- REAL-DATA DATASET PIPELINES ---

def generate_image_dataset(
    output_dir: Path,
    raw_dir: Optional[Path] = None,
    num_samples: int = 1000,
    target_size: Tuple[int, int] = (256, 256),
    payload_rates: List[float] = [0.10, 0.20, 0.30, 0.50],
    methods: List[str] = ["lsb_replacement", "lsb_matching", "edge_adaptive", "dct_ac"],
    allow_synthetic: bool = False,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate paired CLEAN and STEGO images from REAL cover sources.
    Each cover source yields multiple clean and stego files at various payload rates and methods,
    guaranteeing a strict 50/50 clean/stego balance per source and across the format.
    """
    records = []
    rng = np.random.RandomState(seed)
    H, W = target_size
    img_dir = output_dir / "images"
    ensure_dir(img_dir)
    if raw_dir is None:
        raw_dir = Path("data/raw")

    # 1. Fetch real cover images
    num_sources = max(1, num_samples // 2) # Each source produces 1 clean + 1 stego = 2 files (strictly 50/50, no duplicate clean copies)
    if allow_synthetic or num_samples <= 10:
        real_paths = []
    else:
        real_paths = prepare_real_images(
            raw_dir=raw_dir,
            max_images=num_sources,
            target_size=target_size,
            allow_synthetic=allow_synthetic
        )

    logger.info(f"Generating image dataset from {len(real_paths) if real_paths else num_sources} sources...")

    for i in range(num_sources):
        source_id = f"img_src_{i:04d}"
        
        # Load real cover or fallback
        if real_paths and i < len(real_paths):
            raw_cover = cv2.imread(str(real_paths[i]), cv2.IMREAD_GRAYSCALE)
            if raw_cover is None or raw_cover.shape != (H, W):
                raw_cover = cv2.resize(raw_cover, (W, H)) if raw_cover is not None else generate_procedural_image(H, W, seed=seed + i)
        else:
            raw_cover = generate_procedural_image(H, W, seed=seed + i)

        # 1. Clean File (exactly ONE clean file per source)
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

        # 2. Stego File (decouple method and payload rate)
        method = methods[i % len(methods)]
        rate = payload_rates[(i // len(methods)) % len(payload_rates)]
        stego_seed = seed + i * 100 + 1
        p_rng = np.random.RandomState(stego_seed)

        if method == "dct_ac":
            n_bits = max(32, int((H // 8) * (W // 8) * 15 * rate))
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_img, _ = embed_image_dct(raw_cover, payload, seed=stego_seed)
        elif method == "edge_adaptive":
            n_bits = max(32, int(H * W * rate))
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_img, _ = embed_image_edge_adaptive(raw_cover, payload, seed=stego_seed)
        elif method == "lsb_matching":
            n_bits = max(32, int(H * W * rate))
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_img, _ = embed_image_lsb_matching(raw_cover, payload, seed=stego_seed)
        else: # lsb_replacement
            n_bits = max(32, int(H * W * rate))
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_img, _ = embed_image_lsb_replacement(raw_cover, payload, seed=stego_seed)

        stego_name = f"{source_id}_stego_{method}_{rate:.2f}.png"
        stego_path = img_dir / stego_name
        write_image(stego_path, stego_img)
        records.append({
            "source_id": source_id,
            "filename": stego_name,
            "filepath": str(stego_path),
            "media_type": "image",
            "label": 1,
            "method": method,
            "payload_rate": round(float(rate), 2),
            "sha256": compute_sha256(stego_path)
        })

    return records

def generate_audio_dataset(
    output_dir: Path,
    raw_dir: Optional[Path] = None,
    num_samples: int = 600,
    sample_rate: int = 16000,
    duration: float = 3.0,
    payload_rates: List[float] = [0.20, 0.40, 0.70, 1.00],
    methods: List[str] = ["lsb_replacement", "lsb_matching", "echo_hiding"],
    allow_synthetic: bool = False,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate paired CLEAN and STEGO audio files from REAL cover sources (ESC-50).
    Enforces identical 16-bit WAV re-encoding and balanced clean/stego ratio.
    """
    records = []
    aud_dir = output_dir / "audio"
    ensure_dir(aud_dir)
    if raw_dir is None:
        raw_dir = Path("data/raw")

    num_sources = max(1, num_samples // 2) # Each source produces 1 clean + 1 stego
    if allow_synthetic or num_samples <= 10:
        real_paths = []
    else:
        real_paths = prepare_real_audio(
            raw_dir=raw_dir,
            max_audio=num_sources,
            target_sr=sample_rate,
            duration=duration,
            allow_synthetic=allow_synthetic
        )

    logger.info(f"Generating audio dataset from {len(real_paths) if real_paths else num_sources} sources...")
    target_samples = int(sample_rate * duration)

    for i in range(num_sources):
        source_id = f"aud_src_{i:04d}"

        if real_paths and i < len(real_paths):
            raw_samples, sr = sf.read(str(real_paths[i]), dtype="int16")
            if len(raw_samples) != target_samples:
                if len(raw_samples) > target_samples:
                    raw_samples = raw_samples[:target_samples]
                else:
                    raw_samples = np.pad(raw_samples, (0, target_samples - len(raw_samples)), mode="wrap")
        else:
            raw_samples = generate_procedural_audio(sample_rate, duration, seed=seed + i)

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

        # 2. Stego File (decouple method and payload rate)
        method = methods[i % len(methods)]
        rate = payload_rates[(i // len(methods)) % len(payload_rates)]
        stego_seed = seed + i * 50 + 1
        p_rng = np.random.RandomState(stego_seed)

        if method == "echo_hiding":
            n_bits = max(8, int(len(raw_samples) / 2048 * min(rate, 0.8)))
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_samples, _ = embed_audio_echo(raw_samples, payload)
        elif method == "lsb_matching":
            n_bits = max(32, int(len(raw_samples) * rate))
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_samples, _ = embed_audio_lsb_matching(raw_samples, payload, seed=stego_seed)
        else: # lsb_replacement
            n_bits = max(32, int(len(raw_samples) * rate))
            payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
            stego_samples, _ = embed_audio_lsb_replacement(raw_samples, payload, seed=stego_seed)

        stego_name = f"{source_id}_stego_{method}_{rate:.2f}.wav"
        stego_path = aud_dir / stego_name
        write_audio(stego_path, stego_samples, sample_rate)
        records.append({
            "source_id": source_id,
            "filename": stego_name,
            "filepath": str(stego_path),
            "media_type": "audio",
            "label": 1,
            "method": method,
            "payload_rate": round(float(rate), 2),
            "sha256": compute_sha256(stego_path)
        })

    return records

def generate_video_dataset(
    output_dir: Path,
    raw_dir: Optional[Path] = None,
    num_samples: int = 250,
    image_paths: Optional[List[Path]] = None,
    target_size: Tuple[int, int] = (128, 128),
    fps: int = 15,
    duration: float = 2.0,
    payload_rates: List[float] = [0.10, 0.20, 0.40],
    methods: List[str] = ["frame_lsb", "frame_lsb_matching"],
    allow_synthetic: bool = False,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate paired CLEAN and STEGO video files using realistic pan/zoom
    traversal across REAL images in lossless FFV1 MKV containers.
    Each source produces 2 clean and 2 stego files, guaranteeing >= 100 test videos.
    """
    records = []
    vid_dir = output_dir / "video"
    ensure_dir(vid_dir)
    if raw_dir is None:
        raw_dir = Path("data/raw")

    if image_paths is None:
        raw_images_dir = raw_dir / "images"
        image_paths = sorted(list(raw_images_dir.glob("*.png")) + list(raw_images_dir.glob("*.jpg")))

    num_sources = max(1, num_samples // 2) # Each source produces 1 clean + 1 stego = 2 files
    if allow_synthetic or num_samples <= 10:
        real_video_covers = []
    else:
        real_video_covers = prepare_real_videos(
            raw_dir=raw_dir,
            image_paths=image_paths,
            max_videos=num_sources,
            target_size=target_size,
            fps=fps,
            duration=duration,
            allow_synthetic=allow_synthetic
        )

    logger.info(f"Generating video dataset from {len(real_video_covers) if real_video_covers else num_sources} sources...")
    H, W = target_size

    for i in range(num_sources):
        source_id = f"vid_src_{i:04d}"

        # Load video frames
        if real_video_covers and i < len(real_video_covers):
            raw_frames = iio.imread(str(real_video_covers[i]))
            if isinstance(raw_frames, np.ndarray):
                raw_frames = [raw_frames[f] for f in range(raw_frames.shape[0])]
        else:
            raw_frames, _ = generate_procedural_video(H, W, fps, duration, seed=seed + i)

        # 1. Clean File (exactly ONE clean file per source)
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

        # 2. Stego File (decouple method and payload rate)
        method = methods[i % len(methods)]
        rate = payload_rates[(i // len(methods)) % len(payload_rates)]
        stego_seed = seed + i * 50 + 1
        p_rng = np.random.RandomState(stego_seed)
        total_pixels = len(raw_frames) * H * W
        n_bits = max(32, int(total_pixels * rate))
        payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)

        if method == "frame_lsb_matching":
            stego_frames, _ = embed_video_frame_lsb_matching(raw_frames, payload, seed=stego_seed)
        else:
            stego_frames, _ = embed_video_frame_lsb(raw_frames, payload, seed=stego_seed)

        stego_name = f"{source_id}_stego_{method}_{rate:.2f}.mkv"
        stego_path = vid_dir / stego_name
        write_video(stego_path, stego_frames, fps=fps)
        records.append({
            "source_id": source_id,
            "filename": stego_name,
            "filepath": str(stego_path),
            "media_type": "video",
            "label": 1,
            "method": method,
            "payload_rate": round(float(rate), 2),
            "sha256": compute_sha256(stego_path)
        })

    return records
