import os
import shutil
import urllib.request
import tarfile
import zipfile
import io
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import soundfile as sf
import librosa
import imageio.v3 as iio
import psutil

from vaultbreaker.utils.logger import get_logger

logger = get_logger("Downloader")

IMAGENETTE_URLS = [
    "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    "https://s3.amazonaws.com/fast-ai-imagerec/imagenette2-160.tgz"
]

ESC50_URLS = [
    "https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master",
    "https://github.com/karolpiczak/ESC-50/archive/master.zip"
]

def get_system_capacity() -> Dict[str, float]:
    """Returns available RAM (GB) and available disk space (GB)."""
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    disk_stat = shutil.disk_usage(".")
    disk_free_gb = disk_stat.free / (1024 ** 3)
    return {"ram_gb": ram_gb, "disk_free_gb": disk_free_gb}

def get_ram_aware_dataset_sizes(
    config_img: Optional[int] = None,
    config_aud: Optional[int] = None,
    config_vid: Optional[int] = None,
    fast: bool = False
) -> Dict[str, int]:
    """
    Computes optimal dataset cover counts based on system memory and disk limits.
    Default targets on >=8GB RAM: 2000 images, 1000 audio, 250 video covers.
    Auto-reduces if machine has < 8GB RAM or very limited disk space.
    """
    if fast:
        return {"images": 40, "audio": 30, "video": 16}

    caps = get_system_capacity()
    ram_gb = caps["ram_gb"]
    disk_free_gb = caps["disk_free_gb"]

    # Base target values
    target_img = config_img or 4000
    target_aud = config_aud or 2000
    target_vid = config_vid or 700

    if ram_gb < 7.5 or disk_free_gb < 1.0:
        factor = 0.5
        target_img = max(400, int(target_img * factor))
        target_aud = max(300, int(target_aud * factor))
        target_vid = max(80, int(target_vid * factor))
        logger.warning(
            f"SYSTEM CONSTRAINT DETECTED (RAM: {ram_gb:.1f} GB, Free Disk: {disk_free_gb:.1f} GB). "
            f"Auto-scaled targets: {target_img} images, {target_aud} audio, {target_vid} videos."
        )
    else:
        logger.info(
            f"System specs verified (RAM: {ram_gb:.1f} GB, Free Disk: {disk_free_gb:.1f} GB). "
            f"Selected full targets: {target_img} images, {target_aud} audio, {target_vid} videos."
        )

    return {"images": target_img, "audio": target_aud, "video": target_vid}

def download_stream(urls: List[str], dest_file: Path, timeout: int = 300) -> bool:
    """Download file with progress logging and mirror failover."""
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    
    for url in urls:
        logger.info(f"Connecting to download source: {url}")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "VaultBreaker-Forensics/2.0 (Mozilla/5.0)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = response.headers.get("Content-Length")
                total_bytes = int(total_size) if total_size else 0
                downloaded = 0
                block_size = 1024 * 1024 # 1MB chunks
                
                start_time = time.time()
                with open(dest_file, "wb") as f_out:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        downloaded += len(chunk)
                        if total_bytes > 0 and downloaded % (10 * block_size) == 0:
                            pct = (downloaded / total_bytes) * 100
                            mb = downloaded / (1024 * 1024)
                            logger.info(f"Downloading {dest_file.name}: {pct:.1f}% ({mb:.1f} MB)")

                dur = max(0.1, time.time() - start_time)
                mb_total = downloaded / (1024 * 1024)
                logger.info(f"Completed download of {dest_file.name} ({mb_total:.1f} MB in {dur:.1f}s, {mb_total/dur:.2f} MB/s)")
                return True
        except Exception as e:
            logger.warning(f"Download from {url} failed: {e}")
            if dest_file.exists():
                dest_file.unlink()

    return False

def prepare_real_images(
    raw_dir: Path,
    max_images: int = 2000,
    target_size: Tuple[int, int] = (256, 256),
    allow_synthetic: bool = False
) -> List[Path]:
    """
    Downloads Imagenette2-160 (or checks existing images in raw_dir),
    crops/resizes to 256x256 grayscale, and returns list of image file paths.
    """
    raw_images_dir = raw_dir / "images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted and prepared
    existing_files = sorted(list(raw_images_dir.glob("*.png")) + list(raw_images_dir.glob("*.jpg")) + list(raw_images_dir.glob("*.JPEG")))
    if len(existing_files) >= max_images:
        logger.info(f"Found {len(existing_files)} existing image covers in {raw_images_dir}. Reusing.")
        return existing_files[:max_images]

    logger.info(f"Downloading real image dataset (Imagenette2-160) to {raw_images_dir}...")
    temp_archive = raw_dir / "imagenette2-160.tgz"
    success = download_stream(IMAGENETTE_URLS, temp_archive)

    if not success:
        if not allow_synthetic:
            err_msg = (
                "CRITICAL: Failed to download real image dataset (Imagenette2-160) from mirrors, "
                "and no user-provided images exist in data/raw/images. "
                "VaultBreaker refuses to fall back to procedural synthetic covers without --allow-synthetic!"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        else:
            logger.warning("Falling back to procedural synthetic images due to --allow-synthetic flag.")
            return []

    # Extract directly and process to preserve disk space
    logger.info("Extracting, grayscale-converting, and center-cropping images to 256x256...")
    processed_paths = []
    H, W = target_size

    try:
        with tarfile.open(temp_archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
            logger.info(f"Found {len(members)} raw images in archive. Processing first {max_images}...")
            
            count = 0
            for m in members:
                if count >= max_images:
                    break
                f = tar.extractfile(m)
                if f is None:
                    continue
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Center crop and resize to target_size
                h_orig, w_orig = img.shape
                # Scale so smaller dimension matches target
                scale = max(H / h_orig, W / w_orig)
                new_w, new_h = int(round(w_orig * scale)), int(round(h_orig * scale))
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # Center crop
                start_y = (new_h - H) // 2
                start_x = (new_w - W) // 2
                img_crop = img_resized[start_y : start_y + H, start_x : start_x + W]

                out_path = raw_images_dir / f"cover_img_{count:04d}.png"
                cv2.imwrite(str(out_path), img_crop, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                processed_paths.append(out_path)
                count += 1

                if count % 250 == 0:
                    logger.info(f"  Processed {count}/{max_images} image covers...")

    finally:
        # Delete temporary .tgz immediately to conserve disk space
        if temp_archive.exists():
            temp_archive.unlink()
            logger.info("Removed temporary archive imagenette2-160.tgz to conserve disk space.")

    logger.info(f"Successfully prepared {len(processed_paths)} real cover images.")
    return processed_paths

def prepare_real_audio(
    raw_dir: Path,
    max_audio: int = 1000,
    target_sr: int = 16000,
    duration: float = 3.0,
    allow_synthetic: bool = False
) -> List[Path]:
    """
    Downloads ESC-50 dataset (or checks existing audio in raw_dir),
    converts to mono 16 kHz 16-bit WAV (3.0s clips), and returns file paths.
    """
    raw_audio_dir = raw_dir / "audio"
    raw_audio_dir.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(list(raw_audio_dir.glob("*.wav")))
    if len(existing_files) >= max_audio:
        logger.info(f"Found {len(existing_files)} existing audio covers in {raw_audio_dir}. Reusing.")
        return existing_files[:max_audio]

    logger.info(f"Downloading real audio dataset (ESC-50) to {raw_audio_dir}...")
    temp_archive = raw_dir / "esc50.zip"
    success = download_stream(ESC50_URLS, temp_archive)

    if not success:
        if not allow_synthetic:
            err_msg = (
                "CRITICAL: Failed to download real audio dataset (ESC-50) from mirrors, "
                "and no user-provided audio files exist in data/raw/audio. "
                "VaultBreaker refuses to fall back to procedural synthetic covers without --allow-synthetic!"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        else:
            logger.warning("Falling back to procedural synthetic audio due to --allow-synthetic flag.")
            return []

    logger.info("Extracting and standardizing audio clips to mono 16kHz 16-bit WAV (3.0s)...")
    processed_paths = []
    target_samples = int(target_sr * duration)

    try:
        with zipfile.ZipFile(temp_archive, "r") as z:
            wav_members = [name for name in z.namelist() if name.lower().endswith(".wav") and not name.startswith("__MACOSX")]
            logger.info(f"Found {len(wav_members)} audio clips in ESC-50. Processing up to {max_audio}...")

            count = 0
            for name in wav_members:
                if count >= max_audio:
                    break
                with z.open(name) as wav_file:
                    wav_bytes = io.BytesIO(wav_file.read())
                    try:
                        data, orig_sr = sf.read(wav_bytes)
                    except Exception:
                        continue

                    # Convert to mono if stereo
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)

                    # Resample to 16kHz
                    if orig_sr != target_sr:
                        data = librosa.resample(data.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)

                    # Slice to exact duration: find window with maximum energy
                    if len(data) >= target_samples:
                        hop = target_sr // 4
                        best_start = 0
                        best_energy = -1.0
                        for s in range(0, len(data) - target_samples + 1, hop):
                            energy = float(np.sum(data[s : s + target_samples] ** 2))
                            if energy > best_energy:
                                best_energy = energy
                                best_start = s
                        data = data[best_start : best_start + target_samples]
                    else:
                        pad_len = target_samples - len(data)
                        data = np.pad(data, (0, pad_len), mode="wrap")

                    # Skip clips that are digital silence even in their maximum-energy window
                    if np.max(np.abs(data)) < 1e-3:
                        continue

                    # Normalize amplitude and convert to 16-bit PCM
                    max_abs = np.max(np.abs(data)) + 1e-8
                    data = (data / max_abs) * 0.90
                    int_samples = (data * 32767.0).astype(np.int16)

                    out_path = raw_audio_dir / f"cover_aud_{count:04d}.wav"
                    sf.write(str(out_path), int_samples, target_sr, subtype="PCM_16", format="WAV")
                    processed_paths.append(out_path)
                    count += 1

                    if count % 200 == 0:
                        logger.info(f"  Processed {count}/{max_audio} audio covers...")

    finally:
        # Delete zip archive immediately
        if temp_archive.exists():
            temp_archive.unlink()
            logger.info("Removed temporary archive esc50.zip to conserve disk space.")

    logger.info(f"Successfully prepared {len(processed_paths)} real cover audio clips.")
    return processed_paths

def prepare_real_videos(
    raw_dir: Path,
    image_paths: List[Path],
    max_videos: int = 250,
    target_size: Tuple[int, int] = (128, 128),
    fps: int = 15,
    duration: float = 2.0,
    allow_synthetic: bool = False
) -> List[Path]:
    """
    Checks for user-provided .mp4/.avi/.mkv in data/raw/video.
    If none exist, creates realistic videos by smooth pan/zoom subwindow motion
    across REAL images (from Imagenette). Never uses procedural noise.
    Saved as lossless FFV1 MKV.
    """
    raw_video_dir = raw_dir / "video"
    raw_video_dir.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(
        list(raw_video_dir.glob("*.mkv")) +
        list(raw_video_dir.glob("*.mp4")) +
        list(raw_video_dir.glob("*.avi"))
    )
    if len(existing_files) >= max_videos:
        logger.info(f"Found {len(existing_files)} existing video covers in {raw_video_dir}. Reusing.")
        return existing_files[:max_videos]

    if not image_paths:
        if not allow_synthetic:
            err_msg = "Cannot generate realistic pan/zoom video: no real images available and --allow-synthetic not passed."
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        return []

    logger.info(f"Synthesizing {max_videos} realistic pan/zoom videos from real images in lossless FFV1 MKV...")
    processed_paths = []
    n_frames = int(fps * duration)
    H, W = target_size

    for idx in range(max_videos):
        # Pick a base real image
        img_p = image_paths[idx % len(image_paths)]
        base_img = cv2.imread(str(img_p), cv2.IMREAD_COLOR) # BGR
        if base_img is None:
            base_img = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)

        # Upscale canvas to 2x target size for ample pan/zoom space
        canvas_h, canvas_w = H * 2, W * 2
        canvas = cv2.resize(base_img, (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)

        rng = np.random.RandomState(42 + idx)
        max_dx = canvas_w - W
        max_dy = canvas_h - H
        start_x = rng.randint(0, max(1, max_dx // 2))
        start_y = rng.randint(0, max(1, max_dy // 2))
        end_x = rng.randint(max_dx // 2, max_dx)
        end_y = rng.randint(max_dy // 2, max_dy)

        frames = []
        for f in range(n_frames):
            alpha = f / max(1, n_frames - 1)
            cx = int(round(start_x + alpha * (end_x - start_x)))
            cy = int(round(start_y + alpha * (end_y - start_y)))
            cx = np.clip(cx, 0, max_dx)
            cy = np.clip(cy, 0, max_dy)
            frame = canvas[cy : cy + H, cx : cx + W]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        out_path = raw_video_dir / f"cover_vid_{idx:04d}.mkv"
        iio.imwrite(
            str(out_path),
            np.stack(frames, axis=0),
            extension=".mkv",
            codec="ffv1",
            fps=fps
        )
        processed_paths.append(out_path)

        if (idx + 1) % 50 == 0:
            logger.info(f"  Generated {idx + 1}/{max_videos} video covers...")

    logger.info(f"Successfully generated {len(processed_paths)} realistic video covers.")
    return processed_paths
