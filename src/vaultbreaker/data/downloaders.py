import os
import shutil
import urllib.request
import tarfile
import zipfile
import io
import time
import hashlib
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import soundfile as sf
import librosa
import imageio.v3 as iio
import imageio_ffmpeg
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

# Global tracker for console table display
INGESTION_STATS = {
    "images": {"found": 0, "used": 0, "skipped": 0},
    "audio": {"found": 0, "used": 0, "skipped": 0},
    "video": {"found": 0, "used": 0, "skipped": 0}
}


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for deduplication."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


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

    import ssl
    try:
        import certifi
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ssl_ctx = ssl._create_unverified_context()

    for url in urls:
        logger.info(f"Connecting to download source: {url}")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "VaultBreaker-Forensics/2.0 (Mozilla/5.0)"}
            )
            with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as response:
                total_size = response.headers.get("Content-Length")
                total_bytes = int(total_size) if total_size else 0
                downloaded = 0
                block_size = 1024 * 1024  # 1MB chunks
                
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


# --- USER RAW FILE PROCESSORS (Images, Audio, Video) ---

def process_user_images(
    raw_images_dir: Path,
    target_size: Tuple[int, int] = (256, 256)
) -> Tuple[List[Path], Dict[str, int]]:
    """
    Scans user-provided images in data/raw/images (png/jpg/jpeg/bmp, case-insensitive),
    deduplicates by sha256, skips corrupt files with warning,
    crops/resizes to target_size (grayscale), and writes to standardized path.
    """
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    std_dir = raw_images_dir / ".standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    user_media_dirs = [raw_images_dir]
    user_media_root = Path("data/user_media")
    if user_media_root.exists():
        user_media_dirs.append(user_media_root)
        if (user_media_root / "images").exists():
            user_media_dirs.append(user_media_root / "images")

    candidates = []
    for d in user_media_dirs:
        if d.exists():
            candidates.extend([
                p for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in valid_exts and not p.name.startswith("cover_img_")
            ])
    candidates = sorted(list(set(candidates)))

    found = len(candidates)
    used = 0
    skipped = 0
    seen_hashes = set()
    processed_paths = []
    H, W = target_size

    for p in candidates:
        try:
            sha = compute_file_sha256(p)
            if sha in seen_hashes:
                logger.warning(f"Skipping duplicate image {p.name} (SHA-256 match)")
                skipped += 1
                continue
            seen_hashes.add(sha)

            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                from PIL import Image
                with Image.open(p) as pil_img:
                    img = np.array(pil_img.convert("L"))

            if img is None or img.size == 0:
                logger.warning(f"Skipping unreadable or corrupt image: {p.name}")
                skipped += 1
                continue

            h_orig, w_orig = img.shape
            scale = max(H / h_orig, W / w_orig)
            new_w, new_h = int(round(w_orig * scale)), int(round(h_orig * scale))
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            start_y = (new_h - H) // 2
            start_x = (new_w - W) // 2
            img_crop = img_resized[start_y : start_y + H, start_x : start_x + W]

            out_p = std_dir / f"user_img_{used:04d}_{sha[:8]}.png"
            cv2.imwrite(str(out_p), img_crop, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            processed_paths.append(out_p)
            used += 1
        except Exception as e:
            logger.warning(f"Error processing image {p.name}: {e}. Skipping.")
            skipped += 1

    stats = {"found": found, "used": used, "skipped": skipped}
    return processed_paths, stats


def process_user_audio(
    raw_audio_dir: Path,
    target_sr: int = 16000,
    duration: float = 3.0
) -> Tuple[List[Path], Dict[str, int]]:
    """
    Scans user-provided audio in data/raw/audio (wav/mp3/flac/m4a, case-insensitive),
    deduplicates by sha256, skips corrupt files with warning,
    converts to 16 kHz mono 16-bit WAV (3.0s window with max energy) using imageio-ffmpeg.
    """
    valid_exts = {".wav", ".mp3", ".flac", ".m4a"}
    raw_audio_dir.mkdir(parents=True, exist_ok=True)
    std_dir = raw_audio_dir / ".standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    user_media_dirs = [raw_audio_dir]
    user_media_root = Path("data/user_media")
    if user_media_root.exists():
        user_media_dirs.append(user_media_root)
        if (user_media_root / "audio").exists():
            user_media_dirs.append(user_media_root / "audio")

    candidates = []
    for d in user_media_dirs:
        if d.exists():
            candidates.extend([
                p for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in valid_exts and not p.name.startswith("cover_aud_")
            ])
    candidates = sorted(list(set(candidates)))

    found = len(candidates)
    used = 0
    skipped = 0
    seen_hashes = set()
    processed_paths = []
    target_samples = int(target_sr * duration)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    for p in candidates:
        try:
            sha = compute_file_sha256(p)
            if sha in seen_hashes:
                logger.warning(f"Skipping duplicate audio {p.name} (SHA-256 match)")
                skipped += 1
                continue
            seen_hashes.add(sha)

            # Convert to 16kHz mono WAV via imageio-ffmpeg
            temp_wav = std_dir / f"_temp_{sha[:8]}.wav"
            cmd = [
                ffmpeg_exe, "-y", "-v", "error",
                "-i", str(p),
                "-vn", "-acodec", "pcm_s16le", "-ar", str(target_sr), "-ac", "1",
                "-f", "wav", str(temp_wav)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            if res.returncode != 0 or not temp_wav.exists():
                logger.warning(f"Skipping unreadable or corrupt audio: {p.name} (ffmpeg decode failed)")
                skipped += 1
                if temp_wav.exists():
                    temp_wav.unlink()
                continue

            data, sr = sf.read(str(temp_wav))
            if temp_wav.exists():
                temp_wav.unlink()

            if data.ndim > 1:
                data = np.mean(data, axis=1)

            # Max energy window or circular pad
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

            if np.max(np.abs(data)) < 1e-3:
                logger.warning(f"Skipping audio {p.name}: digital silence")
                skipped += 1
                continue

            max_abs = np.max(np.abs(data)) + 1e-8
            data = (data / max_abs) * 0.90
            int_samples = (data * 32767.0).astype(np.int16)

            out_p = std_dir / f"user_aud_{used:04d}_{sha[:8]}.wav"
            sf.write(str(out_p), int_samples, target_sr, subtype="PCM_16", format="WAV")
            processed_paths.append(out_p)
            used += 1
        except Exception as e:
            logger.warning(f"Error processing audio {p.name}: {e}. Skipping.")
            skipped += 1

    stats = {"found": found, "used": used, "skipped": skipped}
    return processed_paths, stats


def process_user_videos(
    raw_video_dir: Path,
    target_size: Tuple[int, int] = (128, 128),
    fps: int = 15,
    duration: float = 2.0
) -> Tuple[List[Path], Dict[str, int]]:
    """
    Scans user-provided videos in data/raw/video (mp4/avi/mkv/mov/webm, case-insensitive),
    deduplicates by sha256, skips corrupt files with warning,
    decodes first 2s at 15 fps (30 frames), resizes to 128x128,
    and saves as lossless FFV1 MKV with pixelformat='rgb24'.
    """
    valid_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    raw_video_dir.mkdir(parents=True, exist_ok=True)
    std_dir = raw_video_dir / ".standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    user_media_dirs = [raw_video_dir]
    user_media_root = Path("data/user_media")
    if user_media_root.exists():
        user_media_dirs.append(user_media_root)
        if (user_media_root / "video").exists():
            user_media_dirs.append(user_media_root / "video")

    candidates = []
    for d in user_media_dirs:
        if d.exists():
            candidates.extend([
                p for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in valid_exts and not p.name.startswith("cover_vid_")
            ])
    candidates = sorted(list(set(candidates)))

    found = len(candidates)
    used = 0
    skipped = 0
    seen_hashes = set()
    processed_paths = []
    target_frames = int(fps * duration)
    H, W = target_size

    for p in candidates:
        try:
            sha = compute_file_sha256(p)
            if sha in seen_hashes:
                logger.warning(f"Skipping duplicate video {p.name} (SHA-256 match)")
                skipped += 1
                continue
            seen_hashes.add(sha)

            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                logger.warning(f"Skipping unreadable or corrupt video: {p.name}")
                skipped += 1
                continue

            frames = []
            while len(frames) < target_frames:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (W, H), interpolation=cv2.INTER_AREA)
                frames.append(frame_resized)
            cap.release()

            if not frames:
                logger.warning(f"Skipping video {p.name}: zero frames decoded")
                skipped += 1
                continue

            # Loop frames if video is shorter than target duration
            orig_len = len(frames)
            while len(frames) < target_frames:
                frames.append(frames[len(frames) % orig_len])
            frames = frames[:target_frames]

            out_p = std_dir / f"user_vid_{used:04d}_{sha[:8]}.mkv"
            iio.imwrite(
                str(out_p),
                np.stack(frames, axis=0),
                extension=".mkv",
                codec="ffv1",
                fps=fps,
                pixelformat="rgb24"
            )
            processed_paths.append(out_p)
            used += 1
        except Exception as e:
            logger.warning(f"Error processing video {p.name}: {e}. Skipping.")
            skipped += 1

    stats = {"found": found, "used": used, "skipped": skipped}
    return processed_paths, stats


def print_ingestion_summary_table():
    """Print clean ASCII summary table of user files found, used, and skipped."""
    print("\n" + "=" * 62)
    print("           VAULTBREAKER RAW DATA INGESTION SUMMARY")
    print("=" * 62)
    print(f"{'Format':<10} | {'Found':>7} | {'Used':>7} | {'Skipped':>9}")
    print("-" * 11 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 11)
    for fmt_name in ["images", "audio", "video"]:
        s = INGESTION_STATS.get(fmt_name, {"found": 0, "used": 0, "skipped": 0})
        label = fmt_name.capitalize()
        print(f"{label:<10} | {s.get('found', 0):>7} | {s.get('used', 0):>7} | {s.get('skipped', 0):>9}")
    print("=" * 62 + "\n")


# --- PUBLIC DATASET DOWNLOAD & TOP-UP PIPELINES ---

def prepare_real_images(
    raw_dir: Path,
    max_images: int = 2000,
    target_size: Tuple[int, int] = (256, 256),
    allow_synthetic: bool = False,
    own_only: bool = False
) -> List[Path]:
    """
    Ingests user images from data/raw/images first.
    If own_only=True: uses only user images.
    If own_only=False: tops up remaining covers from Imagenette2-160.
    """
    raw_images_dir = raw_dir / "images"
    user_paths, stats = process_user_images(raw_images_dir, target_size=target_size)
    INGESTION_STATS["images"] = stats

    if own_only:
        logger.info(f"[--own-only] Using {len(user_paths)} user-provided image covers (Imagenette download disabled).")
        return user_paths

    if len(user_paths) >= max_images:
        logger.info(f"Using {max_images} user-provided image covers (out of {len(user_paths)} available).")
        return user_paths[:max_images]

    # Check for previously downloaded and prepared public covers
    existing_covers = sorted([p for p in raw_images_dir.glob("cover_img_*.png") if p.is_file()])
    needed = max_images - len(user_paths)

    if len(existing_covers) >= needed:
        logger.info(f"Reusing {needed} existing prepared covers + {len(user_paths)} user images.")
        return user_paths + existing_covers[:needed]

    logger.info(f"User provided {len(user_paths)} images. Downloading Imagenette to top up remaining {needed} covers...")
    temp_archive = raw_dir / "imagenette2-160.tgz"
    success = download_stream(IMAGENETTE_URLS, temp_archive)

    if not success:
        if not allow_synthetic and len(user_paths) == 0:
            err_msg = (
                "CRITICAL: Failed to download real image dataset (Imagenette2-160) from mirrors, "
                "and no user-provided images exist in data/raw/images."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        elif len(user_paths) > 0:
            logger.warning(f"Download failed; proceeding with {len(user_paths)} user images.")
            return user_paths
        else:
            return []

    processed_topup = []
    H, W = target_size
    try:
        with tarfile.open(temp_archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = 0
            for m in members:
                if count >= needed:
                    break
                f = tar.extractfile(m)
                if f is None:
                    continue
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                h_orig, w_orig = img.shape
                scale = max(H / h_orig, W / w_orig)
                new_w, new_h = int(round(w_orig * scale)), int(round(h_orig * scale))
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                start_y = (new_h - H) // 2
                start_x = (new_w - W) // 2
                img_crop = img_resized[start_y : start_y + H, start_x : start_x + W]

                out_path = raw_images_dir / f"cover_img_{count:04d}.png"
                cv2.imwrite(str(out_path), img_crop, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                processed_topup.append(out_path)
                count += 1
    finally:
        if temp_archive.exists():
            temp_archive.unlink()

    return user_paths + processed_topup


def prepare_real_audio(
    raw_dir: Path,
    max_audio: int = 1000,
    target_sr: int = 16000,
    duration: float = 3.0,
    allow_synthetic: bool = False,
    own_only: bool = False
) -> List[Path]:
    """
    Ingests user audio from data/raw/audio first.
    If own_only=True: uses only user audio.
    If own_only=False: tops up remaining covers from ESC-50 dataset.
    """
    raw_audio_dir = raw_dir / "audio"
    user_paths, stats = process_user_audio(raw_audio_dir, target_sr=target_sr, duration=duration)
    INGESTION_STATS["audio"] = stats

    if own_only:
        logger.info(f"[--own-only] Using {len(user_paths)} user-provided audio covers (ESC-50 download disabled).")
        return user_paths

    if len(user_paths) >= max_audio:
        logger.info(f"Using {max_audio} user-provided audio covers (out of {len(user_paths)} available).")
        return user_paths[:max_audio]

    existing_covers = sorted([p for p in raw_audio_dir.glob("cover_aud_*.wav") if p.is_file()])
    needed = max_audio - len(user_paths)

    if len(existing_covers) >= needed:
        logger.info(f"Reusing {needed} existing prepared audio covers + {len(user_paths)} user audio.")
        return user_paths + existing_covers[:needed]

    logger.info(f"User provided {len(user_paths)} audio clips. Downloading ESC-50 to top up remaining {needed} covers...")
    temp_archive = raw_dir / "esc50.zip"
    success = download_stream(ESC50_URLS, temp_archive)

    if not success:
        if not allow_synthetic and len(user_paths) == 0:
            err_msg = "CRITICAL: Failed to download ESC-50 and no user audio exists in data/raw/audio."
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        elif len(user_paths) > 0:
            logger.warning(f"Download failed; proceeding with {len(user_paths)} user audio.")
            return user_paths
        else:
            return []

    processed_topup = []
    target_samples = int(target_sr * duration)
    try:
        with zipfile.ZipFile(temp_archive, "r") as z:
            wav_members = [name for name in z.namelist() if name.lower().endswith(".wav") and not name.startswith("__MACOSX")]
            count = 0
            for name in wav_members:
                if count >= needed:
                    break
                with z.open(name) as wav_file:
                    wav_bytes = io.BytesIO(wav_file.read())
                    try:
                        data, orig_sr = sf.read(wav_bytes)
                    except Exception:
                        continue

                    if data.ndim > 1:
                        data = np.mean(data, axis=1)

                    if orig_sr != target_sr:
                        data = librosa.resample(data.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)

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

                    if np.max(np.abs(data)) < 1e-3:
                        continue

                    max_abs = np.max(np.abs(data)) + 1e-8
                    data = (data / max_abs) * 0.90
                    int_samples = (data * 32767.0).astype(np.int16)

                    out_path = raw_audio_dir / f"cover_aud_{count:04d}.wav"
                    sf.write(str(out_path), int_samples, target_sr, subtype="PCM_16", format="WAV")
                    processed_topup.append(out_path)
                    count += 1
    finally:
        if temp_archive.exists():
            temp_archive.unlink()

    return user_paths + processed_topup


def prepare_real_videos(
    raw_dir: Path,
    image_paths: List[Path],
    max_videos: int = 250,
    target_size: Tuple[int, int] = (128, 128),
    fps: int = 15,
    duration: float = 2.0,
    allow_synthetic: bool = False,
    own_only: bool = False
) -> List[Path]:
    """
    Ingests user videos from data/raw/video first.
    If own_only=True: uses only user videos.
    If own_only=False: tops up remaining covers by pan/zoom across image covers.
    """
    raw_video_dir = raw_dir / "video"
    user_paths, stats = process_user_videos(raw_video_dir, target_size=target_size, fps=fps, duration=duration)
    INGESTION_STATS["video"] = stats

    if own_only:
        logger.info(f"[--own-only] Using {len(user_paths)} user-provided video covers.")
        return user_paths

    if len(user_paths) >= max_videos:
        logger.info(f"Using {max_videos} user-provided video covers (out of {len(user_paths)} available).")
        return user_paths[:max_videos]

    existing_covers = sorted([p for p in raw_video_dir.glob("cover_vid_*.mkv") if p.is_file()])
    needed = max_videos - len(user_paths)

    if len(existing_covers) >= needed:
        logger.info(f"Reusing {needed} existing prepared video covers + {len(user_paths)} user videos.")
        return user_paths + existing_covers[:needed]

    if not image_paths:
        if not allow_synthetic and len(user_paths) == 0:
            err_msg = "Cannot generate realistic pan/zoom video: no real images available and --allow-synthetic not passed."
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        return user_paths

    logger.info(f"User provided {len(user_paths)} videos. Synthesizing remaining {needed} pan/zoom videos from real images...")
    processed_topup = []
    n_frames = int(fps * duration)
    H, W = target_size

    for idx in range(needed):
        img_p = image_paths[idx % len(image_paths)]
        base_img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if base_img is None:
            base_img = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)

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
            fps=fps,
            pixelformat="rgb24"
        )
        processed_topup.append(out_path)

    return user_paths + processed_topup
