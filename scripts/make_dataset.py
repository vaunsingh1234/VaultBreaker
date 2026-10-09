#!/usr/bin/env python3
"""Dataset Generation Script for VaultBreaker.
Generates paired CLEAN and STEGO files for Image, Audio, and Video media from REAL covers,
enforces identical re-encoding pipelines, and produces source-disjoint manifests.
"""

import argparse
from pathlib import Path
import sys

from vaultbreaker.utils.config import load_config
from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.seed import seed_everything
from vaultbreaker.data.downloaders import get_ram_aware_dataset_sizes
from vaultbreaker.data.generator import (
    generate_image_dataset,
    generate_audio_dataset,
    generate_video_dataset
)
from vaultbreaker.data.split import split_dataset_by_source, save_splits
from vaultbreaker.data.integrity import assert_no_leakage, assert_no_duplicate_sha256_within_label

logger = get_logger("MakeDataset")

def main():
    parser = argparse.ArgumentParser(description="Generate steganography dataset from real covers for VaultBreaker.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--fast", action="store_true", help="Generate tiny dataset for fast demonstration / testing")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow fallback to procedural synthetic data if download fails")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    seed_everything(seed)

    paths_cfg = config.get("paths", {})
    raw_dir = Path(paths_cfg.get("raw_dir", "data/raw"))
    output_dir = Path(paths_cfg.get("generated_dir", "data/generated"))
    splits_dir = Path(paths_cfg.get("splits_dir", "data/splits"))

    ds_cfg = config.get("dataset", {})
    img_cfg = ds_cfg.get("image", {})
    aud_cfg = ds_cfg.get("audio", {})
    vid_cfg = ds_cfg.get("video", {})
    split_cfg = ds_cfg.get("split", {})

    # Compute RAM/disk-aware dataset sizes
    sizes = get_ram_aware_dataset_sizes(
        config_img=img_cfg.get("num_samples", 4000),
        config_aud=aud_cfg.get("num_samples", 2000),
        config_vid=vid_cfg.get("num_samples", 700),
        fast=args.fast
    )
    n_img, n_aud, n_vid = sizes["images"], sizes["audio"], sizes["video"]

    if args.fast:
        logger.info(f"=== RUNNING IN FAST MODE (Target Files: Images={n_img}, Audio={n_aud}, Video={n_vid}) ===")
    else:
        logger.info(f"=== RUNNING IN FULL SCALE MODE (Target Files: Images={n_img}, Audio={n_aud}, Video={n_vid}) ===")

    all_records = []

    # 1. Image Dataset
    logger.info(">>> STEP 1: GENERATING IMAGE DATASET (Real covers) <<<")
    img_records = generate_image_dataset(
        output_dir=output_dir,
        raw_dir=raw_dir,
        num_samples=n_img,
        target_size=tuple(img_cfg.get("target_size", [256, 256])),
        payload_rates=img_cfg.get("payload_rates", [0.10, 0.20, 0.30, 0.50]),
        methods=img_cfg.get("methods", ["lsb_replacement", "lsb_matching", "edge_adaptive", "dct_ac"]),
        allow_synthetic=args.allow_synthetic,
        seed=seed
    )
    all_records.extend(img_records)

    # Collect prepared image cover paths for realistic video pan/zoom
    raw_images_dir = raw_dir / "images"
    real_img_paths = sorted(list(raw_images_dir.glob("*.png")) + list(raw_images_dir.glob("*.jpg")))

    # 2. Audio Dataset
    logger.info(">>> STEP 2: GENERATING AUDIO DATASET (ESC-50 real covers) <<<")
    aud_records = generate_audio_dataset(
        output_dir=output_dir,
        raw_dir=raw_dir,
        num_samples=n_aud,
        sample_rate=aud_cfg.get("sample_rate", 16000),
        duration=aud_cfg.get("duration", 3.0),
        payload_rates=aud_cfg.get("payload_rates", [0.20, 0.40, 0.70, 1.00]),
        methods=aud_cfg.get("methods", ["lsb_replacement", "lsb_matching", "echo_hiding"]),
        allow_synthetic=args.allow_synthetic,
        seed=seed
    )
    all_records.extend(aud_records)

    # 3. Video Dataset
    logger.info(">>> STEP 3: GENERATING VIDEO DATASET (Pan/zoom across real image covers) <<<")
    vid_records = generate_video_dataset(
        output_dir=output_dir,
        raw_dir=raw_dir,
        num_samples=n_vid,
        image_paths=real_img_paths,
        target_size=tuple(vid_cfg.get("target_size", [128, 128])),
        fps=vid_cfg.get("fps", 15),
        duration=vid_cfg.get("duration", 2.0),
        payload_rates=vid_cfg.get("payload_rates", [0.10, 0.20, 0.40]),
        methods=vid_cfg.get("methods", ["frame_lsb", "frame_lsb_matching"]),
        allow_synthetic=args.allow_synthetic,
        seed=seed
    )
    all_records.extend(vid_records)

    logger.info(f"Total media files generated: {len(all_records)} (Images: {len(img_records)}, Audio: {len(aud_records)}, Video: {len(vid_records)})")

    # 4. Source-disjoint splitting
    logger.info(">>> STEP 4: SOURCE-DISJOINT PARTITIONING (70/15/15) <<<")
    df_manifest = split_dataset_by_source(
        records=all_records,
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
        seed=seed
    )

    # 5. Integrity verification
    assert_no_leakage(df_manifest)
    assert_no_duplicate_sha256_within_label(df_manifest)
    logger.info("VERIFICATION PASSED: No source ID leakage and no duplicate SHA-256 files detected across manifest.")

    # 6. Save manifests
    manifest_paths = save_splits(df_manifest, splits_dir)
    logger.info("Dataset manifests saved successfully:")
    for k, p in manifest_paths.items():
        sub = df_manifest[df_manifest["split"] == k] if k in ["train", "val", "test"] else df_manifest
        logger.info(f"  {k}: {len(sub)} samples ({p})")

    # Per-format split breakdown
    for split_name in ["train", "val", "test"]:
        s_df = df_manifest[df_manifest["split"] == split_name]
        logger.info(f"Split [{split_name}]: {len(s_df)} total files")
        for fmt in ["image", "audio", "video"]:
            f_count = len(s_df[s_df["media_type"] == fmt])
            c_count = len(s_df[(s_df["media_type"] == fmt) & (s_df["label"] == 0)])
            s_count = len(s_df[(s_df["media_type"] == fmt) & (s_df["label"] == 1)])
            logger.info(f"   - {fmt.capitalize()}: {f_count} (Clean: {c_count}, Stego: {s_count})")

if __name__ == "__main__":
    main()
