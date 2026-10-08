#!/usr/bin/env python3
"""Dataset Generation Script for VaultBreaker.
Generates paired CLEAN and STEGO files for Image, Audio, and Video media,
enforces identical re-encoding pipelines, and produces source-disjoint manifests.
"""

import argparse
from pathlib import Path
import sys

from vaultbreaker.utils.config import load_config
from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.seed import seed_everything
from vaultbreaker.data.generator import (
    generate_image_dataset,
    generate_audio_dataset,
    generate_video_dataset
)
from vaultbreaker.data.split import split_dataset_by_source, save_splits
from vaultbreaker.data.integrity import assert_no_leakage

logger = get_logger("MakeDataset")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic steganography dataset for VaultBreaker.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--fast", action="store_true", help="Generate tiny dataset for fast demonstration / testing")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    seed_everything(seed)

    paths_cfg = config.get("paths", {})
    output_dir = Path(paths_cfg.get("generated_dir", "data/generated"))
    splits_dir = Path(paths_cfg.get("splits_dir", "data/splits"))

    ds_cfg = config.get("dataset", {})
    img_cfg = ds_cfg.get("image", {})
    aud_cfg = ds_cfg.get("audio", {})
    vid_cfg = ds_cfg.get("video", {})
    split_cfg = ds_cfg.get("split", {})

    if args.fast:
        n_img = img_cfg.get("fast_num_samples", 40)
        n_aud = aud_cfg.get("fast_num_samples", 30)
        n_vid = vid_cfg.get("fast_num_samples", 16)
        logger.info(f"--- RUNNING IN FAST MODE (Images: {n_img}, Audio: {n_aud}, Video: {n_vid}) ---")
    else:
        n_img = img_cfg.get("num_samples", 1000)
        n_aud = aud_cfg.get("num_samples", 600)
        n_vid = vid_cfg.get("num_samples", 200)
        logger.info(f"--- RUNNING IN FULL MODE (Images: {n_img}, Audio: {n_aud}, Video: {n_vid}) ---")

    all_records = []

    # 1. Image Dataset
    img_records = generate_image_dataset(
        output_dir=output_dir,
        num_samples=n_img,
        target_size=tuple(img_cfg.get("target_size", [256, 256])),
        payload_rates=img_cfg.get("payload_rates", [0.05, 0.1, 0.2, 0.4]),
        methods=img_cfg.get("methods", ["lsb_replacement", "lsb_matching", "edge_adaptive", "dct_ac"]),
        seed=seed
    )
    all_records.extend(img_records)

    # 2. Audio Dataset
    aud_records = generate_audio_dataset(
        output_dir=output_dir,
        num_samples=n_aud,
        sample_rate=aud_cfg.get("sample_rate", 16000),
        duration=aud_cfg.get("duration", 3.0),
        payload_rates=aud_cfg.get("payload_rates", [0.05, 0.1, 0.3, 0.5]),
        methods=aud_cfg.get("methods", ["lsb_replacement", "lsb_matching", "echo_hiding"]),
        seed=seed
    )
    all_records.extend(aud_records)

    # 3. Video Dataset
    vid_records = generate_video_dataset(
        output_dir=output_dir,
        num_samples=n_vid,
        target_size=tuple(vid_cfg.get("target_size", [128, 128])),
        fps=vid_cfg.get("fps", 15),
        duration=vid_cfg.get("duration", 2.0),
        payload_rates=vid_cfg.get("payload_rates", [0.05, 0.1, 0.2]),
        methods=vid_cfg.get("methods", ["frame_lsb", "audio_lsb"]),
        seed=seed
    )
    all_records.extend(vid_records)

    logger.info(f"Total media files generated: {len(all_records)}")

    # 4. Source-disjoint splitting
    df_manifest = split_dataset_by_source(
        records=all_records,
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
        seed=seed
    )

    # 5. Integrity verification
    assert_no_leakage(df_manifest)
    logger.info("VERIFICATION PASSED: No source ID leakage detected across splits.")

    # 6. Save manifests
    manifest_paths = save_splits(df_manifest, splits_dir)
    logger.info("Dataset generation completed successfully.")
    for k, p in manifest_paths.items():
        logger.info(f"  {k}: {p}")

if __name__ == "__main__":
    main()
