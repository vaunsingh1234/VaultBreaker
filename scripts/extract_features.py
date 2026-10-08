#!/usr/bin/env python3
"""Feature Extraction CLI for VaultBreaker.
Extracts format-specific feature vectors with disk-caching and multiprocessing,
fits StandardScalers on training data only, and stores scaled matrices.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from vaultbreaker.utils.config import load_config
from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.io import ensure_dir
from vaultbreaker.features.extractor import MultiModalFeaturePipeline

logger = get_logger("ExtractFeatures")

def main():
    parser = argparse.ArgumentParser(description="Extract multi-modal steganography features.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    splits_dir = Path(paths_cfg.get("splits_dir", "data/splits"))
    features_dir = Path(paths_cfg.get("features_dir", "data/features"))
    models_dir = Path(paths_cfg.get("models_dir", "models_saved"))
    ensure_dir(features_dir)
    ensure_dir(models_dir)

    manifest_all_path = splits_dir / "manifest_all.csv"
    if not manifest_all_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_all_path}. Please run scripts/make_dataset.py first.")

    df_all = pd.read_csv(manifest_all_path)
    logger.info(f"Loaded manifest with {len(df_all)} items.")

    pipeline = MultiModalFeaturePipeline(cache_dir=features_dir / "cache", num_workers=args.num_workers)

    # 1. Extract all features
    logger.info("Extracting features across all items...")
    features_map = pipeline.extract_dataset(df_all)

    # 2. Fit Scalers STRICTLY on Train split
    df_train = df_all[df_all["split"] == "train"]
    scalers = pipeline.fit_scalers(features_map, df_train)
    pipeline.save_scalers(scalers, models_dir)

    # 3. Assemble and save split matrices
    for split_name in ["train", "val", "test"]:
        sub_df = df_all[df_all["split"] == split_name].reset_index(drop=True)
        if sub_df.empty:
            continue
            
        data_by_format = {}
        for m_type in ["image", "audio", "video"]:
            m_sub = sub_df[sub_df["media_type"] == m_type].reset_index(drop=True)
            if m_sub.empty:
                continue
            
            raw_X = np.stack([features_map[fp] for fp in m_sub["filepath"]], axis=0)
            norm_X = scalers[m_type].transform(raw_X)
            y = m_sub["label"].to_numpy(dtype=np.int64)
            methods = m_sub["method"].to_numpy()
            rates = m_sub["payload_rate"].to_numpy(dtype=np.float32)
            filepaths = m_sub["filepath"].to_numpy()

            data_by_format[f"X_{m_type}"] = norm_X
            data_by_format[f"y_{m_type}"] = y
            data_by_format[f"methods_{m_type}"] = methods
            data_by_format[f"rates_{m_type}"] = rates
            data_by_format[f"filepaths_{m_type}"] = filepaths

        split_matrix_path = features_dir / f"split_{split_name}.npz"
        np.savez_compressed(split_matrix_path, **data_by_format)
        logger.info(f"Saved {split_name} feature matrices to {split_matrix_path}")

    logger.info("Feature extraction pipeline completed successfully.")

if __name__ == "__main__":
    main()
