import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.io import ensure_dir

logger = get_logger("Splitter")

def split_dataset_by_source(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Splits records into Train/Val/Test strictly by SOURCE FILE.
    Every source file and its paired stego counterparts are assigned
    to the identical split, guaranteeing zero data leakage.
    """
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Cannot split empty records list.")

    unique_sources = df["source_id"].unique()
    rng = np.random.RandomState(seed)
    shuffled_sources = rng.permutation(unique_sources)

    n_total = len(shuffled_sources)
    n_train = int(np.round(n_total * train_ratio))
    n_val = int(np.round(n_total * val_ratio))
    
    train_sources = set(shuffled_sources[:n_train])
    val_sources = set(shuffled_sources[n_train : n_train + n_val])
    test_sources = set(shuffled_sources[n_train + n_val :])

    def assign_split(src: str) -> str:
        if src in train_sources:
            return "train"
        elif src in val_sources:
            return "val"
        else:
            return "test"

    df["split"] = df["source_id"].apply(assign_split)
    
    # Assert pairwise disjointness
    assert train_sources.isdisjoint(val_sources), "Train and Val sources overlap!"
    assert train_sources.isdisjoint(test_sources), "Train and Test sources overlap!"
    assert val_sources.isdisjoint(test_sources), "Val and Test sources overlap!"

    logger.info(
        f"Splits created: Train={len(train_sources)} sources ({len(df[df['split']=='train'])} files), "
        f"Val={len(val_sources)} sources ({len(df[df['split']=='val'])} files), "
        f"Test={len(test_sources)} sources ({len(df[df['split']=='test'])} files)"
    )
    return df

def save_splits(df: pd.DataFrame, splits_dir: Path) -> Dict[str, Path]:
    """Save split manifests to CSV files."""
    ensure_dir(splits_dir)
    manifest_path = splits_dir / "manifest_all.csv"
    df.to_csv(manifest_path, index=False)
    
    paths = {"all": manifest_path}
    for split_name in ["train", "val", "test"]:
        sub_df = df[df["split"] == split_name]
        split_path = splits_dir / f"manifest_{split_name}.csv"
        sub_df.to_csv(split_path, index=False)
        paths[split_name] = split_path
        
    for media_type in ["image", "audio", "video"]:
        m_df = df[df["media_type"] == media_type]
        m_path = splits_dir / f"manifest_{media_type}.csv"
        m_df.to_csv(m_path, index=False)
        paths[media_type] = m_path
        
    logger.info(f"Saved manifest files to {splits_dir}")
    return paths
