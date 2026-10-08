import os
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.io import compute_sha256, ensure_dir
from vaultbreaker.features.image_features import ImageFeatureExtractor
from vaultbreaker.features.audio_features import AudioFeatureExtractor
from vaultbreaker.features.video_features import VideoFeatureExtractor

logger = get_logger("FeaturePipeline")

def _extract_worker(task: Tuple[str, str, str, Path]) -> Tuple[str, np.ndarray, int]:
    """Worker function for multiprocessing extraction."""
    filepath_str, media_type, sha256, cache_dir = task
    cache_file = cache_dir / f"{sha256}.npz"
    
    if cache_file.exists():
        try:
            data = np.load(cache_file)
            return filepath_str, data["features"], 1
        except Exception:
            pass # re-extract on cache corruption

    # Fresh extraction
    p = Path(filepath_str)
    if media_type == "image":
        extractor = ImageFeatureExtractor()
        vec = extractor.extract_from_file(p)
    elif media_type == "audio":
        extractor = AudioFeatureExtractor()
        vec = extractor.extract_from_file(p)
    elif media_type == "video":
        extractor = VideoFeatureExtractor()
        vec = extractor.extract_from_file(p)
    else:
        raise ValueError(f"Unsupported media type: {media_type}")

    # Save to cache
    np.savez_compressed(cache_file, features=vec)
    return filepath_str, vec, 0

class MultiModalFeaturePipeline:
    """
    Unified feature pipeline that coordinates format extractors,
    disk caching by content hash, parallel processing, and
    leakage-free StandardScaler management.
    """
    def __init__(self, cache_dir: Union[str, Path] = "data/features", num_workers: int = 4):
        self.cache_dir = ensure_dir(cache_dir)
        self.num_workers = num_workers
        self.img_extractor = ImageFeatureExtractor()
        self.aud_extractor = AudioFeatureExtractor()
        self.vid_extractor = VideoFeatureExtractor()

    def extract_single(self, filepath: Union[str, Path], media_type: str) -> np.ndarray:
        """Extract features for a single file with disk cache check."""
        p = Path(filepath)
        sha256 = compute_sha256(p)
        cache_file = self.cache_dir / f"{sha256}.npz"
        
        if cache_file.exists():
            try:
                data = np.load(cache_file)
                return data["features"]
            except Exception:
                pass

        if media_type == "image":
            vec = self.img_extractor.extract_from_file(p)
        elif media_type == "audio":
            vec = self.aud_extractor.extract_from_file(p)
        elif media_type == "video":
            vec = self.vid_extractor.extract_from_file(p)
        else:
            raise ValueError(f"Unknown media type: {media_type}")

        np.savez_compressed(cache_file, features=vec)
        return vec

    def extract_dataset(
        self,
        manifest_df: pd.DataFrame,
        desc: str = "Extracting Features"
    ) -> Dict[str, np.ndarray]:
        """
        Extract features for all files in a manifest using multiprocessing.
        Returns dictionary mapping filepath -> feature vector.
        """
        tasks = []
        for _, row in manifest_df.iterrows():
            tasks.append((row["filepath"], row["media_type"], row["sha256"], self.cache_dir))

        results = {}
        cached_count = 0

        # Run extraction
        if self.num_workers <= 1:
            for task in tqdm(tasks, desc=desc):
                fp, vec, was_cached = _extract_worker(task)
                results[fp] = vec
                cached_count += was_cached
        else:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(_extract_worker, task) for task in tasks]
                for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    fp, vec, was_cached = f.result()
                    results[fp] = vec
                    cached_count += was_cached

        logger.info(f"Extracted {len(results)} items ({cached_count} from cache, {len(results) - cached_count} computed)")
        return results

    def fit_scalers(
        self,
        train_features: Dict[str, np.ndarray],
        train_manifest: pd.DataFrame
    ) -> Dict[str, StandardScaler]:
        """
        Fit StandardScalers STRICTLY on the training split features for each format.
        """
        scalers = {}
        for m_type in ["image", "audio", "video"]:
            sub = train_manifest[train_manifest["media_type"] == m_type]
            if sub.empty:
                continue
            feats_list = [train_features[fp] for fp in sub["filepath"]]
            X = np.stack(feats_list, axis=0)
            scaler = StandardScaler()
            scaler.fit(X)
            scalers[m_type] = scaler
            logger.info(f"Fit StandardScaler for {m_type} on {len(X)} training samples (mean shape: {scaler.mean_.shape})")
        return scalers

    def save_scalers(self, scalers: Dict[str, StandardScaler], save_dir: Union[str, Path]) -> None:
        """Save fitted scalers to disk."""
        s_dir = ensure_dir(save_dir)
        for m_type, sc in scalers.items():
            joblib.dump(sc, s_dir / f"scaler_{m_type}.joblib")
        logger.info(f"Saved format scalers to {s_dir}")

    def load_scalers(self, save_dir: Union[str, Path]) -> Dict[str, StandardScaler]:
        """Load fitted scalers from disk."""
        s_dir = Path(save_dir)
        scalers = {}
        for m_type in ["image", "audio", "video"]:
            sc_path = s_dir / f"scaler_{m_type}.joblib"
            if sc_path.exists():
                scalers[m_type] = joblib.load(sc_path)
        return scalers

    def transform_single(
        self,
        vec: np.ndarray,
        media_type: str,
        scalers: Dict[str, StandardScaler]
    ) -> np.ndarray:
        """Standardize a single vector with the format's training scaler."""
        if media_type in scalers:
            sc = scalers[media_type]
            norm_vec = sc.transform(vec.reshape(1, -1))[0]
            return np.nan_to_num(norm_vec, nan=0.0).astype(np.float32)
        return vec.astype(np.float32)
