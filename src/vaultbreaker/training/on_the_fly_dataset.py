"""On-The-Fly Paired Stego Training Dataset.

Embeds steganographic messages into real cover media dynamically during data loading.
Pairs each clean cover with its own freshly embedded stego counterpart in the same batch,
applying strictly stego-safe data augmentations (rot90, flips, circular time shifts, frame offsets).
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import cv2
import soundfile as sf
import imageio.v3 as iio

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
from vaultbreaker.models.unified_net import METHOD_TO_IDX


class OnTheFlyPairedStegoDataset(Dataset):
    """
    On-The-Fly Paired Stego Dataset for joint training.
    Every index produces TWO samples: (Clean Cover, Stego Variant).
    """
    def __init__(
        self,
        cover_records: List[Dict[str, Any]],
        image_methods: Optional[List[str]] = None,
        audio_methods: Optional[List[str]] = None,
        video_methods: Optional[List[str]] = None,
        curriculum_epoch: int = 0,
        high_payload_epochs: int = 2,
        augment: bool = True
    ):
        self.covers = cover_records
        self.image_methods = image_methods or ["lsb_replacement", "lsb_matching", "edge_adaptive", "dct_ac"]
        self.audio_methods = audio_methods or ["lsb_replacement", "lsb_matching", "echo_hiding"]
        self.video_methods = video_methods or ["frame_lsb", "frame_lsb_matching"]
        self.curriculum_epoch = curriculum_epoch
        self.high_payload_epochs = high_payload_epochs
        self.augment = augment

    def set_epoch(self, epoch: int):
        self.curriculum_epoch = epoch

    def __len__(self) -> int:
        return len(self.covers)

    def _sample_payload_rate(self, media_type: str) -> float:
        """Sample payload rate based on curriculum schedule."""
        if self.curriculum_epoch < self.high_payload_epochs:
            # High payloads during initial warm-up epochs
            if media_type == "audio":
                return float(np.random.choice([0.40, 0.70, 1.00]))
            else:
                return float(np.random.choice([0.30, 0.40, 0.50]))
        else:
            # Broad spectrum curriculum
            if media_type == "audio":
                return float(np.random.choice([0.10, 0.20, 0.40, 0.70, 1.00]))
            else:
                return float(np.random.choice([0.05, 0.10, 0.20, 0.30, 0.40, 0.50]))

    def _augment_image(self, clean: np.ndarray, stego: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply identical stego-safe geometric transforms to both clean and stego images."""
        if not self.augment:
            return clean, stego

        # 1. Random 90-degree rotations (preserves discrete grid relationships)
        k = np.random.randint(0, 4)
        if k > 0:
            clean = np.rot90(clean, k)
            stego = np.rot90(stego, k)

        # 2. Random horizontal flip
        if np.random.rand() > 0.5:
            clean = np.fliplr(clean)
            stego = np.fliplr(stego)

        # 3. Random vertical flip
        if np.random.rand() > 0.5:
            clean = np.flipud(clean)
            stego = np.flipud(stego)

        return clean.copy(), stego.copy()

    def _augment_audio(self, clean: np.ndarray, stego: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply circular time shift (no pitch, gain, or resampling)."""
        if not self.augment:
            return clean, stego

        # Shift up to +/- 4000 samples (0.25 seconds)
        shift = np.random.randint(-4000, 4000)
        clean = np.roll(clean, shift)
        stego = np.roll(stego, shift)
        return clean.copy(), stego.copy()

    def _augment_video(self, clean: np.ndarray, stego: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply identical spatial flips and rot90 across all video frames."""
        if not self.augment:
            return clean, stego

        k = np.random.randint(0, 4)
        do_hflip = (np.random.rand() > 0.5)

        if k > 0:
            clean = np.array([np.rot90(f, k) for f in clean])
            stego = np.array([np.rot90(f, k) for f in stego])

        if do_hflip:
            clean = np.array([np.fliplr(f) for f in clean])
            stego = np.array([np.fliplr(f) for f in stego])

        return clean.copy(), stego.copy()

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        record = self.covers[idx]
        media_type = record["media_type"]
        filepath = Path(record["filepath"])
        source_id = record.get("source_id", filepath.stem)

        # -------------------------------------------------------------
        # IMAGE MODALITY
        # -------------------------------------------------------------
        if media_type == "image":
            clean_img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            if clean_img is None or clean_img.shape != (256, 256):
                clean_img = np.zeros((256, 256), dtype=np.uint8)

            method = random.choice(self.image_methods)
            rate = self._sample_payload_rate("image")
            stego_seed = random.randint(1, 1000000)
            H, W = clean_img.shape
            p_rng = np.random.RandomState(stego_seed)

            try:
                if method == "dct_ac":
                    n_bits = max(32, int((H // 8) * (W // 8) * 15 * rate))
                    payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
                    stego_img, _ = embed_image_dct(clean_img, payload, seed=stego_seed)
                elif method == "edge_adaptive":
                    n_bits = max(32, int(H * W * rate))
                    payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
                    stego_img, _ = embed_image_edge_adaptive(clean_img, payload, seed=stego_seed)
                elif method == "lsb_matching":
                    n_bits = max(32, int(H * W * rate))
                    payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
                    stego_img, _ = embed_image_lsb_matching(clean_img, payload, seed=stego_seed)
                else: # lsb_replacement
                    n_bits = max(32, int(H * W * rate))
                    payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
                    stego_img, _ = embed_image_lsb_replacement(clean_img, payload, seed=stego_seed)
            except Exception:
                stego_img = clean_img.copy()

            clean_img, stego_img = self._augment_image(clean_img, stego_img)

            # Format to tensor
            clean_tensor = torch.from_numpy(clean_img).unsqueeze(0).float() # (1, 256, 256)
            stego_tensor = torch.from_numpy(stego_img).unsqueeze(0).float() # (1, 256, 256)

            m_idx = METHOD_TO_IDX.get(method, 1)

            clean_item = {
                "x": clean_tensor,
                "y": torch.tensor(0.0, dtype=torch.float32),
                "method_idx": torch.tensor(0, dtype=torch.long),
                "rate": torch.tensor(0.0, dtype=torch.float32),
                "media_type": "image",
                "source_id": source_id
            }
            stego_item = {
                "x": stego_tensor,
                "y": torch.tensor(1.0, dtype=torch.float32),
                "method_idx": torch.tensor(m_idx, dtype=torch.long),
                "rate": torch.tensor(rate, dtype=torch.float32),
                "media_type": "image",
                "source_id": source_id
            }
            return clean_item, stego_item

        # -------------------------------------------------------------
        # AUDIO MODALITY
        # -------------------------------------------------------------
        elif media_type == "audio":
            try:
                clean_samples, sr = sf.read(str(filepath), dtype="int16")
                if len(clean_samples) < 48000:
                    clean_samples = np.pad(clean_samples, (0, 48000 - len(clean_samples)), mode="wrap")
                elif len(clean_samples) > 48000:
                    clean_samples = clean_samples[:48000]
            except Exception:
                clean_samples = np.zeros(48000, dtype=np.int16)

            method = random.choice(self.audio_methods)
            rate = self._sample_payload_rate("audio")
            stego_seed = random.randint(1, 1000000)
            p_rng = np.random.RandomState(stego_seed)

            try:
                if method == "echo_hiding":
                    n_bits = max(8, int(len(clean_samples) / 2048 * min(rate, 0.8)))
                    payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
                    stego_samples, _ = embed_audio_echo(clean_samples, payload)
                elif method == "lsb_matching":
                    n_bits = max(32, int(len(clean_samples) * min(rate, 1.0)))
                    payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
                    stego_samples, _ = embed_audio_lsb_matching(clean_samples, payload, seed=stego_seed)
                else: # lsb_replacement
                    n_bits = max(32, int(len(clean_samples) * min(rate, 1.0)))
                    payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
                    stego_samples, _ = embed_audio_lsb_replacement(clean_samples, payload, seed=stego_seed)
            except Exception:
                stego_samples = clean_samples.copy()

            clean_samples, stego_samples = self._augment_audio(clean_samples, stego_samples)

            # Scale strictly by 1/32768.0 without loudness normalization
            clean_tensor = torch.from_numpy(clean_samples.astype(np.float32) / 32768.0).unsqueeze(0) # (1, 48000)
            stego_tensor = torch.from_numpy(stego_samples.astype(np.float32) / 32768.0).unsqueeze(0) # (1, 48000)

            m_idx = METHOD_TO_IDX.get(method, 1)

            clean_item = {
                "x": clean_tensor,
                "y": torch.tensor(0.0, dtype=torch.float32),
                "method_idx": torch.tensor(0, dtype=torch.long),
                "rate": torch.tensor(0.0, dtype=torch.float32),
                "media_type": "audio",
                "source_id": source_id
            }
            stego_item = {
                "x": stego_tensor,
                "y": torch.tensor(1.0, dtype=torch.float32),
                "method_idx": torch.tensor(m_idx, dtype=torch.long),
                "rate": torch.tensor(rate, dtype=torch.float32),
                "media_type": "audio",
                "source_id": source_id
            }
            return clean_item, stego_item

        # -------------------------------------------------------------
        # VIDEO MODALITY
        # -------------------------------------------------------------
        else:
            try:
                raw_frames = iio.imread(str(filepath)) # (N, H, W, 3) or (N, H, W)
                if raw_frames.ndim == 4:
                    # Convert to grayscale (N, H, W)
                    raw_frames = np.dot(raw_frames[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                if len(raw_frames) < 8:
                    pad = np.zeros((8 - len(raw_frames), raw_frames.shape[1], raw_frames.shape[2]), dtype=np.uint8)
                    raw_frames = np.concatenate([raw_frames, pad], axis=0)
                if raw_frames.shape[1:3] != (128, 128):
                    raw_frames = np.array([cv2.resize(f, (128, 128)) for f in raw_frames])
                clean_frames = raw_frames[:8]
            except Exception:
                clean_frames = np.zeros((8, 128, 128), dtype=np.uint8)

            method = random.choice(self.video_methods)
            rate = self._sample_payload_rate("video")
            stego_seed = random.randint(1, 1000000)
            p_rng = np.random.RandomState(stego_seed)

            try:
                total_pix = clean_frames.size
                n_bits = max(32, int(total_pix * rate))
                payload = p_rng.randint(0, 2, size=n_bits, dtype=np.uint8)
                if method == "frame_lsb_matching":
                    stego_frames, _ = embed_video_frame_lsb_matching(clean_frames, payload, seed=stego_seed)
                else:
                    stego_frames, _ = embed_video_frame_lsb(clean_frames, payload, seed=stego_seed)
                stego_frames = np.asarray(stego_frames, dtype=np.uint8)
            except Exception:
                stego_frames = clean_frames.copy()

            clean_frames = np.asarray(clean_frames, dtype=np.uint8)
            stego_frames = np.asarray(stego_frames, dtype=np.uint8)

            clean_frames, stego_frames = self._augment_video(clean_frames, stego_frames)

            clean_tensor = torch.from_numpy(clean_frames).unsqueeze(1).float()  # (8, 1, 128, 128)
            stego_tensor = torch.from_numpy(stego_frames).unsqueeze(1).float()  # (8, 1, 128, 128)

            m_idx = METHOD_TO_IDX.get(method, 1)

            clean_item = {
                "x": clean_tensor,
                "y": torch.tensor(0.0, dtype=torch.float32),
                "method_idx": torch.tensor(0, dtype=torch.long),
                "rate": torch.tensor(0.0, dtype=torch.float32),
                "media_type": "video",
                "source_id": source_id
            }
            stego_item = {
                "x": stego_tensor,
                "y": torch.tensor(1.0, dtype=torch.float32),
                "method_idx": torch.tensor(m_idx, dtype=torch.long),
                "rate": torch.tensor(rate, dtype=torch.float32),
                "media_type": "video",
                "source_id": source_id
            }
            return clean_item, stego_item


def collate_paired_batch(batch: List[Tuple[Dict, Dict]]) -> List[Dict]:
    """
    Unpacks (clean, stego) tuples so that every pair resides together in the same mini-batch.
    """
    flat = []
    for clean_item, stego_item in batch:
        flat.append(clean_item)
        flat.append(stego_item)
    return flat
