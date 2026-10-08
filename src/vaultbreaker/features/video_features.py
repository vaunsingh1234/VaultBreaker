import numpy as np
import cv2
import imageio.v3 as iio
from scipy.stats import skew, kurtosis
from scipy.signal import convolve2d
from pathlib import Path
from typing import Union, List, Optional

from vaultbreaker.features.image_features import ImageFeatureExtractor, KV_KERNEL_5X5, LAPLACIAN_KERNEL_3X3
from vaultbreaker.features.audio_features import AudioFeatureExtractor

class VideoFeatureExtractor:
    """
    Extracts a fixed 64-dimensional feature vector for steganography detection
    from video media (spatio-temporal frame aggregation, inter-frame residual dynamics,
    LSB temporal flicker, demuxed audio channel metrics, and container metadata).
    """
    FEATURE_DIM = 64

    def __init__(self, sample_frames: int = 8):
        self.sample_frames = sample_frames
        self.img_extractor = ImageFeatureExtractor()
        self.aud_extractor = AudioFeatureExtractor()

    def extract_from_file(self, file_path: Union[str, Path]) -> np.ndarray:
        """Read video file and extract 64-dim feature vector."""
        path_str = str(file_path)
        # Read frames using imageio
        try:
            frames = iio.imread(path_str, plugin="pyav")
        except Exception:
            try:
                frames = iio.imread(path_str)
            except Exception:
                # Fallback to OpenCV VideoCapture
                cap = cv2.VideoCapture(path_str)
                frame_list = []
                while True:
                    ret, f = cap.read()
                    if not ret:
                        break
                    frame_list.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                cap.release()
                if not frame_list:
                    raise ValueError(f"Could not read video frames from {path_str}")
                frames = np.stack(frame_list, axis=0)

        # File size and metadata
        file_size = Path(path_str).stat().st_size
        return self.extract(frames, file_size=file_size)

    def extract(
        self,
        frames: Union[np.ndarray, List[np.ndarray]],
        audio_samples: Optional[np.ndarray] = None,
        fps: float = 15.0,
        file_size: int = 100000
    ) -> np.ndarray:
        """Extract fixed 64-dimensional feature vector."""
        if isinstance(frames, list):
            frames = np.stack(frames, axis=0)

        n_total_frames = frames.shape[0]
        H, W = frames.shape[1], frames.shape[2]

        # Uniform frame sampling
        k = min(self.sample_frames, n_total_frames)
        sample_indices = np.linspace(0, n_total_frames - 1, k, dtype=int)
        sampled_frames = [frames[i] for i in sample_indices]

        # Convert sampled frames to grayscale
        gray_frames = []
        for f in sampled_frames:
            if f.ndim == 3:
                g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.shape[2] == 3 else f[:, :, 0]
            else:
                g = f
            gray_frames.append(g.astype(np.float32))

        feats = []

        # 1. Spatial Frame Aggregation (24 dims)
        # Core image stego metrics for each frame:
        # [lsb_mean, lsb_trans, lsb_entropy, chi_stat, kv_var, spam_diag_entropy]
        frame_metrics = []
        for g in gray_frames:
            g_u8 = np.clip(g, 0, 255).astype(np.uint8)
            lsb = (g_u8 & 1).astype(np.float32)
            m_lsb = float(np.mean(lsb))
            h_trans = float(np.mean(np.abs(np.diff(lsb, axis=1)))) if lsb.shape[1] > 1 else 0.5
            p1 = np.clip(m_lsb, 1e-9, 1.0 - 1e-9)
            ent_lsb = float(- (p1 * np.log2(p1) + (1.0 - p1) * np.log2(1.0 - p1)))
            
            # Simple Chi-square
            hist, _ = np.histogram(g_u8, bins=256, range=(0, 256))
            even = hist[0::2]
            odd = hist[1::2]
            chi = float(np.sum((even - odd) ** 2 / np.maximum((even + odd) / 2.0, 1.0)))
            
            # KV filter residual
            res_kv = convolve2d(g, KV_KERNEL_5X5, mode="same", boundary="symm")
            kv_var = float(np.var(res_kv))
            
            # Lap variance
            res_lap = convolve2d(g, LAPLACIAN_KERNEL_3X3, mode="same", boundary="symm")
            lap_var = float(np.var(res_lap))
            
            frame_metrics.append([m_lsb, h_trans, ent_lsb, chi, kv_var, lap_var])

        frame_metrics = np.array(frame_metrics, dtype=np.float32) # (K, 6)
        mean_spatial = np.mean(frame_metrics, axis=0) # 6
        std_spatial = np.std(frame_metrics, axis=0)   # 6
        max_spatial = np.max(frame_metrics, axis=0)   # 6
        p90_spatial = np.percentile(frame_metrics, 90, axis=0) - np.percentile(frame_metrics, 10, axis=0) # 6

        feats.extend(mean_spatial.tolist())
        feats.extend(std_spatial.tolist())
        feats.extend(max_spatial.tolist())
        feats.extend(p90_spatial.tolist()) # 24 dims

        # 2. Temporal Frame-Difference & Motion Residuals (16 dims)
        if len(gray_frames) > 1:
            diffs = [gray_frames[i+1] - gray_frames[i] for i in range(len(gray_frames) - 1)]
            all_diffs = np.concatenate([d.ravel() for d in diffs])
            
            mad_diff = float(np.mean(np.abs(all_diffs)))
            var_diff = float(np.var(all_diffs))
            sk_diff = float(skew(all_diffs))
            kt_diff = float(kurtosis(all_diffs))
            
            # Noise-residual inter-frame correlation
            noise_resids = [convolve2d(g, LAPLACIAN_KERNEL_3X3, mode="same") for g in gray_frames]
            corrs = []
            for i in range(len(noise_resids) - 1):
                c = np.corrcoef(noise_resids[i].ravel(), noise_resids[i+1].ravel())[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
            corr_mean = float(np.mean(corrs)) if corrs else 0.0
            corr_std = float(np.std(corrs)) if corrs else 0.0
            
            # Temporal LSB flicker
            lsb_planes = [(np.clip(g, 0, 255).astype(np.uint8) & 1).astype(np.float32) for g in gray_frames]
            flicker_rates = [np.mean(np.abs(lsb_planes[i+1] - lsb_planes[i])) for i in range(len(lsb_planes) - 1)]
            flicker_mean = float(np.mean(flicker_rates))
            flicker_std = float(np.std(flicker_rates))
            flicker_max = float(np.max(flicker_rates))
            
            # 2nd-order temporal acceleration
            if len(gray_frames) > 2:
                accels = [gray_frames[i+2] - 2 * gray_frames[i+1] + gray_frames[i] for i in range(len(gray_frames) - 2)]
                all_accels = np.concatenate([a.ravel() for a in accels])
                accel_var = float(np.var(all_accels))
                accel_kt = float(kurtosis(all_accels))
            else:
                accel_var, accel_kt = 0.0, 0.0
                
            temp_range = float(np.max(all_diffs) - np.min(all_diffs))
            temp_dyn_var = float(np.var([np.var(d) for d in diffs]))
            
            # Pixel trajectory entropy
            traj = np.stack(gray_frames, axis=0) # (K, H, W)
            traj_diff = np.diff(traj, axis=0)
            traj_sign = np.sign(traj_diff)
            traj_flips = float(np.mean(traj_sign[1:] != traj_sign[:-1])) if traj_sign.shape[0] > 1 else 0.0
            traj_std_mean = float(np.mean(np.std(traj, axis=0)))
            traj_max_mean = float(np.mean(np.max(traj, axis=0) - np.min(traj, axis=0)))
        else:
            mad_diff, var_diff, sk_diff, kt_diff = 0.0, 0.0, 0.0, 0.0
            corr_mean, corr_std = 0.0, 0.0
            flicker_mean, flicker_std, flicker_max = 0.0, 0.0, 0.0
            accel_var, accel_kt = 0.0, 0.0
            temp_range, temp_dyn_var = 0.0, 0.0
            traj_flips, traj_std_mean, traj_max_mean = 0.0, 0.0, 0.0

        feats.extend([
            mad_diff, var_diff, sk_diff, kt_diff,
            corr_mean, corr_std,
            flicker_mean, flicker_std, flicker_max,
            accel_var, accel_kt,
            temp_range, temp_dyn_var,
            traj_flips, traj_std_mean, traj_max_mean
        ]) # 16 dims

        # 3. Audio Track Metrics (16 dims)
        if audio_samples is not None and len(audio_samples) > 0:
            full_aud = self.aud_extractor.extract(audio_samples)
            # Pick 16 key audio components:
            # LSB metrics (0..3), Chi metrics (6, 7), LPC metrics (26, 27), Spectral (16..19), MFCC summary (40..43)
            selected_aud = [
                full_aud[0], full_aud[1], full_aud[2], full_aud[3],
                full_aud[6], full_aud[7],
                full_aud[26], full_aud[27],
                full_aud[16], full_aud[17], full_aud[18], full_aud[19],
                full_aud[40], full_aud[41], full_aud[42], full_aud[43]
            ]
        else:
            selected_aud = [0.0] * 16

        feats.extend([float(x) for x in selected_aud]) # 16 dims

        # 4. Container & Codec Metadata (8 dims)
        duration_est = float(n_total_frames) / max(1.0, float(fps))
        aspect_ratio = float(W) / max(1.0, float(H))
        bytes_per_pixel = float(file_size) / max(1.0, float(n_total_frames * H * W))
        is_lossless = 1.0 # Default lossless container
        w_norm = float(W) / 1000.0
        h_norm = float(H) / 1000.0

        feats.extend([
            float(fps), float(n_total_frames), duration_est, aspect_ratio,
            bytes_per_pixel, is_lossless, w_norm, h_norm
        ]) # 8 dims

        # Final check & sanitize
        vec = np.array(feats, dtype=np.float32)
        assert len(vec) == self.FEATURE_DIM, f"Extracted {len(vec)} video features, expected {self.FEATURE_DIM}"
        vec = np.nan_to_num(vec, nan=0.0, posinf=1e4, neginf=-1e4)
        return vec
