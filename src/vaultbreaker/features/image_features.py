import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from scipy.signal import convolve2d
from scipy.fftpack import dct
from pathlib import Path
from typing import Union, Optional

# KV 5x5 High-Pass Kernel from Spatial Rich Models (SRM)
KV_KERNEL_5X5 = np.array([
    [-1,  2,  -2,  2, -1],
    [ 2, -6,   8, -6,  2],
    [-2,  8, -12,  8, -2],
    [ 2, -6,   8, -6,  2],
    [-1,  2,  -2,  2, -1]
], dtype=np.float32) / 12.0

LAPLACIAN_KERNEL_3X3 = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], dtype=np.float32)

class ImageFeatureExtractor:
    """
    Extracts a fixed 72-dimensional feature vector for steganography detection
    from an image (spatial LSB statistics, Chi-square/RS analysis, SRM residuals,
    SPAM co-occurrences, and DCT blockiness).
    """
    FEATURE_DIM = 72

    def __init__(self):
        pass

    def extract_from_file(self, file_path: Union[str, Path]) -> np.ndarray:
        """Read image and extract feature vector."""
        path_str = str(file_path)
        img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image file: {path_str}")
        return self.extract(img)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract fixed 72-dimensional vector from numpy array (uint8)."""
        if image.ndim == 3:
            # Grayscale luminance
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image[:, :, 0]
        else:
            gray = image.copy()

        gray = gray.astype(np.float32)
        gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)

        feats = []

        # 1. LSB-plane features (8 dims)
        lsb_plane = (gray_uint8 & 1).astype(np.float32)
        lsb_mean = float(np.mean(lsb_plane))
        lsb_var = float(np.var(lsb_plane))
        
        # Horizontal & vertical bit transition frequencies
        h_diff = np.abs(np.diff(lsb_plane, axis=1))
        v_diff = np.abs(np.diff(lsb_plane, axis=0))
        h_trans = float(np.mean(h_diff)) if h_diff.size > 0 else 0.5
        v_trans = float(np.mean(v_diff)) if v_diff.size > 0 else 0.5
        
        # Shannon entropy of LSB plane
        p1 = np.clip(lsb_mean, 1e-9, 1.0 - 1e-9)
        p0 = 1.0 - p1
        lsb_entropy = float(- (p0 * np.log2(p0) + p1 * np.log2(p1)))
        
        # LSB 1D run lengths
        flat_lsb = lsb_plane.ravel()
        diffs = np.diff(flat_lsb)
        change_indices = np.where(diffs != 0)[0]
        if len(change_indices) > 1:
            run_lengths = np.diff(change_indices)
            run_mean = float(np.mean(run_lengths))
            run_var = float(np.var(run_lengths))
        else:
            run_mean = float(len(flat_lsb))
            run_var = 0.0
            
        # 2nd LSB correlation
        bit1_plane = ((gray_uint8 >> 1) & 1).astype(np.float32)
        corr_matrix = np.corrcoef(flat_lsb, bit1_plane.ravel())
        lsb_b1_corr = float(corr_matrix[0, 1]) if np.isfinite(corr_matrix[0, 1]) else 0.0
        
        feats.extend([lsb_mean, lsb_var, h_trans, v_trans, lsb_entropy, run_mean, run_var, lsb_b1_corr])

        # 2. Chi-Square & Sample Pair Analysis (6 dims)
        hist, _ = np.histogram(gray_uint8, bins=256, range=(0, 256))
        # Pairs of Values (PoVs): 2k and 2k+1
        chi_stat = 0.0
        dof = 0
        for k in range(128):
            observed_even = hist[2 * k]
            observed_odd = hist[2 * k + 1]
            expected = (observed_even + observed_odd) / 2.0
            if expected > 5.0:
                chi_stat += ((observed_even - expected) ** 2 + (observed_odd - expected) ** 2) / expected
                dof += 1
        chi_normalized = chi_stat / max(1, dof)
        
        # Sample Pair Analysis (SPA) estimate
        # Count sample pairs (u, v) where u is even and v == u+1 vs u is odd and v == u-1
        u = gray_uint8[:, :-1].ravel().astype(np.int32)
        v = gray_uint8[:, 1:].ravel().astype(np.int32)
        p_pairs = np.sum((u % 2 == 0) & (v == u + 1))
        q_pairs = np.sum((u % 2 == 1) & (v == u - 1))
        spa_diff = float(p_pairs - q_pairs) / max(1, len(u))
        
        # RS analysis mask proxy
        diag_diff = np.abs(gray[:-1, :-1] - gray[1:, 1:])
        diag_var = float(np.var(diag_diff))
        
        # Parity bit autocorrelation at lag 1
        parity = (gray_uint8 & 1).astype(np.float32)
        lag1_corr = float(np.mean(parity[:, :-1] * parity[:, 1:]) - (lsb_mean ** 2))
        
        # Even/Odd histogram bin parity disparity
        even_sum = np.sum(hist[0::2])
        odd_sum = np.sum(hist[1::2])
        parity_imbalance = float(abs(even_sum - odd_sum)) / max(1, len(flat_lsb))
        
        feats.extend([chi_normalized, np.log1p(chi_stat), spa_diff, diag_var, lag1_corr, parity_imbalance])

        # 3. Pixel-value histogram moments (8 dims)
        mean_pix = float(np.mean(gray))
        std_pix = float(np.std(gray))
        sk_pix = float(skew(gray.ravel()))
        kt_pix = float(kurtosis(gray.ravel()))
        
        # Histogram smoothness / bin-to-bin differences
        hist_diff = np.diff(hist)
        hist_diff_mean = float(np.mean(np.abs(hist_diff)))
        hist_diff_std = float(np.std(hist_diff))
        zero_ratio = float(hist[0]) / max(1, len(flat_lsb))
        sat_ratio = float(hist[255]) / max(1, len(flat_lsb))
        
        feats.extend([mean_pix, std_pix, sk_pix, kt_pix, hist_diff_mean, hist_diff_std, zero_ratio, sat_ratio])

        # 4. SRM-style High-pass residuals & SPAM Co-occurrence (36 dims)
        # 1st-order horizontal & vertical residuals
        res_h = gray[:, 1:] - gray[:, :-1]
        res_v = gray[1:, :] - gray[:-1, :]
        
        feats.extend([
            float(np.var(res_h)), float(kurtosis(res_h.ravel())), float(np.max(res_h) - np.min(res_h)),
            float(np.var(res_v)), float(kurtosis(res_v.ravel())), float(np.max(res_v) - np.min(res_v))
        ]) # 6 dims
        
        # 2nd-order Laplacian residual
        res_lap = convolve2d(gray, LAPLACIAN_KERNEL_3X3, mode="same", boundary="symm")
        feats.extend([float(np.var(res_lap)), float(kurtosis(res_lap.ravel()))]) # 2 dims
        
        # SRM KV 5x5 filter residual
        res_kv = convolve2d(gray, KV_KERNEL_5X5, mode="same", boundary="symm")
        kv_var = float(np.var(res_kv))
        kv_mad = float(np.mean(np.abs(res_kv)))
        kv_sk = float(skew(res_kv.ravel()))
        kv_kt = float(kurtosis(res_kv.ravel()))
        feats.extend([kv_var, kv_mad, kv_sk, kv_kt]) # 4 dims
        
        # SPAM Markov transition features:
        # Quantize horizontal residual with T = 3 -> {-3, -2, -1, 0, 1, 2, 3}
        T = 3
        q_res_h = np.clip(np.round(res_h), -T, T).astype(np.int32) + T # map to 0..6
        # Transition matrix P(u, v) = count(q_res[i, j] == u and q_res[i, j+1] == v)
        u_h = q_res_h[:, :-1].ravel()
        v_h = q_res_h[:, 1:].ravel()
        spam_mat = np.zeros((7, 7), dtype=np.float32)
        np.add.at(spam_mat, (u_h, v_h), 1.0)
        # Normalize row-wise
        row_sums = np.sum(spam_mat, axis=1, keepdims=True)
        spam_prob = np.divide(spam_mat, np.maximum(row_sums, 1.0))
        
        # Extract 24 discriminative SPAM statistics:
        # Diagonal elements (7), symmetry differences |P(u, v) - P(v, u)| for upper triangle (21 -> sample 17)
        diag_probs = [float(spam_prob[i, i]) for i in range(7)]
        symm_diffs = []
        for i in range(7):
            for j in range(i + 1, 7):
                symm_diffs.append(float(abs(spam_prob[i, j] - spam_prob[j, i])))
        # Take first 17 symmetry differences -> total 24 SPAM dims
        feats.extend(diag_probs)
        feats.extend(symm_diffs[:17]) # 7 + 17 = 24 dims
        # Total SRM & SPAM: 6 + 2 + 4 + 24 = 36 dims

        # 5. Blockiness & DCT Domain Features (14 dims)
        # Horizontal & vertical blockiness at 8x8 block boundaries
        H, W = gray.shape
        if H >= 16 and W >= 16:
            # Block boundaries at indices 7, 15, 23...
            b_rows = [i for i in range(7, H - 1, 8)]
            non_b_rows = [i for i in range(H - 1) if i not in b_rows]
            
            diff_all_v = np.abs(gray[1:, :] - gray[:-1, :])
            b_diff_v = float(np.mean(diff_all_v[b_rows, :])) if b_rows else 0.0
            nb_diff_v = float(np.mean(diff_all_v[non_b_rows, :])) if non_b_rows else 1.0
            blockiness_v = b_diff_v / max(1e-4, nb_diff_v)
            
            b_cols = [j for j in range(7, W - 1, 8)]
            non_b_cols = [j for j in range(W - 1) if j not in b_cols]
            diff_all_h = np.abs(gray[:, 1:] - gray[:, :-1])
            b_diff_h = float(np.mean(diff_all_h[:, b_cols])) if b_cols else 0.0
            nb_diff_h = float(np.mean(diff_all_h[:, non_b_cols])) if non_b_cols else 1.0
            blockiness_h = b_diff_h / max(1e-4, nb_diff_h)
            blockiness_total = (blockiness_v + blockiness_h) / 2.0
        else:
            blockiness_v, blockiness_h, blockiness_total = 1.0, 1.0, 1.0
            
        feats.extend([blockiness_v, blockiness_h, blockiness_total]) # 3 dims
        
        # 8x8 Block 2D-DCT spectrum statistics
        h_crop = (H // 8) * 8
        w_crop = (W // 8) * 8
        if h_crop >= 8 and w_crop >= 8:
            crop = gray[:h_crop, :w_crop] - 128.0
            # Reshape into (N_blocks, 8, 8)
            blocks = crop.reshape(h_crop // 8, 8, w_crop // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
            # Compute 2D-DCT
            dct_blocks = dct(dct(blocks, axis=1, norm='ortho'), axis=2, norm='ortho')
            
            dc_coeffs = dct_blocks[:, 0, 0]
            dc_var = float(np.var(dc_coeffs))
            
            # Mid-frequency energy vs high-frequency energy
            ac_all = dct_blocks[:, :, :].copy()
            ac_all[:, 0, 0] = 0.0 # zero DC
            total_ac_energy = float(np.sum(ac_all ** 2)) + 1e-6
            
            mid_mask = np.zeros((8, 8), dtype=bool)
            mid_mask[1:4, 1:4] = True
            mid_mask[0, 0] = False
            mid_energy = float(np.sum(dct_blocks[:, mid_mask] ** 2)) / total_ac_energy
            
            high_mask = np.zeros((8, 8), dtype=bool)
            high_mask[4:, 4:] = True
            high_energy = float(np.sum(dct_blocks[:, high_mask] ** 2)) / total_ac_energy
            
            # Mid-frequency histogram parity
            mid_vals = np.round(dct_blocks[:, mid_mask].ravel()).astype(np.int32)
            mid_hist, _ = np.histogram(mid_vals, bins=31, range=(-15, 15))
            mid_zero_frac = float(mid_hist[15]) / max(1, len(mid_vals))
            mid_skew = float(skew(mid_vals))
            mid_kurt = float(kurtosis(mid_vals))
            
            # AC coefficient quantization artifacts (fraction of near-integer values)
            near_int = float(np.mean(np.abs(dct_blocks[:, 1:4, 1:4] - np.round(dct_blocks[:, 1:4, 1:4])) < 0.1))
            ac_mean_abs = float(np.mean(np.abs(ac_all)))
            ac_std = float(np.std(ac_all))
            ac_max = float(np.max(np.abs(ac_all)))
            ac_mad = float(np.median(np.abs(ac_all)))
        else:
            dc_var, mid_energy, high_energy, mid_zero_frac, mid_skew, mid_kurt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            near_int, ac_mean_abs, ac_std, ac_max, ac_mad = 0.0, 0.0, 0.0, 0.0, 0.0

        feats.extend([
            dc_var, mid_energy, high_energy, mid_zero_frac, mid_skew, mid_kurt,
            near_int, ac_mean_abs, ac_std, ac_max, ac_mad
        ]) # 11 dims
        # Total Blockiness & DCT: 3 + 11 = 14 dims

        # Final check & sanitize
        vec = np.array(feats, dtype=np.float32)
        assert len(vec) == self.FEATURE_DIM, f"Extracted {len(vec)} features, expected {self.FEATURE_DIM}"
        vec = np.nan_to_num(vec, nan=0.0, posinf=1e4, neginf=-1e4)
        return vec
