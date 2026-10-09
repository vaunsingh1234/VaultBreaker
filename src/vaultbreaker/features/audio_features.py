import numpy as np
import soundfile as sf
import librosa
from scipy.stats import skew, kurtosis
from scipy.signal import lfilter
from pathlib import Path
from typing import Union

class AudioFeatureExtractor:
    """
    Extracts a fixed 56-dimensional feature vector for audio steganography detection.
    Engineered to resolve weak detection in 16-bit PCM audio:
    1. Digital silence / zero-padded frame isolation and quiet-frame transition statistics.
    2. Difference residual parity disparity (1st, 2nd, and 3rd order residuals).
    3. Local difference residual Pairs-of-Values (PoVs) Chi-Square.
    4. RS Steganalysis on 4-sample audio groups.
    5. Active-frame LSB bit-plane runs, entropy, and transitions.
    6. Short-time frame-level aggregations (mean, std, max of difference variance and parity).
    7. Cepstral peak ratio descriptors for echo hiding detection at delay 50 and 80.
    8. Spectral dynamics and LPC prediction residuals.
    """
    FEATURE_DIM = 56

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def extract_from_file(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load audio file and extract feature vector."""
        path_str = str(file_path)
        samples, sr = sf.read(path_str, dtype="int16")
        if samples.ndim > 1:
            samples = samples[:, 0] # mono
        return self.extract(samples, sr=sr)

    def extract(self, samples: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract fixed 56-dimensional feature vector from 16-bit PCM samples."""
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)

        n_samples = len(samples)
        if n_samples == 0:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        feats = []

        # ------------------------------------------------------------------
        # 1. Digital Silence & Quiet Frame Statistics (5 dims)
        # ------------------------------------------------------------------
        flen = 512
        n_fr = n_samples // flen
        if n_fr > 0:
            fr = samples[: n_fr * flen].reshape(n_fr, flen).astype(np.float64)
            fr_rms = np.sqrt(np.mean(fr ** 2, axis=1))
            silent_mask = fr_rms < 15.0
            quiet_mask = (fr_rms >= 15.0) & (fr_rms < 250.0)
            active_mask = fr_rms >= 250.0
            if not np.any(active_mask):
                active_mask = np.ones(n_fr, dtype=bool)

            sil_ratio = float(np.mean(silent_mask))
            quiet_ratio = float(np.mean(quiet_mask))

            if np.sum(silent_mask) > 0:
                sil_samp = fr[silent_mask].ravel().astype(np.int64)
                sil_nz = float(np.mean(sil_samp != 0))
                sil_d1 = np.diff(sil_samp)
                sil_trans = float(np.mean(np.abs(sil_d1) > 0)) if len(sil_d1) > 0 else 0.0
            else:
                sil_nz = 0.0
                sil_trans = 0.0

            if np.sum(quiet_mask) > 0:
                q_samp = fr[quiet_mask].ravel()
                noise_floor_mad = float(np.median(np.abs(q_samp - np.median(q_samp))))
            else:
                noise_floor_mad = 0.0
        else:
            sil_ratio, quiet_ratio, sil_nz, sil_trans, noise_floor_mad = 0.0, 0.0, 0.0, 0.0, 0.0
            active_mask = np.array([True])
            fr = samples.reshape(1, -1).astype(np.float64)

        feats.extend([sil_ratio, quiet_ratio, sil_nz, sil_trans, noise_floor_mad]) # 5 dims

        # ------------------------------------------------------------------
        # 2. Difference Residuals & Parity Anomaly on Active Frames (10 dims)
        # ------------------------------------------------------------------
        if n_fr > 0 and np.any(active_mask):
            act_samp = fr[active_mask].ravel().astype(np.int64)
        else:
            act_samp = samples.astype(np.int64)

        if len(act_samp) < 4:
            act_samp = samples.astype(np.int64)

        d1 = np.diff(act_samp)
        d2 = np.diff(d1) if len(d1) > 1 else np.zeros(1, dtype=np.int64)
        d3 = np.diff(d2) if len(d2) > 1 else np.zeros(1, dtype=np.int64)

        d1_even = float(np.mean(d1 % 2 == 0)) if len(d1) > 0 else 0.5
        d1_parity_dist = float(abs(d1_even - 0.5))
        d2_even = float(np.mean(d2 % 2 == 0)) if len(d2) > 0 else 0.5
        d2_parity_dist = float(abs(d2_even - 0.5))
        d3_even = float(np.mean(d3 % 2 == 0)) if len(d3) > 0 else 0.5
        d3_parity_dist = float(abs(d3_even - 0.5))

        d1_var = float(np.var(d1)) if len(d1) > 0 else 0.0
        d1_kt = float(kurtosis(d1)) if len(d1) > 4 else 0.0
        d2_var = float(np.var(d2)) if len(d2) > 0 else 0.0
        d2_kt = float(kurtosis(d2)) if len(d2) > 4 else 0.0
        ratio_d2_d1 = float(d2_var / max(1.0, d1_var))

        feats.extend([
            d1_even, d1_parity_dist, d2_even, d2_parity_dist, d3_parity_dist,
            d1_var, d1_kt, d2_var, d2_kt, ratio_d2_d1
        ]) # 10 dims

        # ------------------------------------------------------------------
        # 3. Local Difference Residual Chi-Square & Even/Odd Bin Anomaly (7 dims)
        # ------------------------------------------------------------------
        for d_arr in [d1, d2]:
            clip_d = d_arr[np.abs(d_arr) <= 100] + 100
            h, _ = np.histogram(clip_d, bins=200, range=(0, 200))
            p_even, p_odd = h[0::2], h[1::2]
            tot = p_even + p_odd
            m = tot > 10
            if np.sum(m) > 0:
                exp = tot[m] / 2.0
                chi = float(np.sum((p_even[m] - exp) ** 2 / exp) / np.sum(m))
                chi_log = float(np.log1p(chi))
            else:
                chi, chi_log = 0.0, 0.0
            feats.extend([chi, chi_log])

        d1_even_odd_asym = float(abs(np.sum(d1 % 2 == 0) - np.sum(d1 % 2 != 0))) / max(1.0, float(len(d1)))
        d2_even_odd_asym = float(abs(np.sum(d2 % 2 == 0) - np.sum(d2 % 2 != 0))) / max(1.0, float(len(d2)))
        zero_diff_frac = float(np.mean(d1 == 0)) if len(d1) > 0 else 0.0
        feats.extend([d1_even_odd_asym, d2_even_odd_asym, zero_diff_frac]) # 7 dims total in group 3

        # ------------------------------------------------------------------
        # 4. Sample-Pair Analysis (SPA) & RS Analysis on Audio (6 dims)
        # ------------------------------------------------------------------
        g4 = act_samp[: (len(act_samp) // 4) * 4].reshape(-1, 4)
        if len(g4) > 10:
            f_orig = np.sum(np.abs(np.diff(g4, axis=1)), axis=1)
            g_pos = g4.copy()
            g_pos[:, 1] ^= 1
            g_pos[:, 2] ^= 1
            f_pos = np.sum(np.abs(np.diff(g_pos, axis=1)), axis=1)
            g_neg = g4.copy()
            g_neg[:, 1] = np.where(g4[:, 1] % 2 == 0, g4[:, 1] - 1, g4[:, 1] + 1)
            g_neg[:, 2] = np.where(g4[:, 2] % 2 == 0, g4[:, 2] - 1, g4[:, 2] + 1)
            f_neg = np.sum(np.abs(np.diff(g_neg, axis=1)), axis=1)
            r_pos = float(np.mean(f_pos > f_orig))
            s_pos = float(np.mean(f_pos < f_orig))
            r_neg = float(np.mean(f_neg > f_orig))
            s_neg = float(np.mean(f_neg < f_orig))
            rs_gap = float(r_pos - s_pos)
            rs_asym = float(abs(r_pos - r_neg) + abs(s_pos - s_neg))
        else:
            rs_gap, rs_asym, r_pos, s_pos = 0.0, 0.0, 0.0, 0.0

        u_pairs = act_samp[:-1]
        v_pairs = act_samp[1:]
        p_pairs = np.sum((u_pairs % 2 == 0) & (v_pairs == u_pairs + 1))
        q_pairs = np.sum((u_pairs % 2 == 1) & (v_pairs == u_pairs - 1))
        spa_diff = float(p_pairs - q_pairs) / max(1.0, float(len(u_pairs)))
        even_odd_sample_asym = float(abs(np.sum(act_samp % 2 == 0) - np.sum(act_samp % 2 != 0))) / max(1.0, float(len(act_samp)))

        feats.extend([rs_gap, rs_asym, r_pos, s_pos, spa_diff, even_odd_sample_asym]) # 6 dims

        # ------------------------------------------------------------------
        # 5. LSB Bit-Plane Run Lengths & Entropy on Active Samples (6 dims)
        # ------------------------------------------------------------------
        u_act = act_samp.astype(np.int16).view(np.uint16)
        lsb = (u_act & 1).astype(np.float32)
        lsb_mean = float(np.mean(lsb))
        lsb_var = float(np.var(lsb))
        lsb_diff = np.abs(np.diff(lsb))
        lsb_trans = float(np.mean(lsb_diff)) if len(lsb_diff) > 0 else 0.5
        p1 = np.clip(lsb_mean, 1e-9, 1.0 - 1e-9)
        lsb_entropy = float(- (p1 * np.log2(p1) + (1.0 - p1) * np.log2(1.0 - p1)))
        change_idx = np.where(lsb_diff != 0)[0]
        if len(change_idx) > 1:
            runs = np.diff(change_idx)
            run_mean = float(np.mean(runs))
            run_var = float(np.var(runs))
        else:
            run_mean, run_var = float(len(lsb)), 0.0

        feats.extend([lsb_mean, lsb_var, lsb_trans, lsb_entropy, run_mean, run_var]) # 6 dims

        # ------------------------------------------------------------------
        # 6. Short-Time Frame-Level Aggregations (mean, std, max) (8 dims)
        # ------------------------------------------------------------------
        if n_fr > 0:
            non_sil_indices = [k for k in range(n_fr) if not silent_mask[k]]
            if len(non_sil_indices) > 0:
                fr_vars = [float(np.var(np.diff(fr[k]))) for k in non_sil_indices]
                fr_parities = [float(np.mean(np.diff(fr[k].astype(np.int64)) % 2 == 0)) for k in non_sil_indices]
                fr_transs = [float(np.mean(np.abs(np.diff((fr[k].astype(np.int16).view(np.uint16) & 1).astype(np.float32))))) for k in non_sil_indices]

                fr_d1_v_mean = float(np.mean(fr_vars))
                fr_d1_v_std = float(np.std(fr_vars))
                fr_d1_v_max = float(np.max(fr_vars))

                fr_parity_mean = float(np.mean(fr_parities))
                fr_parity_std = float(np.std(fr_parities))
                fr_parity_max_dist = float(np.max(np.abs(np.array(fr_parities) - 0.5)))

                fr_trans_mean = float(np.mean(fr_transs))
                fr_trans_std = float(np.std(fr_transs))
            else:
                fr_d1_v_mean, fr_d1_v_std, fr_d1_v_max = 0.0, 0.0, 0.0
                fr_parity_mean, fr_parity_std, fr_parity_max_dist = 0.5, 0.0, 0.0
                fr_trans_mean, fr_trans_std = 0.5, 0.0
        else:
            fr_d1_v_mean, fr_d1_v_std, fr_d1_v_max = 0.0, 0.0, 0.0
            fr_parity_mean, fr_parity_std, fr_parity_max_dist = 0.5, 0.0, 0.0
            fr_trans_mean, fr_trans_std = 0.5, 0.0

        feats.extend([
            fr_d1_v_mean, fr_d1_v_std, fr_d1_v_max,
            fr_parity_mean, fr_parity_std, fr_parity_max_dist,
            fr_trans_mean, fr_trans_std
        ]) # 8 dims

        # ------------------------------------------------------------------
        # 7. Echo Hiding Cepstral Descriptors (4 dims)
        # ------------------------------------------------------------------
        float_samp = samples.astype(np.float64)
        if len(float_samp) >= 8192:
            seg = float_samp[:8192]
            rfft_mag = np.maximum(np.abs(np.fft.rfft(seg)), 1e-9)
            cep = np.fft.irfft(np.log(rfft_mag))
            cep_baseline = max(1e-9, float(np.mean(np.abs(cep[30:120]))))
            cep_peak_50 = float(abs(cep[50]) / max(1e-9, np.mean(np.abs(cep[42:58]))))
            cep_peak_80 = float(abs(cep[80]) / max(1e-9, np.mean(np.abs(cep[72:88]))))
            cep_peak_max = float(np.max(np.abs(cep[40:100])) / cep_baseline)
            cep_kurtosis = float(kurtosis(cep[40:100])) if len(cep) > 100 else 0.0
        else:
            cep_peak_50, cep_peak_80, cep_peak_max, cep_kurtosis = 1.0, 1.0, 1.0, 0.0

        feats.extend([cep_peak_50, cep_peak_80, cep_peak_max, cep_kurtosis]) # 4 dims

        # ------------------------------------------------------------------
        # 8. Spectral Dynamics & LPC Residuals (10 dims)
        # ------------------------------------------------------------------
        norm_s = (float_samp / 32768.0).astype(np.float32)
        amp_mean = float(np.mean(norm_s))
        amp_std = float(np.std(norm_s))
        zcr = float(np.mean(np.abs(np.diff(np.sign(norm_s))) > 0)) if len(norm_s) > 1 else 0.0

        try:
            S = np.abs(librosa.stft(norm_s, n_fft=1024, hop_length=512))
            spec_cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
            spec_cent_mean = float(np.mean(spec_cent))
            spec_cent_std = float(np.std(spec_cent))
            spec_roll_mean = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]))
            spec_flat_mean = float(np.mean(librosa.feature.spectral_flatness(S=S)[0]))

            freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
            hf_energy = float(np.sum(S[freqs >= 4000, :] ** 2))
            hf_ratio = float(hf_energy / (np.sum(S ** 2) + 1e-8))

            if S.shape[1] > 1:
                flux_mean = float(np.mean(np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))))
            else:
                flux_mean = 0.0
        except Exception:
            spec_cent_mean, spec_cent_std, spec_roll_mean, spec_flat_mean = 0.0, 0.0, 0.0, 0.0
            hf_ratio, flux_mean = 0.0, 0.0

        try:
            a_lpc = librosa.lpc(norm_s, order=8)
            est = lfilter([0] + -1 * a_lpc[1:].tolist(), [1], norm_s)
            e = norm_s - est
            lpc_err_ratio = float(np.var(e) / max(1e-8, amp_std ** 2))
        except Exception:
            lpc_err_ratio = 0.0

        feats.extend([
            amp_mean, amp_std, zcr,
            spec_cent_mean, spec_cent_std, spec_roll_mean, spec_flat_mean,
            hf_ratio, flux_mean, lpc_err_ratio
        ]) # 10 dims

        vec = np.array(feats, dtype=np.float32)
        assert len(vec) == self.FEATURE_DIM, f"Expected {self.FEATURE_DIM} features, got {len(vec)}"
        vec = np.nan_to_num(vec, nan=0.0, posinf=1e4, neginf=-1e4)
        return vec
