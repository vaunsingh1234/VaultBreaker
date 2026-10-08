import numpy as np
import soundfile as sf
import librosa
from scipy.stats import skew, kurtosis
from scipy.signal import lfilter
from pathlib import Path
from typing import Union

class AudioFeatureExtractor:
    """
    Extracts a fixed 56-dimensional feature vector for steganography detection
    from an audio signal (LSB bit-plane statistics, Pairs-of-Values Chi-Square,
    first and second-order difference residuals, noise floor characteristics,
    spectral dynamics, LPC prediction residuals, and MFCC descriptors).
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

        feats = []
        n_samples = len(samples)
        if n_samples == 0:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # 1. LSB Statistics & Proper Raw Pairs-of-Values (PoVs) Chi-Square (10 dims)
        u_samples = samples.view(np.uint16)
        lsb = (u_samples & 1).astype(np.float32)
        lsb_mean = float(np.mean(lsb))
        lsb_var = float(np.var(lsb))
        
        # Transition frequency between adjacent samples
        lsb_diff = np.abs(np.diff(lsb))
        lsb_trans = float(np.mean(lsb_diff)) if len(lsb_diff) > 0 else 0.5
        
        # Shannon entropy
        p1 = np.clip(lsb_mean, 1e-9, 1.0 - 1e-9)
        p0 = 1.0 - p1
        lsb_entropy = float(- (p0 * np.log2(p0) + p1 * np.log2(p1)))
        
        # Run lengths
        change_indices = np.where(lsb_diff != 0)[0]
        if len(change_indices) > 1:
            runs = np.diff(change_indices)
            run_mean = float(np.mean(runs))
            run_var = float(np.var(runs))
        else:
            run_mean = float(n_samples)
            run_var = 0.0

        # Uncut Pairs-of-Values (PoVs) Westfeld & Pfitzmann Chi-Square on raw 16-bit samples
        # Each pair k corresponds to values {2k, 2k+1}
        pairs = u_samples >> 1
        evens = (u_samples & 1) == 0
        c_even = np.bincount(pairs[evens], minlength=32768)
        c_odd = np.bincount(pairs[~evens], minlength=32768)
        pair_totals = c_even + c_odd
        sig_mask = pair_totals > 5
        
        if np.sum(sig_mask) > 0:
            exp_pair = pair_totals[sig_mask] / 2.0
            chi_val = np.sum((c_even[sig_mask] - exp_pair) ** 2 / exp_pair)
            dof = int(np.sum(sig_mask))
            chi_norm = float(chi_val / max(1, dof))
            chi_log = float(np.log1p(chi_val))
        else:
            chi_norm = 0.0
            chi_log = 0.0

        even_odd_asymmetry = float(abs(np.sum(c_even) - np.sum(c_odd))) / max(1.0, float(n_samples))
        feats.extend([
            lsb_mean, lsb_var, lsb_trans, lsb_entropy,
            run_mean, run_var,
            chi_norm, chi_log, even_odd_asymmetry,
            float(np.mean(u_samples % 4 == 0))
        ]) # 10 dims

        # 2. Difference Residuals & Noise Floor (10 dims)
        float_samples = samples.astype(np.float64)
        d1 = np.diff(float_samples)
        d2 = np.diff(d1)
        
        d1_mean = float(np.mean(d1))
        d1_var = float(np.var(d1))
        d1_sk = float(skew(d1)) if len(d1) > 2 else 0.0
        d1_kt = float(kurtosis(d1)) if len(d1) > 2 else 0.0

        d2_var = float(np.var(d2)) if len(d2) > 0 else 0.0
        d2_kt = float(kurtosis(d2)) if len(d2) > 2 else 0.0

        # High-frequency noise floor (Laplacian-like diff)
        mad_d1 = float(np.median(np.abs(d1 - np.median(d1))))
        ratio_d2_d1 = float(d2_var / max(1e-4, d1_var))

        # Difference distribution entropy (binned)
        hist_d1, _ = np.histogram(np.clip(d1, -100, 100), bins=64)
        p_d1 = hist_d1 / max(1.0, float(np.sum(hist_d1)))
        p_d1 = p_d1[p_d1 > 0]
        ent_d1 = float(-np.sum(p_d1 * np.log2(p_d1)))

        hist_d2, _ = np.histogram(np.clip(d2, -100, 100), bins=64)
        p_d2 = hist_d2 / max(1.0, float(np.sum(hist_d2)))
        p_d2 = p_d2[p_d2 > 0]
        ent_d2 = float(-np.sum(p_d2 * np.log2(p_d2)))

        feats.extend([
            d1_mean, d1_var, d1_sk, d1_kt,
            d2_var, d2_kt,
            mad_d1, ratio_d2_d1, ent_d1, ent_d2
        ]) # 10 dims

        # 3. Time-Domain & Zero-Crossing (6 dims)
        norm_samples = (float_samples / 32768.0).astype(np.float32)
        amp_mean = float(np.mean(norm_samples))
        amp_std = float(np.std(norm_samples))
        amp_sk = float(skew(norm_samples))
        amp_kt = float(kurtosis(norm_samples))
        
        zcr = librosa.feature.zero_crossing_rate(norm_samples, frame_length=1024, hop_length=512)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))
        feats.extend([amp_mean, amp_std, amp_sk, amp_kt, zcr_mean, zcr_std]) # 6 dims

        # 4. Spectral & High-Frequency Characteristics (14 dims)
        S = np.abs(librosa.stft(norm_samples, n_fft=1024, hop_length=512))
        spec_cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        spec_roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
        spec_flat = librosa.feature.spectral_flatness(S=S)[0]
        spec_bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        
        # High-frequency band energy (> 4 kHz)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        hf_mask = freqs >= 4000
        hf_energy = float(np.sum(S[hf_mask, :] ** 2))
        total_energy = float(np.sum(S ** 2)) + 1e-8
        hf_ratio = hf_energy / total_energy
        
        # Spectral flux (spectral change between frames)
        if S.shape[1] > 1:
            flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))
            flux_mean = float(np.mean(flux))
            flux_std = float(np.std(flux))
        else:
            flux_mean, flux_std = 0.0, 0.0

        feats.extend([
            float(np.mean(spec_cent)), float(np.std(spec_cent)),
            float(np.mean(spec_roll)), float(np.std(spec_roll)),
            float(np.mean(spec_flat)), float(np.std(spec_flat)),
            float(np.mean(spec_bw)), float(np.std(spec_bw)),
            hf_ratio, float(np.log1p(hf_energy)),
            flux_mean, flux_std,
            float(np.median(spec_cent)), float(np.percentile(spec_cent, 90) - np.percentile(spec_cent, 10))
        ]) # 14 dims

        # 5. Autocorrelation & LPC Residuals (8 dims)
        # Fast autocorrelation
        if n_samples > 2048:
            seg = norm_samples[:2048]
            ac = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
            ac = ac / max(1e-8, ac[0])
            r1, r2, r3, r4 = float(ac[1]), float(ac[2]), float(ac[3]), float(ac[4])
        else:
            r1, r2, r3, r4 = 0.0, 0.0, 0.0, 0.0

        # Linear prediction error (order 8)
        try:
            a_lpc = librosa.lpc(norm_samples, order=8)
            est = lfilter([0] + -1 * a_lpc[1:].tolist(), [1], norm_samples)
            e = norm_samples - est
            lpc_var = float(np.var(e))
            lpc_sk = float(skew(e))
            lpc_kt = float(kurtosis(e))
            lpc_err_ratio = lpc_var / max(1e-8, amp_std ** 2)
        except Exception:
            lpc_var, lpc_sk, lpc_kt, lpc_err_ratio = 0.0, 0.0, 0.0, 0.0

        feats.extend([r1, r2, r3, r4, lpc_var, lpc_sk, lpc_kt, lpc_err_ratio]) # 8 dims

        # 6. MFCC Summary (8 dims)
        try:
            mfcc = librosa.feature.mfcc(y=norm_samples, sr=sr, n_mfcc=4, n_fft=1024, hop_length=512)
            mfcc_means = np.mean(mfcc, axis=1).tolist()
            mfcc_stds = np.std(mfcc, axis=1).tolist()
        except Exception:
            mfcc_means = [0.0] * 4
            mfcc_stds = [0.0] * 4
        feats.extend([float(x) for x in mfcc_means + mfcc_stds]) # 8 dims

        vec = np.array(feats, dtype=np.float32)
        assert len(vec) == self.FEATURE_DIM, f"Expected {self.FEATURE_DIM} features, got {len(vec)}"
        vec = np.nan_to_num(vec, nan=0.0, posinf=1e4, neginf=-1e4)
        return vec
