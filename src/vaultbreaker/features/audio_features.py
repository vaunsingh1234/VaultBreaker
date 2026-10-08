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
    from an audio signal (LSB statistics, Chi-square, sample histogram, spectral
    descriptors, LPC residuals, autocorrelation, and MFCCs).
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

        # 1. LSB Statistics & Chi-Square (8 dims)
        u_samples = samples.view(np.uint16)
        lsb = (u_samples & 1).astype(np.float32)
        lsb_mean = float(np.mean(lsb))
        lsb_var = float(np.var(lsb))
        
        # Transition frequency
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
            
        # Audio Chi-Square statistic on Pairs of Values
        # Downsample amplitude to 256 bins for stable counts
        coarse_amp = np.clip((samples.astype(np.int32) + 32768) // 256, 0, 255)
        hist, _ = np.histogram(coarse_amp, bins=256, range=(0, 256))
        chi_stat = 0.0
        dof = 0
        for k in range(128):
            obs_even = hist[2 * k]
            obs_odd = hist[2 * k + 1]
            exp = (obs_even + obs_odd) / 2.0
            if exp > 5.0:
                chi_stat += ((obs_even - exp) ** 2 + (obs_odd - exp) ** 2) / exp
                dof += 1
        chi_norm = chi_stat / max(1, dof)
        chi_log = float(np.log1p(chi_stat))
        
        feats.extend([lsb_mean, lsb_var, lsb_trans, lsb_entropy, run_mean, run_var, chi_norm, chi_log])

        # 2. Time-Domain Histogram & Zero-Crossing (8 dims)
        float_samples = samples.astype(np.float32) / 32768.0
        amp_mean = float(np.mean(float_samples))
        amp_std = float(np.std(float_samples))
        amp_sk = float(skew(float_samples))
        amp_kt = float(kurtosis(float_samples))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(float_samples, frame_length=1024, hop_length=512)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))
        
        zero_ratio = float(np.sum(samples == 0)) / max(1, n_samples)
        dyn_range = float(np.max(float_samples) - np.min(float_samples))
        
        feats.extend([amp_mean, amp_std, amp_sk, amp_kt, zcr_mean, zcr_std, zero_ratio, dyn_range])

        # 3. Spectral & High-Frequency Characteristics (14 dims)
        # STFT
        S = np.abs(librosa.stft(float_samples, n_fft=1024, hop_length=512))
        
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        
        # High-frequency energy ratio (top quarter frequency bins)
        quarter_bin = S.shape[0] // 4
        high_freq_energy = np.sum(S[3 * quarter_bin :, :] ** 2, axis=0)
        total_energy = np.sum(S ** 2, axis=0) + 1e-9
        hf_ratio = high_freq_energy / total_energy
        
        # Spectral flux (frame-to-frame difference)
        flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0)) if S.shape[1] > 1 else np.array([0.0])
        
        noise_floor_10 = float(np.percentile(total_energy, 10))
        noise_median = float(np.median(total_energy))
        
        feats.extend([
            float(np.mean(centroid)), float(np.std(centroid)),
            float(np.mean(flatness)), float(np.std(flatness)),
            float(np.mean(rolloff)), float(np.std(rolloff)),
            float(np.mean(bandwidth)), float(np.std(bandwidth)),
            float(np.mean(hf_ratio)), float(np.std(hf_ratio)),
            noise_floor_10, noise_median,
            float(np.mean(flux)), float(np.std(flux))
        ])

        # 4. LPC Residual & Autocorrelation (10 dims)
        # Autocorrelation
        max_lag = 100
        auto_corr = librosa.autocorrelate(float_samples, max_size=max_lag)
        if len(auto_corr) > 3 and auto_corr[0] > 0:
            norm_ac = auto_corr / auto_corr[0]
            ac_1 = float(norm_ac[1])
            ac_2 = float(norm_ac[2])
            ac_3 = float(norm_ac[3])
            ac_peak_ratio = float(np.max(norm_ac[10:])) if len(norm_ac) > 10 else 0.0
        else:
            ac_1, ac_2, ac_3, ac_peak_ratio = 0.0, 0.0, 0.0, 0.0
            
        # Linear Predictive Coding (order 8)
        try:
            a = librosa.lpc(float_samples, order=8)
            # Inverse filter: e[n] = x[n] - sum(a[k]*x[n-k])
            residual = lfilter(a, [1.0], float_samples)
            lpc_var = float(np.var(residual))
            lpc_kt = float(kurtosis(residual))
            lpc_sk = float(skew(residual))
            res_zcr = float(np.mean(librosa.feature.zero_crossing_rate(residual)[0]))
            lpc_err_mean = float(np.mean(np.abs(residual)))
            lpc_err_max = float(np.max(np.abs(residual)))
        except Exception:
            lpc_var, lpc_kt, lpc_sk, res_zcr, lpc_err_mean, lpc_err_max = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        feats.extend([ac_1, ac_2, ac_3, ac_peak_ratio, lpc_var, lpc_kt, lpc_sk, res_zcr, lpc_err_mean, lpc_err_max])

        # 5. MFCC Features (16 dims)
        mfcc = librosa.feature.mfcc(y=float_samples, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1) # 13 dims
        feats.extend([float(x) for x in mfcc_mean])
        
        mfcc_std_summary = float(np.mean(np.std(mfcc, axis=1)))
        try:
            width = 9 if mfcc.shape[1] >= 9 else (mfcc.shape[1] // 2 * 2 - 1)
            if width >= 3:
                delta_mfcc = librosa.feature.delta(mfcc, width=width)
                delta_summary = float(np.mean(delta_mfcc))
                delta2_summary = float(np.mean(librosa.feature.delta(mfcc, order=2, width=width)))
            else:
                diff = np.diff(mfcc, axis=1) if mfcc.shape[1] > 1 else np.zeros_like(mfcc)
                delta_summary = float(np.mean(diff))
                delta2_summary = 0.0
        except Exception:
            delta_summary = 0.0
            delta2_summary = 0.0
        
        feats.extend([mfcc_std_summary, delta_summary, delta2_summary])

        # Final check & sanitize
        vec = np.array(feats, dtype=np.float32)
        assert len(vec) == self.FEATURE_DIM, f"Extracted {len(vec)} audio features, expected {self.FEATURE_DIM}"
        vec = np.nan_to_num(vec, nan=0.0, posinf=1e4, neginf=-1e4)
        return vec
