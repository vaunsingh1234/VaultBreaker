import numpy as np
import librosa
from typing import Dict, Any

def explain_audio(samples: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    """
    Generate forensic explainability artifacts for audio:
    1. STFT Spectrogram (power dB)
    2. LSB Run Lengths and Bit Distribution
    """
    if samples.dtype != np.int16:
        samples = samples.astype(np.int16)

    float_samples = samples.astype(np.float32) / 32768.0

    # 1. STFT Spectrogram
    D = librosa.stft(float_samples, n_fft=1024, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # 2. LSB distribution
    u_samples = samples.view(np.uint16)
    lsb = u_samples & 1
    ones_ratio = float(np.mean(lsb))

    # Transitions
    transitions = np.sum(lsb[:-1] != lsb[1:])
    trans_rate = float(transitions) / max(1, len(lsb) - 1)

    return {
        "spectrogram_db": S_db,
        "sample_rate": sr,
        "lsb_ones_ratio": ones_ratio,
        "lsb_transition_rate": trans_rate
    }
