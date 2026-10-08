import numpy as np
from typing import Tuple, Optional

def embed_audio_echo(
    audio_samples: np.ndarray,
    payload_bits: np.ndarray,
    delay_0: int = 50,
    delay_1: int = 80,
    alpha: float = 0.08,
    frame_len: int = 2048
) -> Tuple[np.ndarray, dict]:
    """
    Audio Echo Hiding steganography.
    Segments audio into frames and introduces an attenuated delayed echo.
    delay_0 represents bit 0, delay_1 represents bit 1.
    """
    samples = audio_samples.astype(np.float32).copy()
    n_bits = len(payload_bits)
    max_frames = len(samples) // frame_len

    if n_bits > max_frames:
        raise ValueError(f"Payload bits ({n_bits}) exceed frame capacity ({max_frames})")

    stego = samples.copy()
    for i in range(n_bits):
        bit = int(payload_bits[i]) & 1
        d = delay_1 if bit == 1 else delay_0
        start = i * frame_len
        end = min(start + frame_len, len(samples))
        frame = samples[start:end]

        echo_signal = np.zeros_like(frame)
        if len(frame) > d:
            echo_signal[d:] = frame[:-d] * alpha
        stego[start:end] = frame + echo_signal

    stego_int16 = np.clip(stego, -32768, 32767).astype(np.int16)
    meta = {
        "delay_0": delay_0,
        "delay_1": delay_1,
        "alpha": alpha,
        "frame_len": frame_len,
        "n_bits": n_bits
    }
    return stego_int16, meta

def extract_audio_echo(
    stego_samples: np.ndarray,
    meta: dict
) -> np.ndarray:
    """
    Extract bits from echo-hidden audio using cepstral peak detection.
    """
    samples = stego_samples.astype(np.float32)
    delay_0 = meta["delay_0"]
    delay_1 = meta["delay_1"]
    frame_len = meta["frame_len"]
    n_bits = meta["n_bits"]

    extracted = []
    for i in range(n_bits):
        start = i * frame_len
        end = min(start + frame_len, len(samples))
        frame = samples[start:end]
        if len(frame) < max(delay_0, delay_1) + 10:
            extracted.append(0)
            continue

        # Real cepstrum: ifft(log(|fft(x)|))
        fft_mag = np.abs(np.fft.rfft(frame))
        fft_mag = np.maximum(fft_mag, 1e-12)
        cepstrum = np.fft.irfft(np.log(fft_mag))

        peak_0 = cepstrum[delay_0] if delay_0 < len(cepstrum) else 0
        peak_1 = cepstrum[delay_1] if delay_1 < len(cepstrum) else 0

        extracted.append(1 if peak_1 > peak_0 else 0)

    return np.array(extracted, dtype=np.uint8)
