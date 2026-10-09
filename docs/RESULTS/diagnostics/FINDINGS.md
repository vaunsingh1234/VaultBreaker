# Root Cause Diagnostic Findings (Phase 1)

**Date:** 2026-10-08  
**Scope:** Investigation of Weak Audio Steganography Detection (Problem A) and Sub-optimal Image Detection / Confounded Data (Problem B) in VaultBreaker.

---

## 1. Executive Summary

Empirical diagnostics conducted on the training, validation, and test splits revealed clear root causes for the sub-optimal detection performance in both audio and image modalities:
1. **Audio Detection Weakness (ROC-AUC ~0.61, LSB Replacement 49% accuracy, Echo Hiding 8.7%):**
   - **Echo Hiding Feature Blindness:** The existing 56-d `AudioFeatureExtractor` extracts time-domain, spectral, and LPC features up to lag 4, but **zero** cepstral features or autocorrelation at delays corresponding to echo hiding ($\Delta_0=50, \Delta_1=80$). Consequently, echo hiding samples have mean predicted probability $0.48$ (lower than clean audio at $0.49$), resulting in 8.7% accuracy and dragging down the headline audio metric.
   - **Statistical Starvation of Raw 16-bit Pairs-of-Values (PoVs):** For 16-bit audio, the dynamic range spans 65,536 values ($32,768$ pairs). A 3-second audio clip at 16 kHz has only $48,000$ samples total, yielding an average bin count of only $1.46$ samples/bin. Global PoVs Chi-Square across 32,768 bins is statistically starved, producing identical Chi-Square values ($\approx 0.498$) for both clean and stego audio.
   - **Global Variance Dilution:** Acoustic signal variance across entire 3-second clips is on the order of $10^6 - 10^7$, completely masking the $+1/-1$ bit-flip variance ($\sigma^2 \approx 0.5$) in global difference statistics (`d1_var`, `d2_var`).
   - **Digital Silence Masking / Dilution:** Over 22% of ESC-50 clips contain digital silence or zero-padding (up to 50% of samples are $0$). In clean files, silent frames have $0$ transitions; in stego files, they have random $0/1$ noise. Global averaging across silent and loud frames diluted this contrast.
   - **Coupled Generator Cycling:** In `generator.py`, `methods` had length 3 and `payload_rates` had length 4, coupling high payloads ($0.70, 1.00$) with echo hiding.

2. **Image LSB Detection Conservatism (LSB Replacement 49% - 65% recall):**
   - **Confounded Method & Rate Lockstep:** In `generator.py`, `len(methods) == 4` and `len(payload_rates) == 4`. Thus, `lsb_replacement` was **strictly generated at 0.10 bpp only**, while `dct_ac` was strictly generated at 0.50 bpp. `lsb_replacement` was never evaluated at 0.20, 0.30, or 0.50 bpp.
   - **Decision Threshold Conservatism:** Calibration at 5% FPR across the combined multi-modal validation set produced a decision threshold of $\tau = 0.6389$. At $0.10$ bpp, subtle stego samples produce calibrated probabilities of $0.55 - 0.62$, falling below the conservative 0.6389 cutoff.
   - **Duplicate Cover Sources:** Each source in `generator.py` generated two identical clean files (`clean_01.png` and `clean_02.png`) sharing identical SHA-256 hashes, narrowing confidence intervals and artificially reducing unique source diversity.

---

## 2. Audio Pipeline Diagnostics (Problem A)

### 2.1 Audio Embedder Verification & Bit-Flip Linearity
We tested `embed_audio_lsb_replacement`, `embed_audio_lsb_matching`, and `embed_audio_echo` across payload rates $[0.05, 0.10, 0.20, 0.40, 0.70, 1.00]$:
- **LSB Replacement:** Actual changed samples were strictly monotonic:
  - Rate 0.05: $2.52\%$ samples flipped ($\text{BER} = 0.0000$)
  - Rate 0.10: $5.08\%$ samples flipped ($\text{BER} = 0.0000$)
  - Rate 0.20: $10.12\%$ samples flipped ($\text{BER} = 0.0000$)
  - Rate 0.40: $20.20\%$ samples flipped ($\text{BER} = 0.0000$)
  - Rate 0.70: $35.32\%$ samples flipped ($\text{BER} = 0.0000$)
  - Rate 1.00: $50.41\%$ samples flipped ($\text{BER} = 0.0000$)
  *(Note: Flipping 100% of LSBs with random payload changes ~50% of sample values, as expected).*
- **LSB Matching ($\pm 1$):** Identical monotonic bit-flip scaling, bounded in $[-32768, 32767]$, zero clipping error, $\text{BER} = 0.0000$.
- **Echo Hiding:** Delay peaks at $\Delta_0=50$ and $\Delta_1=80$ samples with $\alpha=0.08$. Frame length = 2048.
- **Read/Write Symmetry:** 16-bit WAV PCM read/write exact roundtrip verified with $\max |s_{\text{read}} - s_{\text{orig}}| = 0$ (exact bit preservation).

### 2.2 Why Existing Audio Features Failed
Evaluating individual features on the training set (`univariate_feature_auc.py`) revealed:
- **Top Audio Features:** `spec_bw_mean` (AUC 0.5861), `spec_cent_mean` (AUC 0.5830).
- **Bottom Audio Features:** `lsb_var` (0.5005), `d1_var` (0.5005), `d2_var` (0.5005), `chi_norm` (0.5352).
- **Empirical Cause:**
  1. The 16-bit audio LSB in natural recordings is already quasi-random due to acoustic sensor noise ($\text{LSB entropy} = 1.0000$, $\text{mean} = 0.5012$).
  2. The raw 16-bit PoV histogram has $32,768$ bins. In a clip of $48,000$ samples, average bin occupancy is $1.46$. Chi-square evaluates to $\approx 0.498$ on BOTH clean and stego audio.

### 2.3 Empirical Discovery: First-Difference Parity & Local Residual Chi-Square
We tested difference residual parity and local residual statistics:
1. **First-Difference Parity Disparity ($d_1 = s[i] - s[i-1]$):**
   - Natural audio exhibits sample continuity: $d_1$ is even **52.56%** of the time ($\text{std} = 0.0756$).
   - LSB Replacement / Matching drives $P(d_1 \text{ is even}) \to 0.50000$:
     - Rate 0.0: $0.52559$ ($\text{distance from } 0.5 = 0.02559$)
     - Rate 0.2: $0.51686$ ($\text{distance} = 0.01686$)
     - Rate 0.4: $0.50925$ ($\text{distance} = 0.00925$)
     - Rate 0.7: $0.50319$ ($\text{distance} = 0.00319$)
     - Rate 1.0: $0.49986$ ($\text{distance} = 0.00014$, $\text{std} = 0.00217$!)
   - This signal is **strictly monotonic** and exhibits near-zero variance at high payloads.
2. **Local Difference Residual Chi-Square:**
   - Limiting the residual to small differences $d_1 \in [-100, 100]$ concentrates all $48,000$ samples into 200 bins (~240 samples/bin).
   - Chi-Square on difference pairs drops monotonically:
     - Rate 0.0 (Clean): $\chi^2_{\text{norm}} = 12.2962$
     - Rate 0.2: $\chi^2_{\text{norm}} = 8.2109$
     - Rate 0.4: $\chi^2_{\text{norm}} = 5.9167$
     - Rate 0.7: $\chi^2_{\text{norm}} = 4.6625$
     - Rate 1.0: $\chi^2_{\text{norm}} = 4.3097$
3. **Digital Silence & Short-Time Frame Statistics:**
   - 112 clips in ESC-50 contain extensive silent segments ($> 1,000$ zero samples).
   - Embedding LSB stego flips zeros to ones ($24,259$ zeros $\to 12,249$ zeros at rate 1.0).
   - Segmenting audio into 512-sample frames and computing quiet-frame residual variance, quiet-frame zero fraction, and active-frame statistics provides robust separation.

---

## 3. Image Pipeline Diagnostics (Problem B)

### 3.1 Confounded Method & Rate Coupling
In `generator.py`:
```python
variants = [
    (methods[i % len(methods)], payload_rates[i % len(payload_rates)]),
    (methods[(i + 1) % len(methods)], payload_rates[(i + 1) % len(payload_rates)])
]
```
Since both `methods` and `payload_rates` have length 4, `lsb_replacement` was strictly paired with rate $0.10$ bpp.
At $0.10$ bpp (only 10% of pixels modified), classic global Chi-Square drops subtly ($2.36 \to 2.17$), within the natural inter-image variance.
In contrast, Spatial Rich Model (SRM) KV-filter variance increases by $43\%$ ($0.4616 \to 0.6600$) and SPAM diagonal features have univariate AUC $0.7557$.

### 3.2 Decision Threshold Analysis
- Calibrating at 5% FPR on the validation set produced threshold $\tau = 0.6389$.
- Clean image false positive rate at $\tau = 0.6389$ is $4.35\%$ ($95.65\%$ specificity).
- For $0.10$ bpp LSB replacement, mean calibrated probability is $0.7197$, but $34.4\%$ of samples fell between $0.50$ and $0.6389$, resulting in $65.6\%$ recall on the test set.
- Operating points on validation:
  - 1% FPR: Threshold = $0.7252$, Recall = $78.4\%$
  - 5% FPR: Threshold = $0.5587$, Recall = $84.3\%$
  - 10% FPR: Threshold = $0.4676$, Recall = $88.8\%$

---

## 4. Root-Cause Remediation Plan (Phase 2 & Phase 3)

| Component | Root Cause | Planned Fix |
| :--- | :--- | :--- |
| **Audio Embedder & Generator** | Method/rate confounded; duplicate clean files; echo hiding mixed into headline. | Decouple method and rate via uniform sampling / offset indexing; generate exactly 1 clean file per source; separate echo hiding reporting from headline audio metric. |
| **Audio Features** | Statistically starved 16-bit PoVs; global variance dilution; silence unhandled. | Add: (1) First & second difference parity disparity, (2) Local difference residual Chi-Square ($d_1 \in [-100, 100]$), (3) Short-time frame statistics (active vs quiet frame RMS & zero-counts), (4) Cepstral peak ratio for echo hiding. |
| **Image Features & Generator** | Confounded 0.10 rate; duplicate clean files. | Generate 1 clean copy per source; uncouple payload rates across all methods; strengthen Sample Pair Analysis (SPA) and SRM KV residual statistics. |
| **Calibration & Threshold** | Conservative joint threshold penalizing low payloads. | Calibrate operating threshold at target FPR on validation split using balanced per-modality validation sets; report at 1%, 5%, and 10% FPR. |
