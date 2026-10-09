# VaultBreaker Empirical Performance Summary

### Summary Performance by Media Format (with 95% Bootstrap CIs)

| Evaluation Split | Samples | Accuracy (95% CI) | Precision | Recall | F1 Score | ROC-AUC (95% CI) | Detection Error $P_E$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Image** | 600 | 0.8800 `[0.855, 0.903]` | 0.9597 | 0.7933 | 0.8686 | 0.9372 `[0.918, 0.955]` | 0.1100 |
| **Audio** | 286 | 0.5629 `[0.524, 0.601]` | 0.6500 | 0.2727 | 0.3842 | 0.6044 `[0.556, 0.651]` | 0.4056 |
| **Video** | 120 | 0.9667 `[0.925, 0.992]` | 1.0000 | 0.9333 | 0.9655 | 0.9972 `[0.990, 1.000]` | 0.0250 |
| **Unified Overall** | **1006** | **0.8002** `[0.776, 0.824]` | **0.9148** | **0.6620** | **0.7682** | **0.8990** `[0.881, 0.917]` | **0.1839** |

> **Audio LSB Headline (Excl. Echo Hiding):** Accuracy: **0.6245** (95% CI: `[0.581, 0.665]`), ROC-AUC: **0.6151** (95% CI: `[0.557, 0.675]`), N=237

### Embedding Method Detection Rates

| Embedding Method | Samples | Detection Accuracy | Mean Stego Probability |
| :--- | :---: | :---: | :---: |
| `clean` | 503 | 0.9384 | 0.2672 |
| `dct_ac` | 89 | 1.0000 | 0.9862 |
| `echo_hiding` | 49 | 0.2653 | 0.5263 |
| `edge_adaptive` | 79 | 0.4557 | 0.5696 |
| `frame_lsb` | 23 | 0.9565 | 0.9485 |
| `frame_lsb_matching` | 37 | 0.9189 | 0.9319 |
| `lsb_matching` | 114 | 0.5614 | 0.7071 |
| `lsb_replacement` | 112 | 0.6696 | 0.7605 |

### Accuracy vs. Payload Density Curve

| Payload Rate | Stego Samples | Detection Accuracy |
| :--- | :---: | :---: |
| **0.10** | 101 | 0.5941 |
| **0.20** | 127 | 0.6614 |
| **0.30** | 73 | 0.8767 |
| **0.40** | 60 | 0.5167 |
| **0.50** | 80 | 1.0000 |
| **0.70** | 31 | 0.1935 |
| **1.00** | 31 | 0.2581 |
