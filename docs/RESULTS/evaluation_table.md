# VaultBreaker Empirical Performance Summary

### Summary Performance by Media Format (with 95% Bootstrap CIs)

| Evaluation Split | Samples | Accuracy (95% CI) | Precision | Recall | F1 Score | ROC-AUC (95% CI) | Detection Error $P_E$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Image** | 276 | 0.8877 `[0.848, 0.920]` | 0.9652 | 0.8043 | 0.8775 | 0.9627 `[0.940, 0.981]` | 0.0942 |
| **Audio** | 152 | 0.5658 `[0.493, 0.645]` | 0.6923 | 0.2368 | 0.3529 | 0.6084 `[0.515, 0.692]` | 0.4211 |
| **Video** | 128 | 0.9609 `[0.922, 0.992]` | 0.9683 | 0.9531 | 0.9606 | 0.9980 `[0.993, 1.000]` | 0.0156 |
| **Unified Overall** | **556** | **0.8165** `[0.784, 0.849]` | **0.9314** | **0.6835** | **0.7884** | **0.9258** `[0.903, 0.944]` | **0.1745** |

> **Audio LSB Headline (Excl. Echo Hiding):** Accuracy: **0.6512** (95% CI: `[0.574, 0.736]`), ROC-AUC: **0.6569** (95% CI: `[0.567, 0.754]`), N=129

### Embedding Method Detection Rates

| Embedding Method | Samples | Detection Accuracy | Mean Stego Probability |
| :--- | :---: | :---: | :---: |
| `clean` | 278 | 0.9496 | 0.2260 |
| `dct_ac` | 35 | 1.0000 | 0.9947 |
| `echo_hiding` | 23 | 0.0870 | 0.4827 |
| `edge_adaptive` | 37 | 0.6216 | 0.7175 |
| `frame_lsb` | 32 | 1.0000 | 0.9723 |
| `frame_lsb_matching` | 32 | 0.9062 | 0.9119 |
| `lsb_matching` | 64 | 0.6562 | 0.7910 |
| `lsb_replacement` | 55 | 0.4909 | 0.6829 |

### Accuracy vs. Payload Density Curve

| Payload Rate | Stego Samples | Detection Accuracy |
| :--- | :---: | :---: |
| **0.10** | 53 | 0.7358 |
| **0.20** | 72 | 0.7500 |
| **0.30** | 37 | 0.6216 |
| **0.40** | 41 | 0.7317 |
| **0.50** | 35 | 1.0000 |
| **0.70** | 14 | 0.2143 |
| **1.00** | 26 | 0.2308 |
