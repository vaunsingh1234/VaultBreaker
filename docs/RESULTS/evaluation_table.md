# VaultBreaker Evaluation Report

### Summary Performance by Media Format

| Evaluation Split | Samples | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Detection Error $P_E$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Image** | 6 | 0.8333 | 1.0000 | 0.6667 | 0.8000 | 0.7778 | 0.1667 |
| **Audio** | 6 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.4444 | 0.3333 |
| **Video** | 2 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| **Unified Overall** | **14** | **0.6429** | **1.0000** | **0.2857** | **0.4444** | **0.5408** | **0.3571** |

### Embedding Method Detection Rates

| Embedding Method | Samples | Detection Accuracy | Mean Stego Probability |
| :--- | :---: | :---: | :---: |
| `clean` | 7 | 1.0000 | 0.4850 |
| `dct_ac` | 1 | 1.0000 | 0.5372 |
| `edge_adaptive` | 2 | 0.5000 | 0.5085 |
| `lsb_replacement` | 1 | 0.0000 | 0.4821 |
| `echo_hiding` | 2 | 0.0000 | 0.4736 |
| `audio_lsb` | 1 | 0.0000 | 0.4831 |

### Accuracy vs. Payload Density Curve

| Payload Rate | Stego Samples | Detection Accuracy |
| :--- | :---: | :---: |
| **0.05000000074505806** | 3 | 0.0000 |
| **0.20000000298023224** | 2 | 0.5000 |
| **0.30000001192092896** | 1 | 0.0000 |
| **0.4000000059604645** | 1 | 1.0000 |
