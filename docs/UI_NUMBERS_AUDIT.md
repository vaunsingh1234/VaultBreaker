# UI Numbers Audit & Provenance Verification: VaultBreaker Presentation Layer

Comprehensive audit verifying the exact origin, file path, key, plain-English label, and one-line help text for every numeric value displayed across the VaultBreaker web console (`app/` and `app/ui/`).

All numbers are strictly derived from verified source files or live scan artifacts. There are **zero** hardcoded fallback numbers, **zero** synthetic/fabricated charts, and **zero** typed-in values.

| Page / Component | Label / Metric | Displayed Value | Verified Source File & Key | Plain-English Label & One-Line Help Text |
| :--- | :--- | :--- | :--- | :--- |
| **Global Sidebar** | Active Checkpoint | `unified_model.pt` | `models_saved/unified_model.pt` (`Path.name`) | **Model Checkpoint**: Active neural checkpoint file |
| **Global Sidebar** | Compiled Date | `2026-10-16 12:10` | `models_saved/unified_model.pt` (`stat.st_mtime`) | **Compiled Date**: Last modified timestamp of model checkpoint |
| **Global Sidebar** | Architecture | `128-d Latent` | `configs/default.yaml` (`models.latent_dim`) | **Unified Latent Dimension**: Embedding dimension shared across image, audio, and video |
| **Global Sidebar** | Target Clean False Alarm Rate | `5.0%` | `models_saved/calibrator.joblib` (`calibrator.target_fpr`) | **False alarm rate on clean files**: Maximum acceptable false alarm rate on verified clean media (5.0%) |
| **Global Sidebar** | Decision Threshold | `0.6272` | `models_saved/calibrator.joblib` (`calibrator.decision_threshold`) | **Alert threshold: files scoring above this are flagged**: Calibrated score cutoff |
| **Global Sidebar (Advanced)** | Calibration Temperature | `0.7851` | `models_saved/calibrator.joblib` (`calibrator.temperature`) | **Calibration temperature**: Post-hoc probability scaling parameter fitted on validation set |
| **Landing (Hero)** | Latent Space Badge | `128-d Unified Latent Space` | `configs/default.yaml` (`models.latent_dim`) | **Unified Latent Dimension**: Cross-modal representation size |
| **Landing (Hero)** | Target False Alarm Badge | `5.0% False Alarm Target` | `models_saved/calibrator.joblib` (`calibrator.target_fpr`) | **False alarm rate on clean files**: Target false alarm limit on clean media |
| **Landing (Hero)** | Uploaded Container Size | `{sz_kb:.1f} KB` | Live uploaded file (`len(file.getvalue()) / 1024`) | **Container Size**: Byte size of media container uploaded for analysis |
| **Landing (Cards)** | Image ROC-AUC | Dynamic (e.g. `0.9372` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`per_format.image.roc_auc`) | **Area under ROC curve**: Diagnostic discrimination ability (1.0 = perfect detection) |
| **Landing (Cards)** | Image Accuracy | Dynamic (e.g. `88.0%` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`per_format.image.accuracy`) | **Overall Accuracy**: Fraction of correct clean vs stego verdicts at calibrated alert threshold |
| **Landing (Cards)** | Image Sources | Dynamic (e.g. `300 distinct source files`) | `docs/RESULTS/metrics_deep.json` (`per_format.image.unique_sources`) | **Distinct Source Files**: Independent source media files evaluated in test partition |
| **Landing (Cards)** | Audio ROC-AUC | Dynamic (e.g. `0.6044` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`per_format.audio.roc_auc`) | **Area under ROC curve**: Diagnostic discrimination ability on audio waveforms |
| **Landing (Cards)** | Audio Accuracy | Dynamic (e.g. `56.3%` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`per_format.audio.accuracy`) | **Overall Accuracy**: Fraction of correct clean vs stego audio verdicts |
| **Landing (Cards)** | Audio Sources | Dynamic (e.g. `143 distinct source files`) | `docs/RESULTS/metrics_deep.json` (`per_format.audio.unique_sources`) | **Distinct Source Files**: Number of independent audio sources evaluated |
| **Landing (Cards)** | Video ROC-AUC | Dynamic (e.g. `0.9972` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`per_format.video.roc_auc`) | **Area under ROC curve**: Diagnostic discrimination ability across video streams |
| **Landing (Cards)** | Video Accuracy | Dynamic (e.g. `96.7%` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`per_format.video.accuracy`) | **Overall Accuracy**: Fraction of correct clean vs stego video verdicts |
| **Landing (Cards)** | Video Sources | Dynamic (e.g. `60 distinct source files`) | `docs/RESULTS/metrics_deep.json` (`per_format.video.unique_sources`) | **Distinct Source Files**: Number of independent video sources evaluated |
| **Landing (How It Works)** | Sequential Step Index | `1, 2, 3, 4` | UI workflow step counter | **Step Number**: Sequential progression step index |
| **Landing (Recent Scans)** | Calibrated Stego Score | `{prob * 100:.1f}%` | `data/scan_history.db` (`scans.probability`) | **Calibrated Stego Score**: Calibrated probability percentage that file contains hidden payload |
| **Scans Page** | Kanban Column Count | `{len(items)}` | `data/scan_history.db` (`COUNT(*) WHERE status=...`) | **Category Count**: Total inspected files matching this status |
| **Scans Page** | Card Score Badge | `{prob * 100:.1f}%` | `data/scan_history.db` (`scans.probability`) | **Calibrated Stego Score**: Calibrated percentage |
| **Scan Details** | Alert Threshold | `0.6272` | `models_saved/calibrator.joblib` (`calibrator.decision_threshold`) | **Alert threshold: files scoring above this are flagged**: Calibrated score cutoff |
| **Scan Details** | Calibrated Stego Likelihood | `{prob * 100:.1f}%` | `data/scan_history.db` (`scans.probability`) | **Calibrated Stego Likelihood**: Percentage probability of steganographic content |
| **Scan Details** | Container Byte Size | `{sz_kb:.1f} KB` | `data/scan_history.db` (`scans.file_size_bytes`) | **Container Size**: Byte size of inspected container |
| **Scan Details (Advanced)** | Calibration Temperature | `0.7851` | `models_saved/calibrator.joblib` (`calibrator.temperature`) | **Calibration temperature**: Logit divisor fitted on validation data |
| **Scan Details (Advanced)** | Raw Uncalibrated Score | `{raw_prob * 100:.1f}%` | `data/scan_history.db` (`scans.raw_probability`) | **Raw uncalibrated score**: Direct sigmoid output prior to temperature scaling |
| **Scan Details** | Checkpoint Name | `unified_model.pt` | `models_saved/unified_model.pt` (`Path.name`) | **Model Checkpoint**: Active model checkpoint |
| **Scan Details** | Latent Dimension | `128-d` | `configs/default.yaml` (`models.latent_dim`) | **Unified Latent Dimension**: Cross-modal representation size |
| **Scan Details (Artifacts)** | Audio LSB Bit-1 Proportion | `{val * 100:.1f}%` (or "Not available yet") | Live scan report (`report['explanation']['lsb_ones_ratio']`) | **LSB Bit-1 Proportion**: Percentage of ones in least significant bit plane (clean ~50.0%) |
| **Scan Details (Artifacts)** | Audio LSB Transition Rate | `{val * 100:.1f}%` (or "Not available yet") | Live scan report (`report['explanation']['lsb_transition_rate']`) | **LSB Transition Frequency**: Frequency of bit flips between consecutive samples |
| **Scan Details (Artifacts)** | Video Per-Frame Residuals | timeline float values | Live scan report (`report['explanation']['frame_timeline']`) | **Per-Frame Residual Variance**: Inter-frame high-frequency variance timeline |
| **Scan Details (Indicators)** | Most Unusual Measurements | `{dev:+.4f} σ` | Live scan report (`report['top_features']`) | **Most unusual measurements**: Forensic measurements furthest from normal (in standard deviations &sigma;) |
| **Models Page** | Active Checkpoint | `unified_model.pt` | `models_saved/unified_model.pt` (`Path.name`) | **Active Checkpoint**: Checkpoint filename |
| **Models Page** | Checkpoint Compile Date | `2026-10-16 12:10` | `models_saved/unified_model.pt` (`stat.st_mtime`) | **Compiled Date**: Checkpoint modification timestamp |
| **Models Page** | Latent Dimension | `128` | `configs/default.yaml` (`models.latent_dim`) | **Unified Latent Dimension**: Latent vector dimension $Z \in \mathbb{R}^{128}$ |
| **Models Page** | Image Extractor Dimension | `72` | `configs/default.yaml` (`features.image_dim`) | **Image Feature Dimension**: SRM, SPAM Markov, and DCT block features |
| **Models Page** | Audio Extractor Dimension | `56` | `configs/default.yaml` (`features.audio_dim`) | **Audio Feature Dimension**: Bitplane entropy, spectral flux, and MFCC descriptors |
| **Models Page** | Video Extractor Dimension | `64` | `configs/default.yaml` (`features.video_dim`) | **Video Feature Dimension**: Frame pooling, residual variance, and temporal flicker |
| **Models Page** | Target False Alarm Rate | `5.0%` | `models_saved/calibrator.joblib` (`calibrator.target_fpr`) | **False alarm rate on clean files**: Calibrated operating target |
| **Models Page** | Alert Threshold | `0.6272` | `models_saved/calibrator.joblib` (`calibrator.decision_threshold`) | **Alert threshold: files scoring above this are flagged**: Operating score cutoff |
| **Models Page** | Overall Test ROC-AUC | Dynamic (e.g. `0.8990 [0.881, 0.917]` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`overall.roc_auc` and `overall.roc_auc_ci95`) | **Overall Test ROC-AUC**: Area under ROC curve across all evaluated test samples |
| **Models Page (Advanced)** | Calibration Temperature | `0.7851` | `models_saved/calibrator.joblib` (`calibrator.temperature`) | **Calibration temperature**: Logit divisor fitted on validation data |
| **Vault Page** | Image Resolution | `256 x 256` | `configs/default.yaml` (`dataset.image.target_size`) | **Standard Image Dimensions**: Center-crop normalization dimension |
| **Vault Page** | Audio Sample Rate & Duration | `16 kHz, 3.0 s` | `configs/default.yaml` (`dataset.audio.sample_rate`, `dataset.audio.duration`) | **Acoustic Standardization**: Mono sample rate and clip length |
| **Vault Page** | Video Resolution & FPS | `128 x 128, 15 fps` | `configs/default.yaml` (`dataset.video.target_size`, `dataset.video.fps`) | **Video Format Specification**: Video resolution and frame rate |
| **Vault Page** | Test Partition Sample Count | `{len(df_test)} SAMPLES` | `data/splits/manifest_test.csv` (`len(df)`) | **Test Partition Size**: Evaluated sample count |
| **UI Components** | Alert Threshold Line | Dynamic `{threshold:.1%}` | `models_saved/calibrator.joblib` (`calibrator.decision_threshold`) | **Alert threshold line**: Dynamic threshold reference line on activity charts |

---

### Purged / Deleted Synthetic Elements
The following synthetic or fabricated elements have been permanently deleted from the codebase per project rules:
1. **Radar Chart (`create_radar_chart`)**: Deleted. Previously used fabricated baseline scores `[0.20, 0.18, 0.22, 0.19, 0.18, 0.20]`.
2. **Radial Chart (`create_radial_chart`)**: Deleted. Previously fabricated method likelihood percentages using synthetic weights `[0.1, 0.1, 0.1, 0.1, 0.1, 0.05]`.
3. **Embedding Estimate (`method_guess`)**: Deleted. Previously computed using ad-hoc keyword heuristics on indicator names.
4. **Hardcoded Fallback Literals in `.get()`**: Deleted. Replaced with graceful `"Not available yet"` state and instruction `"Run make eval"`.
