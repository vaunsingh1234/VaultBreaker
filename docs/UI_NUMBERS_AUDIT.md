# UI Numbers Audit & Provenance Verification: VaultBreaker Presentation Layer

Comprehensive audit verifying the exact origin, file path, key, plain-English label, and one-line help text for every numeric value displayed across the VaultBreaker web console (`app/` and `app/ui/`).

All numbers are strictly derived from verified source files or live scan artifacts. There are **zero** hardcoded fallback numbers, **zero** synthetic/fabricated charts, and **zero** typed-in values.

| Page / Component | Label / Metric | Displayed Value | Verified Source File & Key | Plain-English Label & One-Line Help Text |
| :--- | :--- | :--- | :--- | :--- |
| **Global Sidebar** | Active Checkpoint | Dynamic (e.g. `deep_model.pt`) | `models_saved/deep_model.pt` (`Path.name`) | **Model Checkpoint**: Active neural checkpoint file |
| **Global Sidebar** | Compiled Date | Dynamic | `models_saved/deep_model.pt` (`stat.st_mtime`) | **Compiled Date**: Last modified timestamp of model checkpoint |
| **Global Sidebar** | Architecture | `128-d Latent` | `configs/default.yaml` (`models.latent_dim`) | **Unified Latent Dimension**: Embedding dimension shared across image, audio, and video |
| **Global Sidebar** | Target Clean False Alarm Rate | `5.0%` | `models_saved/calibrator_per_format.json` or calibrator object (`calibrator.target_fpr`) | **False alarm rate on clean files**: Maximum acceptable false alarm rate on verified clean media (5.0%) |
| **Global Sidebar** | Alert Threshold | Dynamic (per-format or calibrator) | `models_saved/calibrator_per_format.json` (`<format>.decision_threshold`) | **Alert threshold: files scoring above this are flagged**: Calibrated score cutoff |
| **Global Sidebar (Advanced)** | Calibration Temperature | Dynamic (per-format) | `models_saved/calibrator_per_format.json` (`<format>.temperature`) | **Calibration temperature**: Post-hoc probability scaling parameter fitted on validation set |
| **Landing (Hero)** | Latent Space Badge | `128-d Unified Latent Space` | `configs/default.yaml` (`models.latent_dim`) | **Unified Latent Dimension**: Cross-modal representation size |
| **Landing (Hero)** | Target False Alarm Badge | `5.0% False Alarm Target` | `configs/default.yaml` (`models.target_fpr`) | **False alarm rate on clean files**: Target false alarm limit on clean media |
| **Landing (Hero)** | Uploaded Container Size | `{sz_kb:.1f} KB` | Live uploaded file (`len(file.getvalue()) / 1024`) | **Container Size**: Byte size of media container uploaded for analysis |
| **Landing (Cards)** | Image ROC-AUC | Dynamic (e.g. `0.9966` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`per_format.image.roc_auc`) | **Area under ROC curve**: Diagnostic discrimination ability (1.0 = perfect detection) |
| **Landing (Cards)** | Image Accuracy | Dynamic (e.g. `97.0%` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`per_format.image.accuracy`) | **Overall Accuracy**: Fraction of correct clean vs stego verdicts at calibrated alert threshold |
| **Landing (Cards)** | Image Sources | Dynamic (e.g. `300 distinct source files`) | `docs/RESULTS/metrics_deep.json` (`per_format.image.unique_sources`) | **Distinct Source Files**: Independent source media files evaluated in test partition |
| **Landing (Cards)** | Audio ROC-AUC | Dynamic (e.g. `0.4997` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`per_format.audio.roc_auc`) | **Area under ROC curve**: Diagnostic discrimination ability on audio waveforms |
| **Landing (Cards)** | Audio Accuracy | Dynamic (e.g. `50.0%` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`per_format.audio.accuracy`) | **Overall Accuracy**: Fraction of correct clean vs stego audio verdicts |
| **Landing (Cards)** | Audio Sources | Dynamic (e.g. `143 distinct source files`) | `docs/RESULTS/metrics_deep.json` (`per_format.audio.unique_sources`) | **Distinct Source Files**: Number of independent audio sources evaluated |
| **Landing (Cards)** | Video ROC-AUC | Dynamic (e.g. `0.5715` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`per_format.video.roc_auc`) | **Area under ROC curve**: Diagnostic discrimination ability across video streams |
| **Landing (Cards)** | Video Accuracy | Dynamic (e.g. `55.8%` or "Not available yet") | `docs/RESULTS/metrics_deep.json` (`per_format.video.accuracy`) | **Overall Accuracy**: Fraction of correct clean vs stego video verdicts |
| **Landing (Cards)** | Video Sources | Dynamic (e.g. `60 distinct source files`) | `docs/RESULTS/metrics_deep.json` (`per_format.video.unique_sources`) | **Distinct Source Files**: Number of independent video sources evaluated |
| **Landing (Recent Scans)** | Calibrated Stego Score | `{prob * 100:.1f}%` | `data/scan_history.db` (`scans.probability`) | **Calibrated Stego Score**: Calibrated probability percentage that file contains hidden payload |
| **Scans Page** | Model Selected | Dynamic (e.g. `Deep Model` or `Gradient Boosting`) | `models_saved/model_selection.json` (`formats.<fmt>.model_name`) | **Selected Model**: Model with highest validation ROC-AUC chosen for this format |
| **Scans Page** | Selection Reason | Dynamic (e.g. `Best validation ROC-AUC (0.8201) vs ...`) | `models_saved/model_selection.json` (`formats.<fmt>.reason`) | **Selection Reason**: Validation ROC-AUC justification |
| **Scans Page** | Kanban Column Count | `{len(items)}` | `data/scan_history.db` (`COUNT(*) WHERE status=...`) | **Category Count**: Total inspected files matching this status |
| **Scan Details** | Alert Threshold | Dynamic (per format) | `models_saved/calibrator_per_format.json` (`<format>.decision_threshold`) | **Alert threshold: files scoring above this are flagged**: Calibrated score cutoff |
| **Scan Details** | Calibrated Stego Likelihood | `{prob * 100:.1f}%` | `data/scan_history.db` (`scans.probability`) | **Calibrated Stego Likelihood**: Percentage probability of steganographic content |
| **Scan Details** | Container Byte Size | `{sz_kb:.1f} KB` | `data/scan_history.db` (`scans.file_size_bytes`) | **Container Size**: Byte size of inspected container |
| **Scan Details (Advanced)** | Calibration Temperature | Dynamic (per format) | `models_saved/calibrator_per_format.json` (`<format>.temperature`) | **Calibration temperature**: Logit divisor fitted on validation data |
| **Scan Details (Advanced)** | Raw Uncalibrated Score | `{raw_prob * 100:.1f}%` | `data/scan_history.db` (`scans.raw_probability`) | **Raw uncalibrated score**: Direct sigmoid output prior to temperature scaling |
| **Results Page** | Run Profile Badge | `REAL RUN: LAPTOP PROFILE` | `docs/RESULTS/metrics_deep.json` (`profile`) | **Training Profile**: Real hardware profile used to generate results |
| **Results Page** | Execution Device | `mps` / `cuda` | `docs/RESULTS/metrics_deep.json` (`device`) | **Execution Device**: Hardware accelerator used for training |
| **Results Page** | Epochs Evaluated | `5` | `docs/RESULTS/metrics_deep.json` (`epochs`) | **Checkpoint Epoch**: Best epoch checkpoint selected on validation loss/AUC |
| **Results Page** | Git Commit Hash | Commit SHA | `docs/RESULTS/metrics_deep.json` (`git_commit`) | **Reproducibility Hash**: Git commit tracking exact repository state |
| **Results Page** | Side-by-Side Comparison | Exact Test Manifest Metrics | `docs/RESULTS/comparison.json` (`comparison.<fmt>.<model>`) | **Model Comparison**: Exact metrics across Deep, Handcrafted MLP, and GBDT on same test manifest |

---

### Purged / Deleted Synthetic Elements
The following synthetic or fabricated elements have been permanently deleted from the codebase per project rules:
1. **Radar Chart (`create_radar_chart`)**: Deleted. Previously used fabricated baseline scores `[0.20, 0.18, 0.22, 0.19, 0.18, 0.20]`.
2. **Radial Chart (`create_radial_chart`)**: Deleted. Previously fabricated method likelihood percentages using synthetic weights `[0.1, 0.1, 0.1, 0.1, 0.1, 0.05]`.
3. **Embedding Estimate (`method_guess`)**: Deleted. Previously computed using ad-hoc keyword heuristics on indicator names.
4. **Hardcoded Fallback Literals in `.get()`**: Deleted. Replaced with graceful `"Not available yet"` state and instruction `"Run make eval"`.
