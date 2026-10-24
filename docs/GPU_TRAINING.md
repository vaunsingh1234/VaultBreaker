# VaultBreaker GPU Training Guide (Google Colab & Kaggle)

This guide provides step-by-step instructions to train the **VaultBreaker Unified Deep Steganography Neural Network** on cloud GPU accelerators (e.g., Google Colab, Kaggle Notebooks, or AWS/GCP GPU instances).

---

## 1. Overview & Cloud Architecture

VaultBreaker's deep architecture combines:
- **SRNetBackbone** (Images): 30 frozen Spatial Rich Model (SRM) high-pass filters + 3 unpooled residual blocks + 4 pooled residual blocks + Global Average Pooling (GAP) -> 128-d latent space.
- **Audio1DBackbone** (Audio): 8 frozen high-order difference filters + 2 unpooled 1D residual blocks + 4 strided 1D residual blocks + GAP -> 128-d latent space.
- **VideoTemporalBackbone** (Video): Weight-shared frame backbone across sampled frames + multi-pool aggregation (Temporal Mean, Temporal Max, Learned Temporal Self-Attention) -> 128-d latent space.
- **SharedStegoHead & AuxiliaryMethodHead**: Unified binary decision logit + auxiliary method classification supervision.

### Target Performance on GPU Profile
- **Training Budget**: 20,000 image covers, 8,000 audio windows, 1,500 video clips over 30 epochs with AdamW, Cosine Annealing, and PyTorch Mixed Precision (AMP).
- **Target Accuracy / ROC-AUC**:
  - Image LSB ($\ge 0.2$ bpp): $\ge 0.95$ ROC-AUC
  - Audio LSB ($\ge 0.4$): $\ge 0.90$ ROC-AUC
  - Video Frame LSB: $\ge 0.95$ ROC-AUC
- **Checkpoint Constraint**: Under 90 MB total file size (current checkpoint: ~8.90 MB).

---

## 2. Deterministic Real-Cover Download & On-The-Fly Stego Ingestion

> [!NOTE]
> **No Pre-Stored Multi-Gigabyte Data Upload Required!**
> VaultBreaker downloads real cover benchmarks (Imagenette2-160 for images, ESC-50 for audio, plus user media from `data/user_media/` if present) deterministically (`seed: 42`). When running on a remote cloud GPU, you do **not** need to upload large pre-generated stego dataset archives. Running the dataset generation script downloads and standardizes real covers, partitions them into source-disjoint splits, and pairs them with on-the-fly steganographic embeddings during training.

---

## 3. Step-by-Step Instructions: Google Colab

### Step 1: Open a New Colab Notebook and Set Runtime to GPU
1. Go to [Google Colab](https://colab.research.google.com).
2. Click **Runtime** > **Change runtime type**.
3. Select **T4 GPU** (free tier) or **A100 / L4 GPU** (Colab Pro).

### Step 2: Clone Repository & Install Dependencies
Run in a Colab code cell:
```bash
# Clone the VaultBreaker repository
!git clone https://github.com/vaunsingh1234/VaultBreaker.git
%cd VaultBreaker

# Install package dependencies
!pip install --upgrade pip
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -r requirements.txt
!pip install -e .
```

### Step 3: Deterministically Generate the GPU Dataset
Run:
```bash
# Generates disjoint train/val/test splits with zero cross-split source leakage
!python scripts/make_dataset.py --profile gpu --seed 42
```

### Step 4: Launch Deep Training with GPU Acceleration & AMP
Run:
```bash
# Runs full 30-epoch training on CUDA GPU with automatic mixed precision and gradient clipping
!python scripts/train_deep.py --profile gpu --device cuda --seed 42
```
Training logs, epoch timing, learning rate schedules, and validation AUCs will be saved to `models_saved/deep_training_log.csv`. The best checkpoint is automatically saved to `models_saved/deep_model.pt` and temperature-calibrated to `models_saved/deep_calibrator.pt`.

### Step 5: Evaluate Test Set & Generate Heatmap
Run:
```bash
# Evaluates on the held-out test split, computes 95% bootstrap CIs, and generates method x payload heatmap
!python scripts/evaluate.py --device cuda
```

### Step 6: Download Checkpoint to Local Machine
Run:
```python
from google.colab import files
files.download('models_saved/deep_model.pt')
files.download('models_saved/deep_calibrator.pt')
files.download('docs/RESULTS/method_payload_heatmap.png')
files.download('docs/RESULTS/metrics.json')
```

---

## 4. Step-by-Step Instructions: Kaggle Notebooks (GPU Accelerator)

You can either upload `notebooks/kaggle_train.ipynb` directly to Kaggle, or create a new notebook and run these exact steps.

### Step 1: Create Notebook & Configure Hardware
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) and click **New Notebook**.
2. In the right-hand panel under **Notebook settings**:
   - **Accelerator**: Select **GPU T4 x 2** or **GPU P100**.
   - **Internet**: Toggle to **On** (required for cloning and dataset downloads).

### Step 2: Run Training Cells Sequentially

**Cell 1: Clone Repository & Install Dependencies**
```bash
!git clone https://github.com/vaunsingh1234/VaultBreaker.git
%cd VaultBreaker
!pip install --upgrade pip
!pip install -r requirements.txt
!pip install -e .
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
```

**Cell 2: Deterministically Generate GPU Dataset**
```bash
!python scripts/make_dataset.py --profile gpu --seed 42
```

**Cell 3: Launch Full Deep Model Training on GPU**
```bash
!python scripts/train_deep.py --profile gpu --device cuda --seed 42
```

**Cell 4: Evaluate Test Set & Select Best Models on Validation**
```bash
!python scripts/evaluate.py --device cuda
```

**Cell 5: Package Checkpoints and Results for Local Download**
```python
!tar -czvf vaultbreaker_gpu_results.tar.gz docs/RESULTS models_saved
from IPython.display import FileLink
FileLink("vaultbreaker_gpu_results.tar.gz")
```

### Step 3: Import GPU Results into Local VaultBreaker Repository
Once `vaultbreaker_gpu_results.tar.gz` is downloaded to your local machine:
1. Move `vaultbreaker_gpu_results.tar.gz` to your local repository root.
2. Extract the archive (this will overwrite `docs/RESULTS/` and `models_saved/` with genuine GPU artifacts):
   ```bash
   tar -xzvf vaultbreaker_gpu_results.tar.gz -C .
   ```
3. Run `evaluate.py` to regenerate `comparison.json`, `metrics_deep.json`, `metrics_handcrafted.json`, and UI tables from the downloaded GPU checkpoint:
   ```bash
   python scripts/evaluate.py --device auto
   ```
4. Start or refresh the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   The UI will immediately display the real GPU profile badge, ROC-AUC, threshold calibrations, and per-format models.

---

## 5. Checkpoint Hosting & Distribution

VaultBreaker's deep model checkpoint size is **8.90 MB**, which easily fits into standard Git repositories without needing specialized high-bandwidth hosting. However, for open-source model releases:

### Option A: Direct Git Commit
Because the file size is ~8.9 MB (< 100 MB GitHub file size limit), you can commit directly:
```bash
git add models_saved/deep_model.pt models_saved/deep_calibrator.pt
git commit -m "feat: Add trained deep model checkpoint (8.9 MB)"
git push origin main
```

### Option B: Hugging Face Hub (Optional)
If publishing to Hugging Face:
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="models_saved/deep_model.pt",
    path_in_repo="deep_model.pt",
    repo_id="your-username/vaultbreaker-deep-stego",
    repo_type="model"
)
```

---

## 6. Training Command Reference

| Profile | Command | Covers (Img / Aud / Vid) | Target Hardware | Epochs | Expected Wall Clock |
|---|---|---|---|---|---|
| **Smoke** | `python scripts/train_deep.py --profile smoke --device cpu` | 60 / 40 / 12 | Laptop CPU | 2 | ~1 min |
| **Laptop** | `python scripts/train_deep.py --profile laptop --device auto` | 20,000 / 8,000 / 1,500 (up to available covers) | Apple Silicon (MPS) / PC | 8 | ~25 min |
| **GPU** | `python scripts/train_deep.py --profile gpu --device cuda` | 20,000 / 8,000 / 1,500 | T4 / A100 / L4 GPU | 30 | ~2.5 hours |

To resume from an existing checkpoint after an interruption:
```bash
python scripts/train_deep.py --profile gpu --device cuda --resume models_saved/deep_model.pt
```
