import json
from pathlib import Path

def create_notebook():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / "kaggle_train.ipynb"

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# VaultBreaker GPU Training on Kaggle (CUDA Enabled)\n",
                    "\n",
                    "This notebook executes full **VaultBreaker** GPU training on a cloud GPU instance.\n",
                    "\n",
                    "### Instructions:\n",
                    "1. In Kaggle Notebook Settings (right sidebar):\n",
                    "   - **Accelerator**: Set to **GPU T4 x 2** or **GPU P100**\n",
                    "   - **Internet**: Set to **On**\n",
                    "2. Run the cells below sequentially.\n",
                    "3. Download the resulting `vaultbreaker_gpu_results.tar.gz` and extract it into your local repository."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 1: Clone Repository and Install Dependencies\n",
                    "!git clone https://github.com/vaunsingh1234/VaultBreaker.git\n",
                    "%cd VaultBreaker\n",
                    "!pip install --upgrade pip\n",
                    "!pip install -r requirements.txt\n",
                    "!pip install -e .\n",
                    "import torch\n",
                    "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"Device: {torch.cuda.get_device_name(0)}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 2: Deterministically Generate GPU Dataset (Disjoint Source Splits)\n",
                    "!python scripts/make_dataset.py --profile gpu --seed 42"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 3: Launch Full Deep Model Training on GPU with AMP\n",
                    "!python scripts/train_deep.py --profile gpu --device cuda --seed 42"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 4: Evaluate Test Set, Run Validation Model Selection, and Generate Heatmap\n",
                    "!python scripts/evaluate.py --device cuda"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 5: Package Checkpoints and Results for Local Download\n",
                    "!tar -czvf vaultbreaker_gpu_results.tar.gz docs/RESULTS models_saved\n",
                    "from IPython.display import FileLink\n",
                    "FileLink('vaultbreaker_gpu_results.tar.gz')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### How to import into your local repository:\n",
                    "1. Download `vaultbreaker_gpu_results.tar.gz`.\n",
                    "2. Place in your repo root and extract:\n",
                    "   ```bash\n",
                    "   tar -xzvf vaultbreaker_gpu_results.tar.gz -C .\n",
                    "   ```\n",
                    "3. Run `python scripts/evaluate.py --device auto` to regenerate all UI metrics and comparisons."
                ]
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Generated {nb_path} successfully.")

if __name__ == "__main__":
    create_notebook()
