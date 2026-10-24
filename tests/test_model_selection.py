"""Unit tests for model selection logic (Item 5).

Verifies:
1. Model selection is chosen STRICTLY on validation data (manifest_val.csv), NEVER test data.
2. Selection picks the model with the highest validation ROC-AUC per format.
3. Model selection is correctly saved to models_saved/model_selection.json.
4. Predictor routes each modality to the selected model and includes validation AUC justification.
"""

import json
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate import select_best_models_on_validation


def test_model_selection_highest_val_auc(tmp_path):
    """Verify select_best_models_on_validation picks the highest validation ROC-AUC per format."""
    candidate_scores = {
        "image": {
            "deep_model": 0.9750,
            "handcrafted_mlp": 0.8200,
            "gradient_boosting": 0.8800
        },
        "audio": {
            "deep_model": 0.4990,
            "handcrafted_mlp": 0.6500,
            "gradient_boosting": 0.8120
        },
        "video": {
            "deep_model": 0.5670,
            "handcrafted_mlp": 0.7100,
            "gradient_boosting": 0.9600
        }
    }

    selection = select_best_models_on_validation(
        models_dir=tmp_path,
        candidate_scores=candidate_scores,
        save_path=tmp_path / "model_selection.json"
    )

    formats = selection["formats"]
    # Image: Deep (0.975) > GBDT (0.880) > MLP (0.820)
    assert formats["image"]["selected_model"] == "deep_model"
    assert formats["image"]["validation_roc_auc"] == 0.9750
    assert "0.9750" in formats["image"]["reason"]

    # Audio: GBDT (0.812) > MLP (0.650) > Deep (0.499)
    assert formats["audio"]["selected_model"] == "gradient_boosting"
    assert formats["audio"]["validation_roc_auc"] == 0.8120
    assert "0.8120" in formats["audio"]["reason"]

    # Video: GBDT (0.960) > MLP (0.710) > Deep (0.567)
    assert formats["video"]["selected_model"] == "gradient_boosting"
    assert formats["video"]["validation_roc_auc"] == 0.9600

    # Verify JSON file was written properly
    out_json = tmp_path / "model_selection.json"
    assert out_json.exists()
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["formats"]["image"]["selected_model"] == "deep_model"
    assert saved["formats"]["audio"]["selected_model"] == "gradient_boosting"


def test_model_selection_never_accesses_test_data(monkeypatch, tmp_path):
    """Verify that model selection logic NEVER opens or accesses manifest_test.csv."""
    accessed_files = []

    real_read_csv = pd.read_csv

    def mock_read_csv(filepath_or_buffer, *args, **kwargs):
        fp_str = str(filepath_or_buffer)
        accessed_files.append(fp_str)
        if "manifest_test.csv" in fp_str or "test_balanced" in fp_str:
            raise AssertionError(f"FORBIDDEN ACCESS: Model selection must NEVER access test data! Accessed: {fp_str}")
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    candidate_scores = {
        "image": {"deep_model": 0.95, "handcrafted_mlp": 0.80, "gradient_boosting": 0.85},
        "audio": {"deep_model": 0.50, "handcrafted_mlp": 0.70, "gradient_boosting": 0.80},
        "video": {"deep_model": 0.60, "handcrafted_mlp": 0.75, "gradient_boosting": 0.90}
    }

    select_best_models_on_validation(
        models_dir=tmp_path,
        candidate_scores=candidate_scores,
        save_path=tmp_path / "model_selection.json"
    )

    # Verify no test files were touched
    for f in accessed_files:
        assert "test" not in f.lower(), f"Test file {f} was accessed during model selection!"


def test_predictor_unified_output_schema():
    """Verify Predictor returns standard output schema with model used and reason."""
    from vaultbreaker.inference.predict import Predictor

    # Initialize predictor in CPU mode
    predictor = Predictor(device="cpu")

    # Predict on a synthetic or real image
    sample_img = Path("data/raw/image/cover_img_0999.png")
    if not sample_img.exists():
        sample_img = next(Path("data").rglob("*.png"), None)

    if sample_img and sample_img.exists():
        res = predictor.predict_file(sample_img)
        assert "score" in res
        assert "verdict" in res
        assert "threshold" in res
        assert "model_used" in res
        assert "model_name" in res
        assert "model_selection_reason" in res
        assert 0.0 <= res["score"] <= 1.0
        assert 0.0 <= res["threshold"] <= 1.0
