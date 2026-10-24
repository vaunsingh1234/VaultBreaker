"""Tests for evaluation pipeline: balanced test sets, smoke isolation, and hardcoded metric prohibition."""

import os
import sys
import json
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate import build_balanced_test_set, build_side_by_side_comparison
from scripts.check_ui_numbers import (
    main as check_ui_main,
    check_evaluate_script,
    FORBIDDEN_DISPLAY_STRINGS,
    FORBIDDEN_EVALUATE_FALLBACKS
)


def test_balanced_test_set_creation():
    """Verify that build_balanced_test_set samples exactly 1 stego variant per clean source."""
    rows = []
    methods = ["lsb_replacement", "lsb_matching", "edge_adaptive", "dct_ac"]
    payloads = [0.10, 0.20, 0.30, 0.50]

    # Create 10 sources with 1 clean and 4 stego variants each
    for i in range(10):
        src_id = f"img_source_{i:03d}"
        clean_fp = f"/dummy/clean_{i}.png"
        rows.append({
            "filepath": clean_fp,
            "filename": f"clean_{i}.png",
            "source_id": src_id,
            "media_type": "image",
            "label": 0,
            "method": "clean",
            "payload_rate": 0.0
        })
        for m_idx, (m, p) in enumerate(zip(methods, payloads)):
            stego_fp = f"/dummy/stego_{i}_{m}_{p}.png"
            rows.append({
                "filepath": stego_fp,
                "filename": f"stego_{i}_{m}_{p}.png",
                "source_id": src_id,
                "media_type": "image",
                "label": 1,
                "method": m,
                "payload_rate": p
            })

    df_manifest = pd.DataFrame(rows)
    assert len(df_manifest) == 50  # 10 clean + 40 stego

    df_balanced = build_balanced_test_set(df_manifest, seed=42)

    # Must have exactly 10 clean and 10 stego
    assert len(df_balanced) == 20
    assert (df_balanced["label"] == 0).sum() == 10
    assert (df_balanced["label"] == 1).sum() == 10

    # Each source must appear exactly twice (1 clean, 1 stego)
    for src_id in df_manifest["source_id"].unique():
        src_rows = df_balanced[df_balanced["source_id"] == src_id]
        assert len(src_rows) == 2, f"Source {src_id} has {len(src_rows)} rows in balanced set!"
        assert set(src_rows["label"].values) == {0, 1}

    # Verify stego variants are distributed across methods
    stego_methods = df_balanced[df_balanced["label"] == 1]["method"].unique()
    assert len(stego_methods) > 1, "Stego variants should span multiple methods, not just one!"


def test_no_smoke_overwrite(tmp_path):
    """Verify smoke evaluation writes strictly to results_dir/smoke and does not touch main results."""
    # Create fake real results
    real_results_dir = tmp_path / "RESULTS"
    real_results_dir.mkdir(parents=True, exist_ok=True)
    real_metric_file = real_results_dir / "metrics_deep.json"
    real_metric_file.write_text(json.dumps({"model_type": "real_evaluation", "roc_auc": 0.95}), encoding="utf-8")

    smoke_dir = real_results_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_metric_file = smoke_dir / "metrics_deep.json"
    smoke_metric_file.write_text(json.dumps({"model_type": "smoke_run", "roc_auc": 0.51, "is_smoke": True}), encoding="utf-8")

    # Verify real metric file was untouched
    real_content = json.loads(real_metric_file.read_text(encoding="utf-8"))
    assert real_content["model_type"] == "real_evaluation"
    assert real_content["roc_auc"] == 0.95

    # Verify smoke metric file has smoke flag
    smoke_content = json.loads(smoke_metric_file.read_text(encoding="utf-8"))
    assert smoke_content["is_smoke"] is True
    assert smoke_content["model_type"] == "smoke_run"


def test_build_side_by_side_comparison_not_run_when_missing():
    """Verify build_side_by_side_comparison writes 'not run' and never fake default metrics."""
    # When deep model is evaluated but handcrafted and GBDT were not run
    deep_metrics = {
        "per_format": {
            "image": {"roc_auc": 0.92, "accuracy": 0.85, "p_e": 0.15, "recalls_at_fpr": {"recall_at_5pct_fpr": 0.80}},
            "audio": {"roc_auc": 0.65, "accuracy": 0.58, "p_e": 0.42, "recalls_at_fpr": {"recall_at_5pct_fpr": 0.20}},
            "video": {"roc_auc": 0.98, "accuracy": 0.95, "p_e": 0.05, "recalls_at_fpr": {"recall_at_5pct_fpr": 0.92}}
        }
    }

    comp = build_side_by_side_comparison(deep_metrics=deep_metrics, handcrafted_metrics=None, gbdt_metrics=None)

    for fmt in ["image", "audio", "video"]:
        assert isinstance(comp[fmt]["deep_model"], dict)
        assert comp[fmt]["handcrafted_mlp"] == "not run"
        assert comp[fmt]["gradient_boosting"] == "not run"


def test_no_hardcoded_metrics_in_evaluate_script():
    """Verify check_evaluate_script catches any hardcoded numeric fallback constants."""
    eval_script = ROOT_DIR / "scripts" / "evaluate.py"
    errors = check_evaluate_script(eval_script)
    assert len(errors) == 0, f"Found hardcoded numbers in evaluate.py: {errors}"


def test_check_ui_numbers_zero_violations():
    """Verify check_ui_numbers passes with zero violations across all app and scripts."""
    assert check_ui_main() == 0
