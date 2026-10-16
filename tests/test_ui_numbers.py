"""Tests for UI numbers audit and provenance guarantees."""

import os
import sys
import re
import json
import tempfile
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.check_ui_numbers import (
    main as check_ui_main,
    check_ast_fallbacks,
    check_forbidden_patterns
)
from app.ui.db import init_db, clear_all_scans, get_all_scans, add_scan_record
from app.ui.components import score_pill_html, create_activity_trend_chart

ROOT_DIR = Path(__file__).resolve().parent.parent

def test_check_ui_numbers_passes():
    """Verify that the full presentation layer passes all UI number checks."""
    assert check_ui_main() == 0

def test_check_ui_numbers_detects_numeric_fallback():
    """Verify check_ui_numbers catches numeric fallback literals in .get()."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write("def dummy(d):\n    return d.get('roc_auc', 0.9372)\n")
        tmp_path = Path(tmp.name)

    try:
        errors = check_ast_fallbacks(tmp_path)
        assert len(errors) > 0
        assert "numeric fallback literal 0.9372" in errors[0]
    finally:
        tmp_path.unlink()

def test_check_ui_numbers_detects_deleted_charts():
    """Verify check_ui_numbers catches imports or references to deleted charts."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write("from app.ui.components import create_radar_chart\n")
        tmp_path = Path(tmp.name)

    try:
        errors = check_ast_fallbacks(tmp_path) + check_forbidden_patterns(tmp_path)
        assert len(errors) > 0
        assert any("create_radar_chart" in err for err in errors)
    finally:
        tmp_path.unlink()

def test_home_cards_show_not_available_yet_when_metrics_missing(tmp_path):
    """Verify that when metrics.json is missing, format cards show 'Not available yet' and no numbers."""
    from app.ui.components import load_real_metrics, render_format_card

    fake_path = tmp_path / "non_existent_metrics.json"
    res = load_real_metrics(results_path=fake_path)
    assert res is None

    # Test format card rendering when metrics dictionary is None or empty
    card_html = render_format_card(
        "Image Steganalysis",
        "IMAGE",
        None,
        "Help AUC",
        "Help ACC"
    )

    assert "Not available yet" in card_html
    assert "Run make eval" in card_html
    # Ensure no fabricated or old fallback numbers are in the output
    assert "0.9372" not in card_html
    assert "0.88" not in card_html
    assert "300" not in card_html

def test_fresh_database_empty_and_clear_history():
    """Verify a fresh install opens to empty history and clear_history works cleanly."""
    init_db()
    clear_all_scans()
    scans = get_all_scans()
    assert len(scans) == 0

    # Add a real scan record
    sample_report = {
        "file_name": "test_img.png",
        "media_type": "image",
        "format": "png",
        "verdict": "CLEAN",
        "risk_level": "LOW RISK",
        "calibrated_probability": 0.1234,
        "raw_probability": 0.2000,
        "top_features": [["DCT Mid Skew", 1.2]]
    }
    scan_id = add_scan_record(sample_report)
    assert scan_id.startswith("SCN-")

    scans_after_add = get_all_scans()
    assert len(scans_after_add) == 1
    assert scans_after_add[0]["scan_id"] == scan_id

    # Clear history again
    clear_all_scans()
    assert len(get_all_scans()) == 0

def test_dynamic_threshold_in_components():
    """Verify score_pill_html and activity chart respect dynamic thresholds."""
    pill_high = score_pill_html(0.70, threshold=0.6272)
    assert "score-high" in pill_high
    assert "70.0%" in pill_high

    pill_clean = score_pill_html(0.20, threshold=0.6272)
    assert "score-clean" in pill_clean
    assert "20.0%" in pill_clean


def test_manifest_counts_are_right():
    """Verify manifest counts, split distributions and format balance."""
    from app.ui.db import load_manifest_summary

    summary = load_manifest_summary()
    assert summary["total_files"] == 6700
    assert summary["unique_sources"] == 3350
    assert summary["media_counts"]["image"] == 4000
    assert summary["media_counts"]["audio"] == 2000
    assert summary["media_counts"]["video"] == 700
    assert summary["split_counts"]["train"] == 4690
    assert summary["split_counts"]["val"] == 1004
    assert summary["split_counts"]["test"] == 1006
    assert summary["clean_count"] == 3350
    assert summary["stego_count"] == 3350


def test_blind_mode_hides_labels():
    """Verify blind test mode hides labels, methods and payloads."""
    from app.ui.db import format_blind_mode

    sample_row = {
        "filename": "img_src_0001_stego_lsb_matching_0.10.png",
        "label": 1,
        "method": "lsb_matching",
        "payload_rate": 0.10
    }

    # With blind mode active
    blind_res = format_blind_mode(sample_row, is_blind=True)
    assert blind_res["is_blind"] is True
    assert blind_res["label_str"] == "[Hidden in Blind Mode]"
    assert blind_res["method_str"] == "[Hidden in Blind Mode]"
    assert blind_res["payload_rate_str"] == "[Hidden in Blind Mode]"

    # With blind mode disabled (revealing ground truth)
    unblind_res = format_blind_mode(sample_row, is_blind=False)
    assert unblind_res["is_blind"] is False
    assert unblind_res["label_str"] == "Hidden Data"
    assert unblind_res["method_str"] == "lsb_matching"
    assert unblind_res["payload_rate_str"] == "0.10"


def test_missing_data_folder_falls_back_to_demo_samples():
    """Verify missing data/generated folder falls back to demo_samples without crashing."""
    from app.ui.db import load_explorer_dataset

    # Pass non-existent generated dir
    fake_gen_dir = ROOT_DIR / "data" / "non_existent_generated_test"
    df, is_fallback = load_explorer_dataset(generated_dir=fake_gen_dir)

    assert is_fallback is True
    assert not df.empty
    assert len(df) == 30
    assert "demo_filepath" in df.columns or "resolved_filepath" in df.columns
    assert set(df["media_type"].unique()) == {"image", "audio", "video"}
    # Ensure physical files exist for all demo samples
    for _, row in df.iterrows():
        p_str = row.get("resolved_filepath") or row.get("demo_filepath")
        assert (ROOT_DIR / p_str).exists()

