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
    import pandas as pd

    summary = load_manifest_summary()
    manifest_path = ROOT_DIR / "data" / "splits" / "manifest_all.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        assert summary["total_files"] == len(df)
        assert summary["unique_sources"] == df["source_id"].nunique()
        assert summary["clean_count"] == int((df["label"] == 0).sum())
        assert summary["stego_count"] == int((df["label"] == 1).sum())
        assert summary["total_files"] == summary["clean_count"] + summary["stego_count"]
        assert summary["total_files"] == sum(summary["media_counts"].values())
        assert summary["total_files"] == sum(summary["split_counts"].values())
    else:
        assert summary["total_files"] == 0


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


def test_ui_tables_dedented_and_no_raw_html_code_blocks():
    """Verify that table HTML strings are dedented (no 4+ leading spaces) and contain required visible headers."""
    import json
    from app.ui.components import build_metrics_table_html, build_comparison_table_html

    sample_metrics = {
        "per_format": {
            "image": {"accuracy": 0.97, "roc_auc": 0.9966, "roc_auc_ci95": [0.990, 1.0], "recalls_at_fpr": {"recall_at_1pct_fpr": 0.98, "recall_at_5pct_fpr": 0.993, "recall_at_10pct_fpr": 0.997}, "sample_count": 600, "unique_sources": 300},
            "audio": {"accuracy": 0.50, "roc_auc": 0.4997, "roc_auc_ci95": [0.450, 0.550], "recalls_at_fpr": {"recall_at_1pct_fpr": 0.02, "recall_at_5pct_fpr": 0.049, "recall_at_10pct_fpr": 0.098}, "sample_count": 286, "unique_sources": 143},
            "video": {"accuracy": 0.5583, "roc_auc": 0.5715, "roc_auc_ci95": [0.480, 0.660], "recalls_at_fpr": {"recall_at_1pct_fpr": 0.08, "recall_at_5pct_fpr": 0.1667, "recall_at_10pct_fpr": 0.250}, "sample_count": 120, "unique_sources": 60},
        },
        "overall": {"accuracy": 0.7873, "roc_auc": 0.8756, "roc_auc_ci95": [0.852, 0.898], "recalls_at_fpr": {"recall_at_1pct_fpr": 0.503, "recall_at_5pct_fpr": 0.5388, "recall_at_10pct_fpr": 0.5944}, "sample_count": 1006, "unique_sources": 503}
    }

    m_html = build_metrics_table_html(sample_metrics, is_smoke=False)
    # Check that no line starts with 4+ spaces (which CommonMark treats as a pre/code block)
    for idx, line in enumerate(m_html.splitlines()):
        assert not line.startswith("    "), f"Line {idx} in metrics table HTML has 4+ leading spaces: {line!r}"
        assert not line.startswith("\t"), f"Line {idx} in metrics table HTML has tab indentation: {line!r}"

    # Verify visible table headers
    assert "<th>Format</th>" in m_html
    assert "<th>Accuracy</th>" in m_html
    assert "<th>ROC-AUC (95% CI)</th>" in m_html
    assert "<th>Recall @1% FPR</th>" in m_html
    assert "<th>Recall @5% FPR</th>" in m_html
    assert "<th>Recall @10% FPR</th>" in m_html
    assert "<th>Test files</th>" in m_html

    # Check side-by-side comparison table
    sample_comp = {
        "comparison": {
            "image": {
                "deep_model": {"roc_auc": 0.9966, "accuracy": 0.97, "recall_at_5pct_fpr": 0.9933},
                "handcrafted_mlp": {"roc_auc": 0.9309, "accuracy": 0.8783, "recall_at_5pct_fpr": 0.8067},
                "gradient_boosting": {"roc_auc": 0.9603, "accuracy": 0.9067, "recall_at_5pct_fpr": 0.8667}
            },
            "audio": {
                "deep_model": {"roc_auc": 0.4997, "accuracy": 0.50, "recall_at_5pct_fpr": 0.049},
                "handcrafted_mlp": {"roc_auc": 0.6205, "accuracy": 0.5629, "recall_at_5pct_fpr": 0.1329},
                "gradient_boosting": {"roc_auc": 0.8108, "accuracy": 0.6573, "recall_at_5pct_fpr": 0.3706}
            },
            "video": {
                "deep_model": {"roc_auc": 0.5715, "accuracy": 0.5583, "recall_at_5pct_fpr": 0.1667},
                "handcrafted_mlp": {"roc_auc": 0.9311, "accuracy": 0.8583, "recall_at_5pct_fpr": 0.9167},
                "gradient_boosting": {"roc_auc": 0.9632, "accuracy": 0.8917, "recall_at_5pct_fpr": 0.90}
            }
        }
    }
    c_html = build_comparison_table_html(sample_comp, is_smoke=False)
    for idx, line in enumerate(c_html.splitlines()):
        assert not line.startswith("    "), f"Line {idx} in comparison table HTML has 4+ leading spaces: {line!r}"

    assert "<th>Format</th>" in c_html
    assert "<th>Deep model</th>" in c_html
    assert "<th>Handcrafted MLP</th>" in c_html
    assert "<th>Gradient boosting</th>" in c_html
    assert "0.9966" in c_html
    assert "0.8108" in c_html


def test_smoke_badge_on_smoke_metrics():
    """Verify visible 'Smoke test, not real performance' badge appears when is_smoke=True."""
    from app.ui.components import build_metrics_table_html, build_comparison_table_html, render_format_card

    dummy_metrics = {"overall": {"accuracy": 0.52, "roc_auc": 0.51}}
    m_smoke = build_metrics_table_html(dummy_metrics, is_smoke=True)
    assert "Smoke test, not real performance" in m_smoke

    m_real = build_metrics_table_html(dummy_metrics, is_smoke=False)
    assert "Smoke test, not real performance" not in m_real

    c_smoke = build_comparison_table_html({}, is_smoke=True)
    assert "Smoke test, not real performance" in c_smoke

    card_smoke = render_format_card("Image", "IMAGE", {"roc_auc": 0.5, "accuracy": 0.5, "unique_sources": 50}, "auc", "acc", is_smoke=True)
    assert "Smoke test, not real performance" in card_smoke

    card_real = render_format_card("Image", "IMAGE", {"roc_auc": 0.9, "accuracy": 0.9, "unique_sources": 50}, "auc", "acc", is_smoke=False)
    assert "Smoke test, not real performance" not in card_real


def test_no_literal_bull_entities_in_ui_files():
    """Verify that &bull; entities are strictly purged from UI presentation code."""
    ui_files = [
        ROOT_DIR / "app" / "streamlit_app.py",
        ROOT_DIR / "app" / "ui" / "components.py",
        ROOT_DIR / "app" / "ui" / "styles.css",
    ]
    for fpath in ui_files:
        content = fpath.read_text(encoding="utf-8")
        assert "&bull;" not in content, f"Found literal '&bull;' in {fpath.name}"


def test_stat_cards_show_readable_lists_without_abbreviations():
    """Verify stat_card supports structured item rows and uses full readable labels (no Img/Aud/Trn/Tst)."""
    from app.ui.components import stat_card

    items = [
        ("Images", "4,000"),
        ("Audio", "2,000"),
        ("Video", "700"),
    ]
    subtext = "Images 4,000 / Audio 2,000 / Video 700"
    html_card = stat_card(title="Format Breakdown", value=subtext, subtext=subtext, items=items)

    assert "Images" in html_card
    assert "Audio" in html_card
    assert "Video" in html_card
    assert "4,000" in html_card
    assert "2,000" in html_card
    assert "700" in html_card
    import re
    assert not re.search(r'\b(Img|Aud|Trn|Tst)\b', html_card)
    assert "stat-items-list" in html_card


def test_active_checkpoint_rows_and_dynamic_fpr():
    """Verify active checkpoints per format show real filenames, thresholds, and dynamic clean FPR."""
    from app.ui.components import get_active_checkpoint_rows

    rows, target_fpr, cal_per_fmt = get_active_checkpoint_rows()
    assert len(rows) == 3
    formats = [r["format"] for r in rows]
    assert formats == ["Image", "Audio", "Video"]

    img_row = next(r for r in rows if r["format"] == "Image")
    assert "deep_model.pt" in img_row["model"]
    assert img_row["threshold"] < 0.20  # calibrated at ~0.0373

    aud_row = next(r for r in rows if r["format"] == "Audio")
    assert "GradientBoosting" in aud_row["model"] or "GBDT" in aud_row["model"]
    assert aud_row["threshold"] > 0.40  # calibrated at ~0.4978

    vid_row = next(r for r in rows if r["format"] == "Video")
    assert "GradientBoosting" in vid_row["model"] or "GBDT" in vid_row["model"]
    assert vid_row["threshold"] > 0.50  # calibrated at ~0.5855

    assert abs(target_fpr - 0.05) < 1e-4


def test_history_table_html_dedented_and_no_raw_html_code_blocks():
    """Verify that build_history_table_html lines are never indented with 4+ spaces and CommonMark renders them as HTML."""
    from markdown_it import MarkdownIt
    from app.ui.components import build_history_table_html

    sample_scans = [
        {"scan_id": "SCN-0003", "created_at": "2026-10-24 12:24:23", "filename": "img_src_0034_clean.png", "media_type": "image", "probability": 0.02, "ground_truth_label": 0, "is_correct": 1},
        {"scan_id": "SCN-0002", "created_at": "2026-10-24 12:24:02", "filename": "img_src_0004_stego_lsb_replacement_0.20.png", "media_type": "image", "probability": 0.99, "ground_truth_label": 1, "is_correct": 1},
        {"scan_id": "SCN-0001", "created_at": "2026-10-24 12:17:17", "filename": "stego_Media (1) (1).png", "media_type": "image", "probability": 0.98, "ground_truth_label": None, "is_correct": None},
    ]

    html_tbl = build_history_table_html(sample_scans, decision_threshold=0.5)

    # 1. No line starts with 4+ spaces or tabs
    for idx, line in enumerate(html_tbl.splitlines()):
        assert not line.startswith("    "), f"Line {idx} in history table HTML has 4+ leading spaces: {line!r}"
        assert not line.startswith("\t"), f"Line {idx} in history table HTML has tab indentation: {line!r}"

    # 2. Contains required table headers
    assert "<th>Scan ID</th>" in html_tbl
    assert "<th>Timestamp</th>" in html_tbl
    assert "<th>File Name</th>" in html_tbl
    assert "<th>Modality</th>" in html_tbl
    assert "<th>Calibrated Score</th>" in html_tbl
    assert "<th>Ground Truth Verification</th>" in html_tbl

    # 3. Streamlit's CommonMark parser does not produce code blocks
    md = MarkdownIt()
    rendered = md.render(html_tbl)
    assert "<pre>" not in rendered, "History table was erroneously converted to <pre> code block!"
    assert "<code>" not in rendered, "History table was erroneously converted to <code> block!"


def test_scan_result_card_html_never_contains_raw_code_blocks():
    """Verify that scan result header with verdict_badge never converts to CommonMark code block."""
    from markdown_it import MarkdownIt
    from app.ui.components import verdict_badge, dedent_html

    fname = "stego_Media (1) (1).png"
    mtype = "image"
    fmt = "png"
    sz_kb = 124.5
    model_name_disp = "VaultBreakerDeepNet"
    model_reason = "Best validation ROC-AUC (0.9998) vs handcrafted_mlp: 0.9573, gradient_boosting"

    badge_html = verdict_badge(True, "Suspected hidden data", is_unsupported=False)
    reason_span = f'<span style="margin-left: 8px; color:#E0B341; font-size:11px;">({model_reason})</span>'

    top1_html = dedent_html(f'''
    <div style="display:flex; align-items:center; gap:12px; margin-bottom: 8px;">
        {badge_html}
        <span style="font-size:16px; font-weight:600; color:var(--text-primary);">{fname}</span>
    </div>
    <div style="font-size:12px; color:var(--text-secondary);">
        Media Type: <strong style="color:var(--text-primary); text-transform:uppercase;">{mtype}</strong> ({fmt.upper()}) · Container Size: <strong style="color:var(--text-primary);">{sz_kb:.1f} KB</strong>
    </div>
    <div style="margin-top: 6px; font-size:12px; color:var(--text-secondary);">
        Model Selected: <strong style="color:#EDEDED;">{model_name_disp}</strong>
        {reason_span}
    </div>
    ''')

    # Check that no lines start with 4+ spaces
    for idx, line in enumerate(top1_html.splitlines()):
        assert not line.startswith("    "), f"Line {idx} in scan top1 HTML has 4+ leading spaces: {line!r}"

    # Render via markdown-it and ensure no pre/code tags
    md = MarkdownIt()
    rendered = md.render(top1_html)
    assert "<pre>" not in rendered, "Scan card was erroneously converted to <pre> code block!"
    assert "<code>" not in rendered, "Scan card was erroneously converted to <code> block!"


def test_all_components_dedented():
    """Verify all UI component library functions produce dedented HTML strings."""
    from app.ui.components import (
        glass_card,
        stat_card,
        verdict_badge,
        prob_gauge,
        ground_truth_strip,
        empty_state,
        render_entities_table,
        render_format_card,
    )

    components_outputs = [
        glass_card("<div>Content</div>", title="Title", subtitle="Sub"),
        stat_card("Title", "1,000", "subtext"),
        verdict_badge(True, "Stego"),
        verdict_badge(False, "Clean"),
        prob_gauge(0.75, 0.5),
        ground_truth_strip(1, "lsb_matching", 0.2, True),
        empty_state("Title", "Subtext"),
        render_entities_table([("lsb_flip_rate", 2.3), ("srm_kv_kurtosis", -1.2)]),
        render_format_card("Image", "IMG", {"roc_auc": 0.9, "accuracy": 0.95, "unique_sources": 50}, "auc", "acc"),
    ]

    for comp_html in components_outputs:
        for idx, line in enumerate(comp_html.splitlines()):
            assert not line.startswith("    "), f"Component HTML line {idx} starts with 4+ spaces: {line!r}"


def test_streamlit_app_uses_st_html_for_all_html_rendering():
    """Verify that streamlit_app.py uses st_html for all HTML rendering, avoiding CommonMark indentation bugs."""
    app_code = (ROOT_DIR / "app" / "streamlit_app.py").read_text()

    # Check that unsafe_allow_html=True only appears in st_html definition and CSS loading
    matches = [line.strip() for line in app_code.splitlines() if "unsafe_allow_html=True" in line]
    assert len(matches) <= 2, f"Found unexpected unsafe_allow_html calls outside st_html/CSS: {matches}"



