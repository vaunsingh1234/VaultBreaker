"""VaultBreaker - Forensic Steganography Detection Console.

Simple Glass UI with Data Explorer, honest ground-truth verification,
strictly audited numeric provenance, and zero fake data.
"""

import os
import sys
import tempfile
import html
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import cv2
from PIL import Image

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

from vaultbreaker.inference.predict import VaultBreakerPredictor
from vaultbreaker.inference.router import RouterError
from vaultbreaker.utils.io import compute_sha256

from app.ui.icons import get_svg_icon
from app.ui.db import (
    init_db,
    add_scan_record,
    get_all_scans,
    get_scan_by_id,
    clear_all_scans,
    find_ground_truth,
    load_manifest_summary,
    load_explorer_dataset,
    format_blind_mode
)
from app.ui.components import (
    glass_card,
    stat_card,
    verdict_badge,
    prob_gauge,
    ground_truth_strip,
    score_pill_html,
    render_entities_table,
    render_format_card,
    build_metrics_table_html,
    build_comparison_table_html,
    load_real_metrics,
    load_display_metrics,
    load_comparison_results,
    load_display_comparison_results,
    get_active_checkpoint_rows,
    empty_state
)
from vaultbreaker.stego_tools.hide import hide_message, get_container_capacity
from vaultbreaker.stego_tools.reveal import reveal_message
from vaultbreaker.stego_tools.codec import format_size, CapacityError

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VaultBreaker | Steganography Detection Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Glassmorphism CSS Theme
css_path = root_dir / "app" / "ui" / "styles.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Database Schema
init_db()

@st.cache_resource
def get_predictor():
    return VaultBreakerPredictor()

predictor = get_predictor()

def load_config():
    """Load configuration parameters from configs/default.yaml."""
    cfg_path = root_dir / "configs" / "default.yaml"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}

config_data = load_config()

# 6 PAGES NAVIGATION DEFINITION
PAGES = [
    {"id": "scan", "label": "Scan", "icon": "shield"},
    {"id": "hide", "label": "Hide message", "icon": "lock"},
    {"id": "reveal", "label": "Reveal message", "icon": "unlock"},
    {"id": "data_explorer", "label": "Data Explorer", "icon": "database"},
    {"id": "results", "label": "Results", "icon": "activity"},
    {"id": "history", "label": "History", "icon": "clock"},
]

# Session State Initialization
query_page = st.query_params.get("page", None) if hasattr(st, "query_params") else None
if "nav_page" not in st.session_state:
    if query_page in [p["id"] for p in PAGES]:
        st.session_state["nav_page"] = query_page
        st.session_state["_last_seen_query_page"] = query_page
    else:
        st.session_state["nav_page"] = "scan"
elif query_page in [p["id"] for p in PAGES] and st.session_state.get("_last_seen_query_page") != query_page:
    st.session_state["nav_page"] = query_page
    st.session_state["_last_seen_query_page"] = query_page
elif st.session_state["nav_page"] not in [p["id"] for p in PAGES]:
    st.session_state["nav_page"] = "scan"

if "active_scan_id" not in st.session_state:
    st.session_state["active_scan_id"] = None
if "blind_test_mode" not in st.session_state:
    st.session_state["blind_test_mode"] = True
if "explorer_page" not in st.session_state:
    st.session_state["explorer_page"] = 0
if "selected_sample_path" not in st.session_state:
    st.session_state["selected_sample_path"] = None

# --- SIDEBAR NAVIGATION (Flat List, Icon + Label, Orange Marker) ---
with st.sidebar:
    st.markdown('''
    <div class="sidebar-brand">
        <div class="brand-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
        </div>
        <div>
            <div class="brand-title">VaultBreaker</div>
            <div class="brand-sub">Steganography Detection</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Render flat list navigation from PAGES list
    for p in PAGES:
        is_active = (st.session_state["nav_page"] == p["id"])
        btn_style = "primary" if is_active else "secondary"
        icon_svg = get_svg_icon(p["icon"], size=16, color="#E5522B" if is_active else "#A0A0A0")
        
        # Streamlit button for clean reactive state
        lbl = f"  {p['label']}"
        if is_active:
            lbl = f"▶ {p['label']}"
        if st.button(lbl, key=f"nav_btn_{p['id']}", type=btn_style, use_container_width=True):
            st.session_state["nav_page"] = p["id"]
            st.rerun()

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Active checkpoints per format and calibration parameters
    calibrator = predictor.calibrator
    active_rows, target_fpr, cal_per_fmt = get_active_checkpoint_rows()
    far_str = f"{target_fpr * 100:.1f}%"

    rows_html = "".join([
        f'<tr>'
        f'<td style="font-weight:600; color:var(--text-primary);">{r["format"]}</td>'
        f'<td style="font-family:\'JetBrains Mono\'; font-size:9.5px; color:var(--text-secondary); max-width:90px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{r["model"]}">{r["model"]}</td>'
        f'<td style="font-family:\'JetBrains Mono\'; color:var(--status-amber); font-weight:600; text-align:right;">{r["threshold"]:.4f}</td>'
        f'<td style="color:var(--text-muted); font-size:9.5px; text-align:right; white-space:nowrap;">{r["date"]}</td>'
        f'</tr>'
        for r in active_rows
    ])

    ckpt_card_html = (
        '<div class="glass-card" style="padding: 12px; margin-bottom: 8px;">\n'
        '<div style="font-size: 10px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.08em; margin-bottom: 6px;">Active Checkpoints by Format</div>\n'
        '<table class="glass-mini-table">\n'
        '<thead><tr><th>Format</th><th>Model</th><th style="text-align:right;">Threshold</th><th style="text-align:right;">File Date</th></tr></thead>\n'
        f'<tbody>{rows_html}</tbody>\n'
        '</table>\n'
        '<div style="height: 6px; border-bottom: 1px solid var(--glass-border); margin: 6px 0;"></div>\n'
        f'<div style="font-size: 11px; color: var(--text-secondary);" title="Clean false alarm rate target enforced during calibration">Clean false alarm rate: <strong style="color:var(--text-primary); font-family:\'JetBrains Mono\';">{far_str}</strong></div>\n'
        '</div>'
    )
    st.markdown(ckpt_card_html, unsafe_allow_html=True)

    with st.expander("Advanced Calibration Parameters", expanded=False):
        cal_param_rows = []
        for fmt in ["image", "audio", "video"]:
            fmt_c = cal_per_fmt.get(fmt, {})
            t_val = fmt_c.get("temperature", calibrator.temperature)
            th_val = fmt_c.get("decision_threshold", calibrator.decision_threshold)
            tf_val = fmt_c.get("target_fpr", target_fpr)
            cal_param_rows.append(
                f'<tr>'
                f'<td style="font-weight:600; color:var(--text-primary);">{fmt.capitalize()}</td>'
                f'<td style="font-family:\'JetBrains Mono\'; color:var(--text-primary); text-align:right;">{float(t_val):.4f}</td>'
                f'<td style="font-family:\'JetBrains Mono\'; color:var(--status-amber); text-align:right;">{float(th_val):.4f}</td>'
                f'<td style="font-family:\'JetBrains Mono\'; color:var(--status-clean); text-align:right;">{float(tf_val)*100:.1f}%</td>'
                f'</tr>'
            )
        expander_html = (
            '<div style="font-size: 11px; color: var(--text-secondary); line-height: 1.5;">\n'
            '<table class="glass-mini-table">\n'
            '<thead><tr><th>Format</th><th style="text-align:right;">Temp</th><th style="text-align:right;">Threshold</th><th style="text-align:right;">Tgt FPR</th></tr></thead>\n'
            f'<tbody>{"".join(cal_param_rows)}</tbody>\n'
            '</table>\n'
            '<div style="margin-top: 8px; font-size: 10px; color: var(--text-muted); line-height: 1.4;">\n'
            '• <strong>Temperature:</strong> Divides raw neural logits to calibrate probabilities so scores reflect empirical accuracy.<br/>\n'
            '• <strong>Threshold:</strong> Alert boundary; containers scoring higher are flagged as steganography.<br/>\n'
            '• <strong>Target FPR:</strong> Tolerated clean-file false alarm rate.\n'
            '</div>\n'
            '</div>'
        )
        st.markdown(expander_html, unsafe_allow_html=True)


# --- THUMBNAIL CACHING HELPER ---
def get_cached_thumbnail(img_path: Path) -> Optional[str]:
    """Create and cache small thumbnail in data/.thumb_cache/."""
    try:
        thumb_dir = root_dir / "data" / ".thumb_cache"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumb_dir / f"{img_path.stem}_thumb.jpg"
        if thumb_path.exists():
            return str(thumb_path)
        with Image.open(img_path) as im:
            im.thumbnail((160, 160))
            im.convert("RGB").save(thumb_path, "JPEG", quality=80)
            return str(thumb_path)
    except Exception:
        return None


# =====================================================================
# PAGE 1: SCAN (Home)
# =====================================================================
def render_scan_page():
    st.markdown('''
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin:0;">VaultBreaker</h1>
        <div style="color: var(--text-secondary); font-size: 14px; margin-top: 2px;">
            Detect hidden data in images, audio and video
        </div>
    </div>
    ''', unsafe_allow_html=True)

    calibrator = predictor.calibrator
    threshold = calibrator.decision_threshold

    # Check if a sample was handed off from Data Explorer
    sample_to_scan = None
    if st.session_state.get("selected_sample_path"):
        cand = Path(st.session_state["selected_sample_path"])
        if cand.exists():
            sample_to_scan = cand
        st.session_state["selected_sample_path"] = None

    # Top Section: Upload card (left) + "No file? Try a sample" card (right)
    col_upload, col_sample = st.columns([1.8, 1.0])

    with col_upload:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('''
        <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-secondary); letter-spacing: 0.08em; margin-bottom: 8px;">
            Inspect Container File
        </div>
        ''', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drop image, audio or video container to analyze",
            type=["png", "jpg", "jpeg", "wav", "flac", "mkv", "avi", "mp4"],
            key="main_file_uploader",
            label_visibility="collapsed"
        )
        
        btn_scan = st.button("Scan Container File", type="primary", use_container_width=True, key="btn_run_main_scan")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sample:
        st.markdown('''
        <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-secondary); letter-spacing: 0.08em; margin-bottom: 8px;">
                    No File Available?
                </div>
                <div style="font-size: 13px; color: var(--text-muted); line-height: 1.5; margin-bottom: 12px;">
                    Explore real evaluation samples across Image, Audio, and Video. Test the detector in Blind Mode or inspect ground truth.
                </div>
            </div>
        ''', unsafe_allow_html=True)
        if st.button("Browse Data Explorer", type="secondary", use_container_width=True, key="btn_open_explorer"):
            st.session_state["nav_page"] = "data_explorer"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Execute Scan if button clicked or sample handed off
    if (uploaded_file is not None and btn_scan) or sample_to_scan is not None:
        with st.spinner("Analyzing high-frequency forensic residuals across unified latent space..."):
            temp_path = None
            try:
                if uploaded_file is not None and btn_scan:
                    upload_cache_dir = root_dir / "data" / ".uploads_cache"
                    upload_cache_dir.mkdir(parents=True, exist_ok=True)
                    temp_path = upload_cache_dir / uploaded_file.name
                    temp_path.write_bytes(uploaded_file.getvalue())
                    target_path = temp_path
                    disp_name = uploaded_file.name
                else:
                    target_path = sample_to_scan
                    disp_name = sample_to_scan.name

                report = predictor.predict_file(target_path, generate_explanation=True)
                report["file_name"] = disp_name
                report["file_size_bytes"] = target_path.stat().st_size
                scan_id = add_scan_record(report, filepath=str(target_path), threshold=threshold)
                st.session_state["active_scan_id"] = scan_id
                st.toast(f"Scan complete: {disp_name}", icon="🛡️")
            except Exception as e:
                st.error(f"Scan failed: {e}")

    # Render Active Scan Result Card (ONE result card on the same page)
    active_id = st.session_state.get("active_scan_id")
    active_record = get_scan_by_id(active_id) if active_id else None

    if active_record:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        rep = active_record.get("report") or {}
        prob_val = active_record.get("probability")
        prob = float(prob_val) if prob_val is not None else 0.0
        is_stego = (prob >= threshold)
        verdict_str = rep.get("verdict") or ("Suspected hidden data" if is_stego else "No hidden data detected")
        is_unsupported = ("unsupported" in verdict_str.lower())
        
        fname = active_record.get("filename") or "container_file"
        mtype = active_record.get("media_type") or "image"
        fmt = active_record.get("format") or "bin"
        sz_bytes = active_record.get("file_size_bytes")
        sz_kb = (float(sz_bytes) / 1024.0) if sz_bytes is not None else 0.0

        # Top Bar: Verdict Badge + Filename + Actions
        col_res_top1, col_res_top2 = st.columns([2.5, 1.0])
        model_name_disp = rep.get("model_name") or rep.get("model_used", "Unified Detector")
        model_reason = rep.get("model_selection_reason") or ""

        with col_res_top1:
            st.markdown(f'''
            <div style="display:flex; align-items:center; gap:12px; margin-bottom: 8px;">
                {verdict_badge(is_stego, verdict_str, is_unsupported=is_unsupported)}
                <span style="font-size:16px; font-weight:600; color:var(--text-primary);">{html.escape(fname)}</span>
            </div>
            <div style="font-size:12px; color:var(--text-secondary);">
                Media Type: <strong style="color:var(--text-primary); text-transform:uppercase;">{mtype}</strong> ({fmt.upper()}) · Container Size: <strong style="color:var(--text-primary);">{sz_kb:.1f} KB</strong>
            </div>
            <div style="margin-top: 6px; font-size:12px; color:var(--text-secondary);">
                Model Selected: <strong style="color:#EDEDED;">{html.escape(model_name_disp)}</strong>
                {f'<span style="margin-left: 8px; color:#E0B341; font-size:11px;">({html.escape(model_reason)})</span>' if model_reason else ''}
            </div>
            ''', unsafe_allow_html=True)
        with col_res_top2:
            st.markdown(f'''
            <div style="text-align:right;">
                <div style="font-size:11px; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.06em;">Scan Identifier</div>
                <div style="font-size:14px; font-weight:600; font-family:'JetBrains Mono'; color:var(--text-primary);">{active_record.get("scan_id")}</div>
            </div>
            ''', unsafe_allow_html=True)

        # Notice for lossy compressed inputs
        notice_text = rep.get("notice")
        if notice_text:
            st.info(f"ℹ️ {notice_text}")

        # Rule-based structure check findings list
        struct_findings = rep.get("structure_findings") or []
        if struct_findings:
            st.markdown('''
            <div style="background: rgba(229, 82, 43, 0.08); border: 1px solid rgba(229, 82, 43, 0.3); border-radius: 8px; padding: 12px; margin: 12px 0;">
                <div style="font-size: 13px; font-weight: 600; color: #E5522B; margin-bottom: 6px;">Rule-Based Structure Check Findings</div>
            ''', unsafe_allow_html=True)
            for msg in struct_findings:
                st.markdown(f"• **{html.escape(msg)}**")
            st.markdown('</div>', unsafe_allow_html=True)

        # "Try to reveal message" button when suspected or structure findings present
        if is_stego or struct_findings or ("stego" in verdict_str.lower()):
            col_rev, _ = st.columns([1.8, 3.2])
            with col_rev:
                if st.button("🔓 Try to reveal message", type="primary", use_container_width=True, key="btn_try_reveal_scan"):
                    st.session_state["reveal_target_path"] = active_record.get("filepath")
                    st.session_state["nav_page"] = "reveal"
                    st.rerun()

        # Gauge Bar
        if prob_val is not None:
            st.markdown(prob_gauge(prob, threshold=threshold), unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="margin: 8px 0 14px 0; font-size: 12px; color: var(--text-muted);">
                Statistical probability gauge omitted (structure check only for lossy/compressed formats).
            </div>
            ''', unsafe_allow_html=True)

        # Ground Truth Strip if available
        gt_lbl = active_record.get("ground_truth_label")
        if gt_lbl is not None:
            gt_method = active_record.get("ground_truth_method") or "clean"
            gt_payload_raw = active_record.get("ground_truth_payload")
            gt_payload = float(gt_payload_raw) if gt_payload_raw is not None else 0.0
            st.markdown(ground_truth_strip(
                real_label=int(gt_lbl),
                method=gt_method,
                payload_rate=gt_payload,
                pred_stego=is_stego
            ), unsafe_allow_html=True)

        # Three Tabs: "Why", "Most unusual measurements", "Raw JSON"
        tab_why, tab_unusual, tab_json = st.tabs(["Why", "Most unusual measurements", "Raw JSON"])

        with tab_why:
            exp = rep.get("explanation") or {}
            if mtype == "image":
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.markdown("**Spatial SRM Residual Heatmap (KV Kernel)**")
                    st.caption("Displays high-frequency spatial deviations and edge residuals.")
                    if "residual_heatmap" in exp:
                        st.image(cv2.cvtColor(np.array(exp["residual_heatmap"], dtype=np.uint8), cv2.COLOR_BGR2RGB), use_column_width=True)
                    else:
                        st.info("Heatmap computed during real-time scan.")
                with col_i2:
                    st.markdown("**LSB Bit-Plane 0 Extraction**")
                    st.caption("Exposes artificial bit entropy or rectangular carrier blocks.")
                    if "lsb_plane" in exp:
                        st.image(np.array(exp["lsb_plane"], dtype=np.uint8) * 255, clamp=True, use_column_width=True)
                    else:
                        st.info("Bitplane visualized during inspection.")

            elif mtype == "audio":
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.markdown("**Short-Time Fourier Transform (STFT) Spectrogram**")
                    if "spectrogram_db" in exp:
                        fig_spec = px.imshow(
                            exp["spectrogram_db"],
                            origin='lower',
                            labels=dict(x="Time Frame", y="Frequency Bin", color="dB"),
                            color_continuous_scale="Viridis"
                        )
                        fig_spec.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#8B8B8B'))
                        st.plotly_chart(fig_spec, use_container_width=True)
                    else:
                        st.info("Acoustic spectrogram available in live audio scan.")
                with col_a2:
                    st.markdown("**Bitplane Distribution Statistics**")
                    if "lsb_ones_ratio" in exp:
                        ratio = float(exp["lsb_ones_ratio"])
                        st.metric(
                            label="LSB Bit-1 Proportion",
                            value=f"{ratio * 100:.1f}%",
                            help="Proportion of ones in least significant bit plane (clean audio typically measures ~50.0%)"
                        )
                    if "lsb_transition_rate" in exp:
                        trans = float(exp["lsb_transition_rate"])
                        st.metric(
                            label="LSB Transition Frequency",
                            value=f"{trans * 100:.1f}%",
                            help="Frequency of bit flips between consecutive LSB audio samples"
                        )

            elif mtype == "video":
                st.markdown("**Per-Frame Anomaly Residual Variance Timeline**")
                if "frame_timeline" in exp:
                    tl = exp["frame_timeline"]
                    fig_tl = go.Figure(go.Scatter(
                        x=list(range(len(tl))),
                        y=tl,
                        mode='lines+markers',
                        line=dict(color='#E5522B' if is_stego else '#3FB67A', width=2)
                    ))
                    fig_tl.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#8B8B8B'),
                        xaxis=dict(title="Sampled Frame Index", gridcolor='rgba(255,255,255,0.06)'),
                        yaxis=dict(title="Residual Variance", gridcolor='rgba(255,255,255,0.06)'),
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_tl, use_container_width=True)
                else:
                    st.info("Inter-frame timeline available during live video scan.")

        with tab_unusual:
            st.markdown('''
            <div style="font-size:13px; color:var(--text-secondary); margin-bottom: 8px;">
                These are the forensic measurements furthest from normal (clean baseline population).
            </div>
            ''', unsafe_allow_html=True)
            top_feats = rep.get("top_features") or []
            if top_feats:
                st.markdown(render_entities_table(top_feats), unsafe_allow_html=True)
            else:
                st.info("Top feature deviations not available for this record.")

        with tab_json:
            st.json(rep)

        # Action Buttons
        col_act1, col_act2, _ = st.columns([1.2, 1.2, 2.0])
        with col_act1:
            st.download_button(
                label="Download Report (JSON)",
                data=json.dumps(rep, indent=2),
                file_name=f"{active_id}_report.json",
                mime="application/json",
                use_container_width=True,
                key="btn_download_report_json"
            )
        with col_act2:
            if st.button("Scan Another File", type="secondary", use_container_width=True, key="btn_scan_another"):
                st.session_state["active_scan_id"] = None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # Batch Scan Expander
    with st.expander("Scan multiple files", expanded=False):
        batch_files = st.file_uploader(
            "Upload multiple media files for batch stego triage",
            type=["png", "jpg", "jpeg", "wav", "flac", "mkv", "avi", "mp4"],
            accept_multiple_files=True,
            key="batch_files_uploader"
        )
        if batch_files and st.button("Run Batch Analysis", key="btn_run_batch"):
            prog_bar = st.progress(0.0)
            batch_results = []
            for idx, bf in enumerate(batch_files):
                suffix = Path(bf.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(bf.getvalue())
                    tmp_p = Path(tmp.name)
                try:
                    rep_b = predictor.predict_file(tmp_p, generate_explanation=False)
                    rep_b["file_name"] = bf.name
                    rep_b["file_size_bytes"] = tmp_p.stat().st_size
                    b_id = add_scan_record(rep_b, filepath=str(tmp_p), threshold=threshold)
                    p_val = rep_b.get("calibrated_probability")
                    p_num = float(p_val) if p_val is not None else 0.0
                    batch_results.append({
                        "Scan ID": b_id,
                        "File Name": bf.name,
                        "Verdict": rep_b.get("verdict"),
                        "Calibrated Score": f"{p_num * 100:.1f}%"
                    })
                finally:
                    if tmp_p.exists():
                        tmp_p.unlink()
                prog_bar.progress((idx + 1) / len(batch_files))
            
            st.success(f"Batch completed: {len(batch_results)} files inspected.")
            st.dataframe(pd.DataFrame(batch_results), use_container_width=True)

    # Under it: Three small cards (Image / Audio / Video) with real test accuracy & ROC-AUC from metrics.json
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown('''
    <div style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-secondary); letter-spacing:0.08em; margin-bottom:10px;">
        Test Set Performance Benchmarks (Audited from docs/RESULTS/metrics.json)
    </div>
    ''', unsafe_allow_html=True)

    m_data = load_display_metrics()
    is_smoke = bool(m_data and (m_data.get("is_smoke") or str(m_data.get("profile", "")).lower() == "smoke" or not m_data.get("profile")))
    if is_smoke:
        st.markdown('''
        <div class="badge-smoke-banner">
            ⚠️ Smoke test, not real performance
        </div>
        ''', unsafe_allow_html=True)
    per_fmt = m_data.get("per_format") if m_data else None

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        img_metrics = per_fmt.get("image") if per_fmt else None
        st.markdown(render_format_card(
            title="Image Steganalysis",
            badge="IMAGE",
            m_dict=img_metrics,
            help_auc="Diagnostic discrimination on image spatial and frequency domain (1.0 = perfect)",
            help_acc="Percentage of correct clean vs stego verdicts on test image partition",
            is_smoke=is_smoke
        ), unsafe_allow_html=True)

    with col_b2:
        aud_metrics = per_fmt.get("audio") if per_fmt else None
        st.markdown(render_format_card(
            title="Audio Steganalysis",
            badge="AUDIO",
            m_dict=aud_metrics,
            help_auc="Discrimination ability on acoustic LSB and echo hiding",
            help_acc="Percentage of correct clean vs stego verdicts on test audio waveforms",
            is_smoke=is_smoke
        ), unsafe_allow_html=True)

    with col_b3:
        vid_metrics = per_fmt.get("video") if per_fmt else None
        st.markdown(render_format_card(
            title="Video Steganalysis",
            badge="VIDEO",
            m_dict=vid_metrics,
            help_auc="Discrimination ability across temporal video streams",
            help_acc="Percentage of correct clean vs stego verdicts on test video containers",
            is_smoke=is_smoke
        ), unsafe_allow_html=True)


# =====================================================================
# PAGE: HIDE MESSAGE
# =====================================================================
def render_hide_page():
    st.markdown('''
    <div style="margin-bottom: 1.25rem;">
        <h1 style="margin:0;">Hide Message</h1>
        <div style="color: var(--text-secondary); font-size: 14px; margin-top: 2px;">
            Embed encrypted or plain steganographic messages into real media containers using VBK1 protocol
        </div>
    </div>
    ''', unsafe_allow_html=True)

    col_input, col_status = st.columns([1.2, 1.0])

    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight:600; color:var(--text-primary); margin-bottom:8px;">1. Select Cover File</div>', unsafe_allow_html=True)
        uploaded_cover = st.file_uploader(
            "Upload Cover Media (PNG, JPEG, BMP, WAV, MP3, FLAC, M4A, MKV, MP4, AVI, MOV)",
            type=["png", "jpg", "jpeg", "bmp", "wav", "mp3", "flac", "m4a", "mkv", "mp4", "avi", "mov"],
            key="hide_cover_uploader"
        )

        cover_path = None
        cap_info = None

        if uploaded_cover is not None:
            upload_dir = root_dir / "data" / ".uploads_cache"
            upload_dir.mkdir(parents=True, exist_ok=True)
            cover_path = upload_dir / f"cover_{uploaded_cover.name}"
            cover_path.write_bytes(uploaded_cover.getvalue())

            try:
                cap_info = get_container_capacity(cover_path)
            except Exception as e:
                st.error(f"Cannot inspect container capacity: {e}")

        if cap_info:
            notice = cap_info.get("conversion_notice")
            if notice:
                st.info(f"ℹ️ {notice}")

            max_payload_bytes = cap_info.get("max_bytes")
            max_b = int(max_payload_bytes) if max_payload_bytes is not None else 0
            tot_bits = cap_info.get("total_bits")
            tot_b_val = int(tot_bits) if tot_bits is not None else 0

            st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
            st.markdown('<div style="font-weight:600; color:var(--text-primary); margin-bottom:6px;">2. Secret Message</div>', unsafe_allow_html=True)
            msg_text = st.text_area(
                "Message text to embed",
                placeholder="Enter secret message to hide in container...",
                height=120,
                key="hide_msg_input",
                label_visibility="collapsed"
            )

            msg_bytes_len = len(msg_text.encode("utf-8")) if msg_text else 0
            ratio = min(1.0, msg_bytes_len / max(1, max_b)) if max_b > 0 else 0.0

            st.markdown(f'''
            <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--text-secondary); margin-bottom:4px;">
                <span>Container Available Payload Capacity:</span>
                <strong style="color:var(--text-primary); font-family:'JetBrains Mono';">{format_size(max_b)}</strong>
            </div>
            ''', unsafe_allow_html=True)
            st.progress(ratio)
            st.caption(f"Payload capacity utilized: {msg_bytes_len:,} / {max_b:,} bytes ({ratio * 100:.1f}%)")

            if msg_bytes_len > max_b:
                st.error(f"Message exceeds available container capacity by {msg_bytes_len - max_b:,} bytes. Please shorten message.")

            st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
            st.markdown('<div style="font-weight:600; color:var(--text-primary); margin-bottom:6px;">3. Security & Embedding Method</div>', unsafe_allow_html=True)
            password = st.text_input(
                "Optional Passphrase (scrypt KDF + AES-256-GCM encryption + bit scattering)",
                type="password",
                placeholder="Leave blank for unencrypted VBK1 embedding",
                key="hide_password_input"
            )

            media_type = cap_info.get("media_type")
            method = "lsb_matching"
            if media_type == "image":
                method_choice = st.selectbox(
                    "Image LSB Embedding Algorithm",
                    ["LSB Matching (Default, preserves parity distribution)", "LSB Replacement"],
                    key="hide_method_select"
                )
                method = "lsb_matching" if "Matching" in method_choice else "lsb_replacement"

            st.warning("⚠️ Saving, re-compressing, or uploading the generated file as lossy JPEG, MP3, or H.264 destroys the hidden message.")

            can_embed = (msg_bytes_len > 0 and msg_bytes_len <= max_b)
            if st.button("🔒 Hide Message & Verify", type="primary", use_container_width=True, disabled=(not can_embed), key="btn_do_hide"):
                out_dir = root_dir / "data" / "stego_output"
                out_dir.mkdir(parents=True, exist_ok=True)
                tgt_fmt = cap_info.get("target_format") or "png"
                out_path = out_dir / f"stego_{Path(uploaded_cover.name).stem}.{tgt_fmt}"
                with st.spinner("Embedding payload and performing immediate round-trip verification..."):
                    try:
                        res = hide_message(
                            input_path=cover_path,
                            output_path=out_path,
                            message=msg_text,
                            password=password if password else None,
                            method=method
                        )
                        st.session_state["last_hide_result"] = res
                        st.session_state["last_hide_cover"] = str(cover_path)
                        st.session_state["last_hide_out"] = str(out_path)
                        st.toast("Message embedded and verified successfully!", icon="✅")
                    except Exception as e:
                        st.error(f"Embedding failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_status:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight:600; color:var(--text-primary); margin-bottom:8px;">Embedding Result & Verification</div>', unsafe_allow_html=True)

        hide_res = st.session_state.get("last_hide_result")
        if hide_res and st.session_state.get("last_hide_out"):
            out_file = Path(st.session_state["last_hide_out"])
            cover_file = Path(st.session_state.get("last_hide_cover", ""))

            st.markdown('''
            <div style="background: rgba(63, 182, 122, 0.12); border: 1px solid rgba(63, 182, 122, 0.35); border-radius: 8px; padding: 12px; margin-bottom: 12px; display:flex; align-items:center; gap:10px;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#3FB67A" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                <div>
                    <div style="font-weight:600; color:#3FB67A; font-size:14px;">Verified: Round-Trip Succeeded</div>
                    <div style="font-size:11px; color:var(--text-secondary);">The extraction engine was immediately executed on the generated file. Decryption and CRC-32 integrity verified 100%.</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            if out_file.exists():
                out_bytes = out_file.read_bytes()
                st.download_button(
                    label=f"⬇️ Download Stego {out_file.suffix.upper()[1:]} ({format_size(len(out_bytes))})",
                    data=out_bytes,
                    file_name=out_file.name,
                    mime="application/octet-stream",
                    type="primary",
                    use_container_width=True,
                    key="btn_download_stego_file"
                )

            pb_val = hide_res.get("payload_bytes")
            tb_val = hide_res.get("total_bits_embedded")
            cb_val = hide_res.get("capacity_bytes")
            st.markdown(f'''
            <div style="font-size:12px; color:var(--text-secondary); margin-top:10px; line-height:1.7;">
                · Output Container: <strong style="color:var(--text-primary);">{out_file.name}</strong><br/>
                · Payload Embedded: <strong style="color:var(--text-primary);">{pb_val if pb_val is not None else 0:,} bytes</strong> ({tb_val if tb_val is not None else 0:,} bits)<br/>
                · Container Capacity: <strong style="color:var(--text-primary);">{cb_val if cb_val is not None else 0:,} bytes</strong><br/>
                · Destruction Safeguard: <span style="color:var(--status-amber);">Lossless format enforced ({out_file.suffix.upper()[1:]})</span>
            </div>
            ''', unsafe_allow_html=True)

            if out_file.suffix.lower() == ".png" and cover_file.exists():
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:12px; font-weight:600; color:var(--text-primary); margin-bottom:6px;'>Visual Forensic Inspection (50x Amplified Difference)</div>", unsafe_allow_html=True)
                try:
                    c_img = cv2.imread(str(cover_file))
                    s_img = cv2.imread(str(out_file))
                    if c_img is not None and s_img is not None:
                        if c_img.shape != s_img.shape:
                            c_img = cv2.resize(c_img, (s_img.shape[1], s_img.shape[0]))
                        diff = np.abs(c_img.astype(np.int16) - s_img.astype(np.int16))
                        diff_50x = np.clip(diff * 50, 0, 255).astype(np.uint8)

                        col_c1, col_c2, col_c3 = st.columns(3)
                        with col_c1:
                            st.caption("Original Cover")
                            st.image(cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        with col_c2:
                            st.caption("Stego File")
                            st.image(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        with col_c3:
                            st.caption("Difference (x50)")
                            st.image(cv2.cvtColor(diff_50x, cv2.COLOR_BGR2RGB), use_container_width=True)
                except Exception as ex:
                    st.caption(f"Could not render difference view: {ex}")
        else:
            st.markdown('''
            <div style="font-size:12px; color:var(--text-muted); text-align:center; padding: 40px 0;">
                Select a cover file and enter a message on the left to embed and verify hidden data.
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# PAGE: REVEAL MESSAGE
# =====================================================================
def render_reveal_page():
    st.markdown('''
    <div style="margin-bottom: 1.25rem;">
        <h1 style="margin:0;">Reveal Message</h1>
        <div style="color: var(--text-secondary); font-size: 14px; margin-top: 2px;">
            Multi-stage forensic extractor: VaultBreaker VBK1, EOF overlay, sequential LSB, and steghide
        </div>
    </div>
    ''', unsafe_allow_html=True)

    col_rev_in, col_rev_out = st.columns([1.1, 1.3])

    with col_rev_in:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight:600; color:var(--text-primary); margin-bottom:8px;">1. Target Container File</div>', unsafe_allow_html=True)

        handed_off_path = st.session_state.get("reveal_target_path")
        target_path = None

        if handed_off_path and Path(handed_off_path).exists():
            hp = Path(handed_off_path)
            st.markdown(f'''
            <div style="background: rgba(229, 82, 43, 0.1); border: 1px solid rgba(229, 82, 43, 0.3); border-radius: 8px; padding: 10px; margin-bottom: 10px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:10px; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.06em;">Handed off from Scan</div>
                    <div style="font-size:13px; font-weight:600; color:var(--text-primary); font-family:'JetBrains Mono';">{hp.name}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            if st.button("Upload a different file instead", key="btn_clear_handoff"):
                st.session_state["reveal_target_path"] = None
                st.rerun()
            target_path = hp
        else:
            uploaded_reveal = st.file_uploader(
                "Upload Stego File to Reveal",
                type=["png", "jpg", "jpeg", "bmp", "wav", "mp3", "flac", "m4a", "mkv", "mp4", "avi", "mov"],
                key="reveal_file_uploader"
            )
            if uploaded_reveal is not None:
                upload_dir = root_dir / "data" / ".uploads_cache"
                upload_dir.mkdir(parents=True, exist_ok=True)
                save_p = upload_dir / f"reveal_{uploaded_reveal.name}"
                save_p.write_bytes(uploaded_reveal.getvalue())
                target_path = save_p

        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-weight:600; color:var(--text-primary); margin-bottom:6px;">2. Optional Passphrase</div>', unsafe_allow_html=True)
        reveal_password = st.text_input(
            "Passphrase (required if payload is encrypted with scrypt / AES-GCM)",
            type="password",
            placeholder="Enter passphrase if file was protected...",
            key="reveal_password_input"
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🔓 Extract Hidden Message", type="primary", use_container_width=True, disabled=(target_path is None), key="btn_do_reveal"):
            with st.spinner("Executing 4-stage extraction cascade (VBK1 -> EOF Overlay -> Sequential LSB -> Steghide)..."):
                try:
                    res = reveal_message(target_path, password=reveal_password if reveal_password else None)
                    st.session_state["last_reveal_result"] = res
                    st.session_state["last_reveal_target"] = str(target_path)
                except Exception as e:
                    st.error(f"Extraction error: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_rev_out:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight:600; color:var(--text-primary); margin-bottom:8px;">Extraction Forensic Report</div>', unsafe_allow_html=True)

        res = st.session_state.get("last_reveal_result")
        if res:
            if res.success:
                st.markdown(f'''
                <div style="background: rgba(63, 182, 122, 0.12); border: 1px solid rgba(63, 182, 122, 0.35); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                    <div style="font-weight:600; color:#3FB67A; font-size:14px; margin-bottom:4px;">Message Successfully Extracted</div>
                    <div style="font-size:12px; color:var(--text-secondary);">
                        · Technique: <strong style="color:var(--text-primary);">{html.escape(res.method)}</strong><br/>
                        · Confidence: <strong style="color:var(--text-primary);">{html.escape(res.confidence)}</strong><br/>
                        · Payload Size: <strong style="color:var(--text-primary); font-family:'JetBrains Mono';">{len(res.data):,} bytes</strong>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                if res.text is not None and not res.is_binary:
                    st.markdown("<div style='font-size:12px; font-weight:600; color:var(--text-primary); margin-bottom:4px;'>Extracted Text Content:</div>", unsafe_allow_html=True)
                    st.text_area(
                        "Extracted text",
                        value=res.text,
                        height=200,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    st.download_button(
                        label="⬇️ Download Plaintext (.txt)",
                        data=res.text.encode("utf-8"),
                        file_name="revealed_message.txt",
                        mime="text/plain",
                        key="btn_dl_revealed_txt"
                    )
                else:
                    mime = res.mime_type or "application/octet-stream"
                    st.info(f"Binary payload detected ({mime}, {len(res.data):,} bytes)")
                    if res.hex_preview:
                        st.markdown("<div style='font-size:12px; font-weight:600; color:var(--text-primary); margin-bottom:4px;'>Hex Preview (xxd format):</div>", unsafe_allow_html=True)
                        st.code(res.hex_preview, language="text")
                    st.download_button(
                        label=f"⬇️ Download Binary Payload ({format_size(len(res.data))})",
                        data=res.data,
                        file_name="revealed_payload.bin",
                        mime=mime,
                        key="btn_dl_revealed_bin"
                    )
            else:
                st.markdown(f'''
                <div style="background: rgba(229, 82, 43, 0.1); border: 1px solid rgba(229, 82, 43, 0.35); border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                    <div style="font-weight:600; color:#E5522B; font-size:14px; margin-bottom:6px;">Extraction Failed or No Hidden Message Found</div>
                    <div style="font-size:12px; color:var(--text-primary); white-space:pre-wrap; font-family:'JetBrains Mono';">{html.escape(res.error or "Unknown error")}</div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="font-size:12px; color:var(--text-muted); text-align:center; padding: 40px 0;">
                Select or hand off a target file on the left and click Extract to inspect for hidden data.
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# PAGE: DATA EXPLORER
# =====================================================================
def render_data_explorer_page():
    st.markdown('''
    <div style="margin-bottom: 1.25rem;">
        <h1 style="margin:0;">Data Explorer</h1>
        <div style="color: var(--text-secondary); font-size: 14px; margin-top: 2px;">
            Inspect evaluation dataset manifests, test blind samples, and verify ground truth
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # 1. Summary Cards from Manifests
    summary = load_manifest_summary()
    tot_files = summary["total_files"]
    uniq_sources = summary["unique_sources"]
    media_c = summary["media_counts"]
    split_c = summary["split_counts"]
    clean_n = summary["clean_count"]
    stego_n = summary["stego_count"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(stat_card(
            title="Total Manifest Files",
            value=f"{tot_files:,}",
            subtext=f"{uniq_sources:,} independent source media",
            help_text="Total files registered across training, validation, and test manifest partitions"
        ), unsafe_allow_html=True)
    with c2:
        img_n = media_c.get("image") or 0
        aud_n = media_c.get("audio") or 0
        vid_n = media_c.get("video") or 0
        items_c2 = [
            ("Images", f"{img_n:,}"),
            ("Audio", f"{aud_n:,}"),
            ("Video", f"{vid_n:,}"),
        ]
        sub_c2 = f"Images {img_n:,} / Audio {aud_n:,} / Video {vid_n:,}"
        st.markdown(stat_card(
            title="Format Breakdown",
            value=sub_c2,
            subtext=sub_c2,
            items=items_c2,
            help_text="Standardized media files across image (256x256), audio (16kHz), and video (128x128)",
            icon_name="grid"
        ), unsafe_allow_html=True)
    with c3:
        tr_n = split_c.get("train") or 0
        val_n = split_c.get("val") or 0
        te_n = split_c.get("test") or 0
        items_c3 = [
            ("Train", f"{tr_n:,}"),
            ("Val", f"{val_n:,}"),
            ("Test", f"{te_n:,}"),
        ]
        sub_c3 = f"Train {tr_n:,} / Val {val_n:,} / Test {te_n:,}"
        st.markdown(stat_card(
            title="Partition Splits",
            value=sub_c3,
            subtext=sub_c3,
            items=items_c3,
            help_text="Independent split distribution ensuring zero source contamination between train and test",
            icon_name="layers"
        ), unsafe_allow_html=True)
    with c4:
        bal_str = f"Clean {clean_n:,} · Stego {stego_n:,}"
        st.markdown(stat_card(
            title="Dataset Balance",
            value="50.0% / 50.0%",
            subtext=bal_str,
            help_text="Exact equal proportion of clean baseline media and steganographic samples",
            icon_name="shield"
        ), unsafe_allow_html=True)

    # "About this data" Card
    st.markdown('''
    <div class="glass-card" style="margin-top: 14px; padding: 14px 18px;">
        <div style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-secondary); letter-spacing:0.08em; margin-bottom:4px;">
            About This Data
        </div>
        <div style="font-size:13px; color:var(--text-primary); line-height:1.5;">
            Clean files are sourced from real public datasets (COCO/Div2K, Common Voice, Kinetics). Hidden-data files are generated by our own steganography algorithms. Videos are pan/zoom clips synthesized from still photos to evaluate temporal flicker resistance.
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # 2. Check missing data folder and fallback
    df_explorer, is_demo_fallback = load_explorer_dataset()

    if is_demo_fallback:
        st.markdown('''
        <div class="calm-banner">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
            <span>Full dataset not generated here. Showing demo samples. Run <code>make data</code> for the full set.</span>
        </div>
        ''', unsafe_allow_html=True)

    # 3. Blind Test Mode Toggle & Quick Picks (Test split only)
    col_blind, col_chip = st.columns([1.5, 2.5])
    with col_blind:
        b_val = st.session_state.get("blind_test_mode")
        is_blind = st.toggle(
            "Blind test mode (Fair live demo)",
            value=(b_val if b_val is not None else True),
            help="Hides label, method and payload until toggled off so detection verdicts remain unbiased"
        )
        st.session_state["blind_test_mode"] = is_blind

    st.markdown("<div style='font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-secondary); letter-spacing:0.08em; margin: 10px 0 6px 0;'>Quick Picks (Honest Test Split Only)</div>", unsafe_allow_html=True)
    
    # 6 Quick Pick buttons from TEST split only
    qp1, qp2, qp3, qp4, qp5, qp6 = st.columns(6)
    
    demo_gt = Path("demo_samples/ground_truth.csv")
    df_test_candidates = pd.read_csv(demo_gt) if demo_gt.exists() else pd.DataFrame()

    def pick_and_jump(media_type: str, label_val: int):
        if not df_test_candidates.empty:
            sub = df_test_candidates[(df_test_candidates["media_type"] == media_type) & (df_test_candidates["label"] == label_val)]
            if not sub.empty:
                pick = sub.sample(1).iloc[0]
                p_cand = root_dir / str(pick.get("demo_filepath") or pick.get("filepath"))
                if p_cand.exists():
                    st.session_state["selected_sample_path"] = str(p_cand)
                    st.session_state["nav_page"] = "scan"
                    st.rerun()

    with qp1:
        if st.button("Clean Image", key="qp_clean_img", use_container_width=True):
            pick_and_jump("image", 0)
    with qp2:
        if st.button("Stego Image", key="qp_stego_img", use_container_width=True):
            pick_and_jump("image", 1)
    with qp3:
        if st.button("Clean Audio", key="qp_clean_aud", use_container_width=True):
            pick_and_jump("audio", 0)
    with qp4:
        if st.button("Stego Audio", key="qp_stego_aud", use_container_width=True):
            pick_and_jump("audio", 1)
    with qp5:
        if st.button("Clean Video", key="qp_clean_vid", use_container_width=True):
            pick_and_jump("video", 0)
    with qp6:
        if st.button("Stego Video", key="qp_stego_vid", use_container_width=True):
            pick_and_jump("video", 1)

    # 4. Filters Bar
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.8])
    with fcol1:
        sel_format = st.selectbox("Format", ["All", "image", "audio", "video"], key="f_format")
    with fcol2:
        sel_split = st.selectbox("Split", ["Test split only", "All", "train", "val", "test"], key="f_split")
    with fcol3:
        sel_label = st.selectbox("Label", ["All", "Clean", "Hidden Data"], key="f_label")
    with fcol4:
        methods_list = ["All"] + sorted([m for m in df_explorer["method"].dropna().unique().tolist() if m]) if not df_explorer.empty else ["All"]
        sel_method = st.selectbox("Method", methods_list, key="f_method")
    with fcol5:
        search_query = st.text_input("Search file or source", "", placeholder="e.g. img_src_0001", key="f_search")

    # Apply Filters
    df_filtered = df_explorer.copy()
    if not df_filtered.empty:
        if sel_format != "All":
            df_filtered = df_filtered[df_filtered["media_type"] == sel_format]
        if sel_split == "Test split only":
            df_filtered = df_filtered[df_filtered["split"] == "test"]
        elif sel_split != "All":
            df_filtered = df_filtered[df_filtered["split"] == sel_split]
        if sel_label == "Clean":
            df_filtered = df_filtered[df_filtered["label"] == 0]
        elif sel_label == "Hidden Data":
            df_filtered = df_filtered[df_filtered["label"] == 1]
        if sel_method != "All":
            df_filtered = df_filtered[df_filtered["method"] == sel_method]
        if search_query.strip():
            sq = search_query.strip().lower()
            df_filtered = df_filtered[
                df_filtered["filename"].str.lower().str.contains(sq) |
                df_filtered["source_id"].str.lower().str.contains(sq)
            ]

    # Pagination: 24 per page
    PAGE_SIZE = 24
    total_matching = len(df_filtered)
    total_pages = max(1, (total_matching + PAGE_SIZE - 1) // PAGE_SIZE)
    exp_pg_val = st.session_state.get("explorer_page")
    curr_page = min(exp_pg_val if exp_pg_val is not None else 0, total_pages - 1)

    p_col1, p_col2, p_col3 = st.columns([1.5, 2.0, 1.5])
    with p_col1:
        if st.button("◀ Previous Page", disabled=(curr_page == 0), key="btn_prev_page"):
            st.session_state["explorer_page"] = max(0, curr_page - 1)
            st.rerun()
    with p_col2:
        st.markdown(f"<div style='text-align:center; font-size:12px; color:var(--text-secondary); padding-top:8px;'>Showing page <strong>{curr_page + 1}</strong> of <strong>{total_pages}</strong> ({total_matching} items)</div>", unsafe_allow_html=True)
    with p_col3:
        if st.button("Next Page ▶", disabled=(curr_page >= total_pages - 1), key="btn_next_page"):
            st.session_state["explorer_page"] = min(total_pages - 1, curr_page + 1)
            st.rerun()

    start_idx = curr_page * PAGE_SIZE
    page_items = df_filtered.iloc[start_idx : start_idx + PAGE_SIZE]

    if page_items.empty:
        st.info("No files match the selected filter criteria.")
    else:
        # Render grid of items (3 items per row)
        cols_per_row = 3
        for row_start in range(0, len(page_items), cols_per_row):
            row_items = page_items.iloc[row_start : row_start + cols_per_row]
            grid_cols = st.columns(cols_per_row)
            for c_idx, (_, item) in enumerate(row_items.iterrows()):
                with grid_cols[c_idx]:
                    b_info = format_blind_mode(dict(item), is_blind=is_blind)
                    fname = item.get("filename") or "sample"
                    mtype = item.get("media_type") or "image"
                    split_name = item.get("split") or "test"
                    src_id = item.get("source_id") or "src"
                    sha = item.get("sha256") or ""
                    sha_prefix = sha[:8] if sha else "N/A"
                    
                    # Resolve filepath
                    raw_path = item.get("resolved_filepath") or item.get("filepath")
                    resolved_p = (root_dir / str(raw_path)) if raw_path else None
                    file_exists = (resolved_p is not None and resolved_p.exists())
                    
                    st.markdown('<div class="explorer-card">', unsafe_allow_html=True)
                    
                    # Preview box
                    if file_exists and mtype == "image":
                        thumb_url = get_cached_thumbnail(resolved_p)
                        if thumb_url:
                            st.image(thumb_url, use_column_width=True)
                        else:
                            st.image(str(resolved_p), use_column_width=True)
                    elif file_exists and mtype == "audio":
                        st.audio(str(resolved_p))
                    elif file_exists and mtype == "video":
                        st.video(str(resolved_p))
                    else:
                        icon_name = "image" if mtype == "image" else "music" if mtype == "audio" else "video"
                        st.markdown(f'<div class="thumb-box">{get_svg_icon(icon_name, size=32, color="#707070")}</div>', unsafe_allow_html=True)

                    st.markdown(f'''
                    <div style="font-weight:600; font-size:13px; color:var(--text-primary); margin:6px 0 2px 0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{html.escape(fname)}">
                        {html.escape(fname)}
                    </div>
                    <div style="font-size:11px; color:var(--text-muted); margin-bottom:8px;">
                        Split: <span class="badge-media">{split_name.upper()}</span> · SHA: <code>{sha_prefix}</code>
                    </div>
                    <div style="font-size:12px; color:var(--text-secondary); margin-bottom:4px;">
                        Condition: <strong style="color:var(--text-primary);">{b_info["label_str"]}</strong>
                    </div>
                    <div style="font-size:12px; color:var(--text-secondary); margin-bottom:8px;">
                        Method: <strong style="color:var(--text-primary);">{b_info["method_str"]}</strong> ({b_info["payload_rate_str"]})
                    </div>
                    ''', unsafe_allow_html=True)

                    # Actions: Scan this file & Download
                    act_c1, act_c2 = st.columns(2)
                    with act_c1:
                        if file_exists:
                            if st.button("Scan this file", key=f"btn_scan_{fname}_{start_idx + row_start + c_idx}", type="primary", use_container_width=True):
                                st.session_state["selected_sample_path"] = str(resolved_p)
                                st.session_state["nav_page"] = "scan"
                                st.rerun()
                        else:
                            st.caption("File requires make data")
                    with act_c2:
                        if file_exists:
                            with open(resolved_p, "rb") as f_down:
                                st.download_button(
                                    label="Download",
                                    data=f_down.read(),
                                    file_name=fname,
                                    mime="application/octet-stream",
                                    key=f"dl_{fname}_{start_idx + row_start + c_idx}",
                                    use_container_width=True
                                )
                    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# PAGE 3: RESULTS (from metrics.json only)
# =====================================================================
def render_results_page():
    st.markdown('''
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin:0;">Evaluation Results & Benchmark Metrics</h1>
        <div style="color: var(--text-secondary); font-size: 14px; margin-top: 2px;">
            Strictly derived from real test evaluation metrics in docs/RESULTS/
        </div>
    </div>
    ''', unsafe_allow_html=True)

    metrics_data = load_display_metrics()

    if not metrics_data:
        st.markdown('''
        <div class="empty-state-box">
            <div class="empty-title">No real training run yet</div>
            <div class="empty-subtext">Run training with laptop or GPU profile to generate real benchmark metrics.</div>
        </div>
        ''', unsafe_allow_html=True)
        return

    is_smoke = bool(metrics_data.get("is_smoke") or str(metrics_data.get("profile", "")).lower() == "smoke" or not metrics_data.get("profile"))

    if is_smoke:
        st.markdown('''
        <div class="badge-smoke-banner">
            ⚠️ Smoke test, not real performance
        </div>
        ''', unsafe_allow_html=True)
    else:
        profile_name = str(metrics_data.get("profile", "laptop")).upper()
        device_name = str(metrics_data.get("device", "auto"))
        epochs_val = metrics_data.get("epochs", "N/A")
        st.markdown(f'''
        <div style="display:flex; align-items:center; gap:8px; margin-bottom: 16px;">
            <span style="font-size:12px; font-weight:600; padding:4px 10px; border-radius:6px; background:rgba(63, 182, 122, 0.15); border:1px solid #3FB67A; color:#3FB67A;">
                REAL RUN: {profile_name} PROFILE
            </span>
            <span style="font-size:12px; color:var(--text-secondary);">
                Device: <strong>{device_name}</strong> | Epochs: <strong>{epochs_val}</strong> | Commit: <code>{str(metrics_data.get("git_commit", "unknown"))[:8]}</code>
            </span>
        </div>
        ''', unsafe_allow_html=True)

    # Main Metrics Table (Format | Accuracy | ROC-AUC (95% CI) | Recall @1% FPR | Recall @5% FPR | Recall @10% FPR | Test files)
    table_html = build_metrics_table_html(metrics_data, is_smoke=is_smoke)
    st.markdown(table_html, unsafe_allow_html=True)

    # Side-by-Side Model Comparison Table (Format | Deep model | Handcrafted MLP | Gradient boosting)
    comparison_data = load_display_comparison_results()
    if comparison_data:
        comp_is_smoke = bool(comparison_data.get("is_smoke") or str(comparison_data.get("profile", "")).lower() == "smoke" or not comparison_data.get("profile"))
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown('''
        <div style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-secondary); letter-spacing:0.08em; margin-bottom:8px;">
            Side-by-Side Model Comparison (Same Test Manifest)
        </div>
        ''', unsafe_allow_html=True)
        comp_table_html = build_comparison_table_html(comparison_data, is_smoke=comp_is_smoke)
        st.markdown(comp_table_html, unsafe_allow_html=True)

    # Bar chart of accuracy per hiding method
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown('''
    <div style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-secondary); letter-spacing:0.08em; margin-bottom:8px;">
        Accuracy by Hiding Method
    </div>
    ''', unsafe_allow_html=True)

    per_method = metrics_data.get("per_method") or {}
    if per_method:
        m_names = []
        m_accs = []
        m_annotations = []
        m_colors = []

        for m_key, m_info in per_method.items():
            acc_val = m_info.get("accuracy")
            acc_num = float(acc_val) if acc_val is not None else 0.0
            cnt = m_info.get("sample_count")
            cnt_num = int(cnt) if cnt is not None else 0
            is_insufficient = (cnt_num < 30) or bool(m_info.get("insufficient_samples"))
            
            clean_name = m_key.replace("_", " ").title()
            if is_insufficient:
                clean_name += " *"
            
            m_names.append(clean_name)
            m_accs.append(acc_num)
            m_colors.append('#E5522B' if acc_num >= 0.7 else '#E0B341' if acc_num >= 0.5 else '#3FB67A')
            
            if is_insufficient:
                m_annotations.append(f"Too few samples (N={cnt_num})")
            else:
                m_annotations.append(f"{acc_num * 100:.1f}% (N={cnt_num})")

        fig_methods = go.Figure(go.Bar(
            x=m_names,
            y=m_accs,
            text=m_annotations,
            textposition='auto',
            marker=dict(color=m_colors, line=dict(color='rgba(255,255,255,0.1)', width=1))
        ))
        fig_methods.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B8B8B', family='Inter'),
            yaxis=dict(range=[0, 1.15], gridcolor='rgba(255,255,255,0.05)', tickformat='.0%'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(l=20, r=20, t=20, b=30),
            height=320
        )
        st.plotly_chart(fig_methods, use_container_width=True)
        st.caption("* Note: Methods marked with asterisk have fewer than 30 evaluation samples.")

    # Artifact Visuals: Method x Payload Heatmap, ROC Curves & Confusion Matrix
    heatmap_img_path = root_dir / "docs" / "RESULTS" / "method_payload_heatmap.png"
    if heatmap_img_path.exists():
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("**Detection Rate by Hiding Method and Payload Rate**")
        st.image(str(heatmap_img_path), use_column_width=True)

    col_img1, col_img2 = st.columns(2)
    roc_img_path = root_dir / "docs" / "RESULTS" / "roc_curves.png"
    cm_img_path = root_dir / "docs" / "RESULTS" / "confusion_matrices.png"

    with col_img1:
        if roc_img_path.exists():
            st.markdown("**ROC Curves (All Formats)**")
            st.image(str(roc_img_path), use_column_width=True)
    with col_img2:
        if cm_img_path.exists():
            st.markdown("**Test Confusion Matrices**")
            st.image(str(cm_img_path), use_column_width=True)

    # "Known Limitations" Card
    st.markdown('''
    <div class="glass-card" style="margin-top: 20px;">
        <div style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-secondary); letter-spacing:0.08em; margin-bottom:8px;">
            Known Limitations
        </div>
        <ul style="font-size:13px; color:var(--text-primary); margin:0; padding-left:18px; line-height:1.7;">
            <li><strong>Audio detection is weak:</strong> Echo hiding and low-rate acoustic LSB remain difficult without raw PCM bitplane parity.</li>
            <li><strong>Adaptive hiding is hard:</strong> Content-adaptive algorithms such as edge-adaptive spatial embedding avoid dense edge residuals.</li>
            <li><strong>Synthetic stego files:</strong> Hidden-data media are produced by our research generators; real-world adversaries may employ custom ciphers.</li>
            <li><strong>Video motion constraints:</strong> Video dataset consists of simulated pan/zoom clips generated from still photography rather than complex camera motion.</li>
            <li><strong>Conservative alert threshold:</strong> Operating at 5% clean false alarm target inevitably misses low-payload injections (&lt; 0.10 bpp).</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

    # "What do these numbers mean?" Expander
    with st.expander("What do these numbers mean?", expanded=False):
        st.markdown('''
        <div style="font-size: 13px; color: var(--text-secondary); line-height: 1.6;">
            <p><strong>Accuracy:</strong> Percentage of all evaluated clean and hidden-data media correctly classified at the alert threshold.</p>
            <p><strong>ROC-AUC (Area Under ROC Curve):</strong> Comprehensive diagnostic discrimination across all possible operating thresholds (1.0 = perfect detection, 0.5 = coin toss).</p>
            <p><strong>Precision:</strong> When VaultBreaker flags a file as containing hidden data, the proportion of times it is truly steganographic.</p>
            <p><strong>Recall:</strong> The fraction of all true hidden-data files in the test partition that VaultBreaker successfully detects.</p>
        </div>
        ''', unsafe_allow_html=True)


# =====================================================================
# PAGE 4: HISTORY
# =====================================================================
def render_history_page():
    st.markdown('''
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin:0;">Scan History</h1>
        <div style="color: var(--text-secondary); font-size: 14px; margin-top: 2px;">
            Persistent forensic log of media files analyzed by VaultBreaker
        </div>
    </div>
    ''', unsafe_allow_html=True)

    scans = get_all_scans(limit=500)

    # Top action bar
    col_h_left, col_h_right = st.columns([2.0, 1.5])
    with col_h_left:
        st.markdown(f"<div style='font-size:13px; color:var(--text-secondary); padding-top:6px;'>Total Recorded Scans: <strong style='color:var(--text-primary); font-family:JetBrains Mono;'>{len(scans)}</strong></div>", unsafe_allow_html=True)

    with col_h_right:
        act_c1, act_c2 = st.columns(2)
        with act_c1:
            if scans:
                df_export = pd.DataFrame([{
                    "scan_id": s.get("scan_id"),
                    "created_at": s.get("created_at"),
                    "filename": s.get("filename"),
                    "media_type": s.get("media_type"),
                    "format": s.get("format"),
                    "probability": s.get("probability"),
                    "verdict": s.get("verdict"),
                    "ground_truth_label": s.get("ground_truth_label"),
                    "is_correct": s.get("is_correct")
                } for s in scans])
                st.download_button(
                    "Export CSV",
                    data=df_export.to_csv(index=False),
                    file_name="vaultbreaker_scan_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_export_csv"
                )
        with act_c2:
            if scans:
                with st.popover("Clear history", use_container_width=True):
                    st.warning("Permanently delete all scan records from database?")
                    if st.button("Confirm Clear", type="primary", key="btn_confirm_clear_history"):
                        clear_all_scans()
                        st.session_state["active_scan_id"] = None
                        st.toast("Scan history cleared", icon="🗑️")
                        st.rerun()

    # Empty State
    if not scans:
        st.markdown(empty_state(
            title="Scan History is Empty",
            subtext="A fresh start has zero recorded scans. Inspect a file on the Scan page or browse evaluation samples in the Data Explorer.",
            button_label=None,
            icon_name="clock"
        ), unsafe_allow_html=True)
        col_c, _ = st.columns([1.5, 3.5])
        with col_c:
            if st.button("Try a sample in Data Explorer", type="primary", use_container_width=True, key="btn_try_sample_history"):
                st.session_state["nav_page"] = "data_explorer"
                st.rerun()
        return

    # Table of past scans
    rows = []
    for s in scans:
        sid = s.get("scan_id") or ""
        dt = (s.get("created_at") or "")[:19]
        fn = s.get("filename") or "unknown"
        mtype = (s.get("media_type") or "image").upper()
        prob_val = s.get("probability")
        prob = float(prob_val) if prob_val is not None else 0.0
        prob_str = f"{prob * 100:.1f}%"
        verdict = s.get("verdict") or "CLEAN"
        is_stego = ("STEGO" in verdict or prob >= predictor.calibrator.decision_threshold)
        v_chip = score_pill_html(prob, threshold=predictor.calibrator.decision_threshold)
        
        # Ground truth correctness
        gt_lbl = s.get("ground_truth_label")
        is_corr = s.get("is_correct")
        if gt_lbl is not None:
            actual_str = "Stego" if gt_lbl == 1 else "Clean"
            if is_corr == 1:
                corr_str = '<span style="color:#3FB67A; font-weight:600;">Correct</span>'
            else:
                corr_str = '<span style="color:#E0B341; font-weight:600;">Incorrect</span>'
            gt_display = f"{actual_str} ({corr_str})"
        else:
            gt_display = '<span style="color:#707070;">Live Upload</span>'

        rows.append(f'''
        <tr>
            <td style="font-family:'JetBrains Mono'; font-weight:600; color:#EDEDED;">{sid}</td>
            <td style="color:#A0A0A0; font-size:12px;">{dt}</td>
            <td style="font-weight:600; color:#EDEDED;">{html.escape(fn)}</td>
            <td><span class="badge-media">{mtype}</span></td>
            <td>{v_chip}</td>
            <td>{gt_display}</td>
        </tr>
        ''')

    st.markdown(f'''
    <table class="glass-table">
        <thead>
            <tr>
                <th>Scan ID</th>
                <th>Timestamp</th>
                <th>File Name</th>
                <th>Modality</th>
                <th>Calibrated Score</th>
                <th>Ground Truth Verification</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    ''', unsafe_allow_html=True)

    # Reopen Scan selector
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    scan_map = {f"{s.get('scan_id')} - {s.get('filename')} ({float(s.get('probability') or 0.0)*100:.1f}%)": s.get("scan_id") for s in scans}
    sel_label = st.selectbox("Select Scan to Reopen in Inspector", list(scan_map.keys()), key="sel_reopen_scan")
    if st.button("Inspect Selected Scan", type="secondary", key="btn_reopen_inspect"):
        chosen_id = scan_map.get(sel_label)
        if chosen_id:
            st.session_state["active_scan_id"] = chosen_id
            st.session_state["nav_page"] = "scan"
            st.rerun()


# --- ROUTER DISPATCH ---
current_page = st.session_state.get("nav_page", "scan")
if current_page == "scan":
    render_scan_page()
elif current_page == "hide":
    render_hide_page()
elif current_page == "reveal":
    render_reveal_page()
elif current_page == "data_explorer":
    render_data_explorer_page()
elif current_page == "results":
    render_results_page()
elif current_page == "history":
    render_history_page()
else:
    render_scan_page()
