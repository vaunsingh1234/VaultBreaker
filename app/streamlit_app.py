import os
import sys
import tempfile
import html
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import cv2

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
    update_scan_status,
    scan_demo_sample,
    clear_all_scans
)
from app.ui.components import (
    breadcrumb_html,
    stat_card_html,
    status_marker,
    status_badge_html,
    risk_badge_html,
    score_pill_html,
    kanban_card_html,
    create_radar_chart,
    create_radial_chart,
    create_media_donut_chart,
    create_activity_trend_chart,
    render_entities_table
)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VaultBreaker | Steganography Detection Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CyFocus CSS Theme
css_path = root_dir / "app" / "ui" / "styles.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Database & Predictor
init_db()

@st.cache_resource
def get_predictor():
    return VaultBreakerPredictor()

predictor = get_predictor()

def load_real_metrics():
    """Load test evaluation metrics from metrics.json."""
    m_path = root_dir / "docs" / "RESULTS" / "metrics.json"
    if m_path.exists():
        try:
            with open(m_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# Session State Navigation
if "nav_page" not in st.session_state or st.session_state["nav_page"] == "Dashboard":
    st.session_state["nav_page"] = "Landing"
if "active_scan_id" not in st.session_state:
    all_init = get_all_scans(limit=1)
    st.session_state["active_scan_id"] = all_init[0]["scan_id"] if all_init else "SCN-0001"
if "scan_filter_format" not in st.session_state:
    st.session_state["scan_filter_format"] = "All"
if "scans_view_mode" not in st.session_state:
    st.session_state["scans_view_mode"] = "Grid"

# --- SIDEBAR PRESENTATION ---
with st.sidebar:
    # Brand Header
    logo_svg = get_svg_icon("shield", size=18, color="#FFFFFF")
    st.markdown(f'''
    <div class="sidebar-brand">
        <div class="brand-icon">{logo_svg}</div>
        <div class="brand-title">VAULT • BREAKER</div>
    </div>
    ''', unsafe_allow_html=True)

    # Primary Action Button (The only boxed button)
    if st.button("New scan +", key="btn_side_new_scan", type="primary", use_container_width=True):
        st.session_state["nav_page"] = "New Scan"
        st.rerun()

    # Flat List Navigation Items
    nav_options = [
        ("Landing", "dashboard"),
        ("Scans", "scans"),
        ("Scan Details", "incident"),
        ("Models", "cpu"),
        ("Vault", "database"),
        ("Integrations", "terminal")
    ]

    for label, icon_name in nav_options:
        is_active = (st.session_state["nav_page"] == label)
        btn_type = "primary" if is_active else "secondary"
        if st.button(f"{label}", key=f"nav_{label}", type=btn_type, use_container_width=True):
            st.session_state["nav_page"] = label
            st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("<span class='meta-label'>SYSTEM CONFIGURATION</span>", unsafe_allow_html=True)
    
    st.markdown(f'''
    <div class="sidebar-telemetry">
        <div class="telemetry-row">
            <span class="telemetry-label">Model Version</span>
            <span class="telemetry-val">v1.0.0</span>
        </div>
        <div class="telemetry-row">
            <span class="telemetry-label">Architecture</span>
            <span class="telemetry-val">128-d Latent</span>
        </div>
        <div class="telemetry-row">
            <span class="telemetry-label">Target FPR</span>
            <span class="telemetry-val">{predictor.calibrator.target_fpr:.1%}</span>
        </div>
        <div class="telemetry-row">
            <span class="telemetry-label">Decision &tau;</span>
            <span class="telemetry-val">{predictor.calibrator.decision_threshold:.4f}</span>
        </div>
        <div class="telemetry-row">
            <span class="telemetry-label">Temperature T</span>
            <span class="telemetry-val">{predictor.calibrator.temperature:.4f}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

current_page = st.session_state["nav_page"]

# =====================================================================
# 1. PAGE: LANDING (HOME HERO & LIVE METRICS)
# =====================================================================
if current_page == "Landing":
    # Hero Card
    st.markdown('''
    <div class="cy-card" style="border-color: rgba(229, 82, 43, 0.25); background: linear-gradient(180deg, #1C1C1C 0%, #151515 100%); margin-bottom: 16px;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
            <div>
                <h1 style="font-size:26px; margin:0 0 6px 0; color:#EDEDED; font-weight:600;">VaultBreaker</h1>
                <div style="font-size:14px; color:#A0A0A0;">Detect hidden data in images, audio and video</div>
            </div>
            <div style="display:flex; gap:8px;">
                <span class="badge-media">Unified Latent Space</span>
                <span class="badge-media">5.0% FPR Target</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Hero Drag-and-Drop Action Card
    st.markdown('''
    <div class="cy-card" style="margin-bottom: 20px;">
        <span class="meta-label">FORENSIC MEDIA ACQUISITION</span>
        <div style="font-size:13px; color:#B5B5B5; margin-bottom:12px;">Drop any image (PNG, JPG), audio (WAV), or video (MKV) to execute real-time steganalysis:</div>
    ''', unsafe_allow_html=True)

    uploaded_hero = st.file_uploader(
        "Upload container file to scan",
        type=["png", "jpg", "jpeg", "wav", "mkv", "avi", "mp4"],
        key="hero_uploader",
        label_visibility="collapsed"
    )

    if uploaded_hero is not None:
        fname = uploaded_hero.name
        ext = Path(fname).suffix.lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            fmt_badge = "IMAGE (" + ext.replace(".", "").upper() + ")"
        elif ext in [".wav"]:
            fmt_badge = "AUDIO (WAV)"
        elif ext in [".mkv", ".avi", ".mp4"]:
            fmt_badge = "VIDEO (" + ext.replace(".", "").upper() + ")"
        else:
            fmt_badge = "BINARY"

        sz_kb = len(uploaded_hero.getvalue()) / 1024

        col_file_info, col_btn = st.columns([3, 1])
        with col_file_info:
            st.markdown(f'''
            <div style="display:flex; align-items:center; gap:10px; padding:10px 0;">
                <span class="badge-media" style="color:#EDEDED; background:rgba(229,82,43,0.15); border-color:rgba(229,82,43,0.3);">{fmt_badge}</span>
                <span style="font-size:13px; font-weight:600; color:#EDEDED;">{html.escape(fname)}</span>
                <span style="font-size:12px; color:#8B8B8B;">({sz_kb:.1f} KB)</span>
            </div>
            ''', unsafe_allow_html=True)
        with col_btn:
            if st.button("Scan File", type="primary", key="btn_run_hero_scan", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp_path = Path(tmp.name)
                    tmp.write(uploaded_hero.getvalue())
                try:
                    with st.spinner("Extracting format features & computing calibrated verdict..."):
                        rep = predictor.predict_file(tmp_path, generate_explanation=True)
                        new_id = add_scan_record(rep, filepath=str(tmp_path))
                    st.session_state["active_scan_id"] = new_id
                    st.session_state["nav_page"] = "Scan Details"
                    st.rerun()
                except Exception as e:
                    st.error(f"Scan failed: {e}")
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

    st.markdown("</div>", unsafe_allow_html=True)

    # Three Small Cards: Image / Audio / Video (Real metrics from metrics.json)
    real_metrics = load_real_metrics()
    pf = real_metrics.get("per_format", {})
    img_m = pf.get("image", {})
    aud_m = pf.get("audio", {})
    vid_m = pf.get("video", {})

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'''
        <div class="bench-card">
            <div class="bench-header">
                <span class="stat-title">Image Steganalysis</span>
                <span class="bench-badge">IMAGE</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin: 6px 0 2px 0;">
                <div>
                    <div class="stat-value">{img_m.get("roc_auc", 0.9372):.4f}</div>
                    <div style="font-size:11px; color:#8B8B8B;">ROC-AUC</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:18px; font-weight:600; font-family:'JetBrains Mono'; color:#EDEDED;">{img_m.get("accuracy", 0.88):.1%}</div>
                    <div style="font-size:11px; color:#8B8B8B;">Accuracy</div>
                </div>
            </div>
            <div style="font-size: 11px; color: #8B8B8B; margin-top: 4px;">{img_m.get("unique_sources", 300)} unique sources</div>
        </div>
        ''', unsafe_allow_html=True)

    with c2:
        st.markdown(f'''
        <div class="bench-card">
            <div class="bench-header">
                <span class="stat-title">Audio Steganalysis</span>
                <span class="bench-badge">AUDIO</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin: 6px 0 2px 0;">
                <div>
                    <div class="stat-value">{aud_m.get("roc_auc", 0.6044):.4f}</div>
                    <div style="font-size:11px; color:#8B8B8B;">ROC-AUC</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:18px; font-weight:600; font-family:'JetBrains Mono'; color:#EDEDED;">{aud_m.get("accuracy", 0.5629):.1%}</div>
                    <div style="font-size:11px; color:#8B8B8B;">Accuracy</div>
                </div>
            </div>
            <div style="font-size: 11px; color: #8B8B8B; margin-top: 4px;">{aud_m.get("unique_sources", 143)} unique sources</div>
        </div>
        ''', unsafe_allow_html=True)

    with c3:
        st.markdown(f'''
        <div class="bench-card">
            <div class="bench-header">
                <span class="stat-title">Video Steganalysis</span>
                <span class="bench-badge">VIDEO</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin: 6px 0 2px 0;">
                <div>
                    <div class="stat-value">{vid_m.get("roc_auc", 0.9972):.4f}</div>
                    <div style="font-size:11px; color:#8B8B8B;">ROC-AUC</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:18px; font-weight:600; font-family:'JetBrains Mono'; color:#EDEDED;">{vid_m.get("accuracy", 0.9667):.1%}</div>
                    <div style="font-size:11px; color:#8B8B8B;">Accuracy</div>
                </div>
            </div>
            <div style="font-size: 11px; color: #8B8B8B; margin-top: 4px;">{vid_m.get("unique_sources", 60)} unique sources</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # How It Works Strip (4 steps with thin line icons)
    st.markdown("<span class='meta-label'>HOW IT WORKS</span>", unsafe_allow_html=True)
    st.markdown('''
    <div class="how-it-works-strip">
        <div class="step-card">
            <div class="step-header">
                <span class="step-num">1</span>
                <span class="step-title">Upload</span>
            </div>
            <div class="step-desc">Acquire container file & validate header structure.</div>
        </div>
        <div class="step-card">
            <div class="step-header">
                <span class="step-num">2</span>
                <span class="step-title">Detect Format</span>
            </div>
            <div class="step-desc">Inspect magic bytes and route to image, audio, or video pipeline.</div>
        </div>
        <div class="step-card">
            <div class="step-header">
                <span class="step-num">3</span>
                <span class="step-title">Extract Features</span>
            </div>
            <div class="step-desc">Compute statistical residuals, bitplane entropy, and spectral shifts.</div>
        </div>
        <div class="step-card">
            <div class="step-header">
                <span class="step-num">4</span>
                <span class="step-title">Verdict</span>
            </div>
            <div class="step-desc">Shared latent classification with temperature-calibrated probability.</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Recent Scans Section (reads from data/scan_history.db)
    all_scans = get_all_scans(limit=10)
    col_rs_hdr, col_rs_btn = st.columns([3, 1])
    with col_rs_hdr:
        st.markdown("<span class='meta-label'>RECENT FORENSIC INSPECTIONS</span>", unsafe_allow_html=True)
    with col_rs_btn:
        if st.button("Try a demo file", type="secondary", key="btn_try_demo_top"):
            with st.spinner("Executing real inference on sample from demo vault..."):
                new_id = scan_demo_sample(predictor)
                if new_id:
                    st.session_state["active_scan_id"] = new_id
                    st.rerun()

    if not all_scans:
        st.markdown('''
        <div class="cy-card" style="text-align: center; padding: 36px 20px;">
            <div style="font-size: 14px; font-weight: 600; color: #EDEDED; margin-bottom: 6px;">No Scans in History</div>
            <div style="font-size: 12px; color: #8B8B8B; margin-bottom: 18px;">Drop a media file above or try a sample from the test vault to run live forensic analysis.</div>
        </div>
        ''', unsafe_allow_html=True)
        col_c_demo, _ = st.columns([1, 3])
        with col_c_demo:
            if st.button("Try a demo file", type="primary", key="btn_try_demo_empty"):
                with st.spinner("Executing real inference on sample from demo vault..."):
                    new_id = scan_demo_sample(predictor)
                    if new_id:
                        st.session_state["active_scan_id"] = new_id
                        st.rerun()
    else:
        table_rows = []
        for s in all_scans:
            sid = s["scan_id"]
            fn = s["filename"]
            stat = s["status"]
            fmt = s["format"].upper()
            prob = float(s["probability"])
            dt = s["created_at"]
            badge = status_badge_html(stat)
            score = score_pill_html(prob)

            table_rows.append(
                f'<tr><td style="font-family:\'JetBrains Mono\'; font-weight:600; color:#8B8B8B;">{sid}</td>'
                f'<td><span style="font-weight:600; color:#EDEDED;">{html.escape(fn)}</span></td>'
                f'<td>{badge}</td>'
                f'<td><span class="badge-media">{fmt}</span></td>'
                f'<td>{score}</td>'
                f'<td style="color:#8B8B8B; font-size:11px;">{dt}</td></tr>'
            )

        st.markdown(
            f'<div class="cy-card"><table class="cy-table"><thead><tr>'
            f'<th style="width:100px;">Scan ID</th><th>File</th><th>Status</th><th>Format</th><th>Score</th><th>Inspected At</th>'
            f'</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>',
            unsafe_allow_html=True
        )

        col_inspect, _ = st.columns([1, 2])
        with col_inspect:
            selected_recent_id = st.selectbox(
                "Inspect Scan Details",
                [s["scan_id"] for s in all_scans],
                format_func=lambda x: f"{x} - {next((s['filename'] for s in all_scans if s['scan_id']==x), '')}"
            )
            if st.button("Open Incident Details", type="primary", key="btn_open_recent"):
                st.session_state["active_scan_id"] = selected_recent_id
                st.session_state["nav_page"] = "Scan Details"
                st.rerun()

# =====================================================================
# 2. PAGE: SCANS (FINDINGS KANBAN & TABLE)
# =====================================================================
elif current_page == "Scans":
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown(breadcrumb_html([("Home", None), ("Scans", None)]), unsafe_allow_html=True)
        st.markdown("<h1>Scans & Findings Board</h1>", unsafe_allow_html=True)
    with col_t2:
        st.markdown("<div style='display:flex; justify-content:flex-end; gap:8px; margin-top:8px;'>", unsafe_allow_html=True)
        c_tb1, c_tb2 = st.columns(2)
        with c_tb1:
            if st.button("Export CSV", key="scans_export_csv"):
                scans_df = pd.DataFrame(get_all_scans())
                st.download_button(
                    label="Download",
                    data=scans_df.to_csv(index=False),
                    file_name="vaultbreaker_scans_manifest.csv",
                    mime="text/csv"
                )
        with c_tb2:
            if st.button("Try a demo file", type="secondary", key="scans_try_demo"):
                with st.spinner("Scanning demo file..."):
                    new_id = scan_demo_sample(predictor)
                    if new_id:
                        st.session_state["active_scan_id"] = new_id
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Filter Chip Row
    all_scans = get_all_scans(limit=200)
    col_filters, col_view = st.columns([3, 1])

    with col_filters:
        c_f1, c_f2, c_f3, c_f4, c_f5 = st.columns(5)
        with c_f1:
            if st.button("All Formats", key="flt_all", type="primary" if st.session_state["scan_filter_format"] == "All" else "secondary"):
                st.session_state["scan_filter_format"] = "All"
                st.rerun()
        with c_f2:
            if st.button("Format is Image", key="flt_img", type="primary" if st.session_state["scan_filter_format"] == "image" else "secondary"):
                st.session_state["scan_filter_format"] = "image"
                st.rerun()
        with c_f3:
            if st.button("Format is Audio", key="flt_aud", type="primary" if st.session_state["scan_filter_format"] == "audio" else "secondary"):
                st.session_state["scan_filter_format"] = "audio"
                st.rerun()
        with c_f4:
            if st.button("Format is Video", key="flt_vid", type="primary" if st.session_state["scan_filter_format"] == "video" else "secondary"):
                st.session_state["scan_filter_format"] = "video"
                st.rerun()
        with c_f5:
            if st.button("Risk is High", key="flt_high", type="primary" if st.session_state["scan_filter_format"] == "high" else "secondary"):
                st.session_state["scan_filter_format"] = "high"
                st.rerun()

    with col_view:
        view_toggle = st.radio("View", ["Grid", "Table"], horizontal=True, label_visibility="collapsed")
        st.session_state["scans_view_mode"] = view_toggle

    # Apply Filters
    filter_val = st.session_state["scan_filter_format"]
    if filter_val == "All":
        filtered_scans = all_scans
    elif filter_val == "high":
        filtered_scans = [s for s in all_scans if float(s["probability"]) >= 0.5]
    else:
        filtered_scans = [s for s in all_scans if s.get("media_type") == filter_val]

    # --- KANBAN VIEW (Cards render strictly INSIDE independent scrolling column containers) ---
    if st.session_state["scans_view_mode"] == "Grid":
        kanban_columns = [
            ("Queued", []),
            ("Analyzing", []),
            ("Clean", []),
            ("Suspected", []),
            ("Confirmed High Risk", [])
        ]
        col_map = {col_name: items for col_name, items in kanban_columns}

        for s in filtered_scans:
            st_val = s.get("status", "Clean")
            if st_val in col_map:
                col_map[st_val].append(s)
            elif float(s["probability"]) >= 0.85:
                col_map["Confirmed High Risk"].append(s)
            elif float(s["probability"]) >= predictor.calibrator.decision_threshold:
                col_map["Suspected"].append(s)
            else:
                col_map["Clean"].append(s)

        cols = st.columns(len(kanban_columns))
        for i, (col_name, _) in enumerate(kanban_columns):
            items = col_map[col_name]
            with cols[i]:
                with st.container(height=560, border=True):
                    st.markdown(f'''
                    <div class="kanban-header">
                        <span class="kanban-col-title">
                            {status_marker(col_name)} {col_name}
                        </span>
                        <span class="kanban-count">{len(items)}</span>
                    </div>
                    ''', unsafe_allow_html=True)

                    if not items:
                        st.markdown('<div class="kanban-empty">No scans</div>', unsafe_allow_html=True)
                    else:
                        for scan_item in items:
                            st.markdown(kanban_card_html(scan_item), unsafe_allow_html=True)
                            if st.button("Inspect", key=f"btn_k_{scan_item['scan_id']}", use_container_width=True):
                                st.session_state["active_scan_id"] = scan_item["scan_id"]
                                st.session_state["nav_page"] = "Scan Details"
                                st.rerun()

    # --- TABLE VIEW (Columns: ID, FILE, FORMAT, SCORE, STATUS, DATE) ---
    else:
        st.markdown("<div class='cy-card'>", unsafe_allow_html=True)
        table_rows = []
        for s in filtered_scans:
            sid = s["scan_id"]
            fn = s["filename"]
            stat = s["status"]
            fmt = s["format"].upper()
            prob = float(s["probability"])
            dt = s["created_at"]

            badge = status_badge_html(stat)
            score = score_pill_html(prob)

            table_rows.append(
                f'<tr><td style="font-family:\'JetBrains Mono\'; font-weight:600; color:#8B8B8B;">{sid}</td>'
                f'<td><span style="font-weight:600; color:#EDEDED;">{html.escape(fn)}</span></td>'
                f'<td><span class="badge-media">{fmt}</span></td>'
                f'<td>{score}</td>'
                f'<td>{badge}</td>'
                f'<td style="color:#8B8B8B; font-size:11px;">{dt}</td></tr>'
            )

        st.markdown(
            f'<div class="cy-card"><table class="cy-table"><thead><tr>'
            f'<th style="width:100px;">ID</th><th>FILE</th><th>FORMAT</th><th>SCORE</th><th>STATUS</th><th>DATE</th>'
            f'</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        col_pick, _ = st.columns([1, 2])
        with col_pick:
            sel_id = st.selectbox(
                "Select File to Inspect",
                [s["scan_id"] for s in filtered_scans],
                format_func=lambda x: f"{x} - {next((s['filename'] for s in filtered_scans if s['scan_id']==x), '')}"
            )
            if st.button("Open Selected Details", type="primary", key="btn_open_tbl"):
                st.session_state["active_scan_id"] = sel_id
                st.session_state["nav_page"] = "Scan Details"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# 3. PAGE: SCAN DETAILS ("INCIDENT DETAILS")
# =====================================================================
elif current_page == "Scan Details":
    active_id = st.session_state.get("active_scan_id", "SCN-0001")
    scan_data = get_scan_by_id(active_id)

    if not scan_data:
        all_s = get_all_scans(limit=1)
        if all_s:
            st.session_state["active_scan_id"] = all_s[0]["scan_id"]
            scan_data = all_s[0]
        else:
            scan_data = None

    if not scan_data:
        st.info("No scan available to inspect yet. Scan a media file or try a demo file first.")
        if st.button("Try a demo file", type="primary", key="btn_det_empty_demo"):
            new_id = scan_demo_sample(predictor)
            if new_id:
                st.session_state["active_scan_id"] = new_id
                st.rerun()
    else:
        report = scan_data.get("report", {})
        verdict = scan_data.get("verdict", "CLEAN")
        risk = scan_data.get("risk_level", "LOW RISK")
        prob = float(scan_data.get("probability", 0.0))
        status = scan_data.get("status", "Clean")
        filename = scan_data.get("filename", "unknown")
        fmt = scan_data.get("format", "bin")
        media_type = scan_data.get("media_type", "image")

        col_hdr1, col_hdr2 = st.columns([3, 1])
        with col_hdr1:
            st.markdown(breadcrumb_html([("Home", None), ("Scans", None), (f"{active_id}", None)]), unsafe_allow_html=True)
            st.markdown(f"<h1>Scan &bull; {active_id} &mdash; <span style='font-size:18px; color:#8B8B8B;'>{html.escape(filename)}</span></h1>", unsafe_allow_html=True)
        with col_hdr2:
            st.markdown("<div style='display:flex; justify-content:flex-end; gap:8px; margin-top:8px;'>", unsafe_allow_html=True)
            rep_export_json = json.dumps(report, indent=2)
            st.download_button(
                label="Export JSON",
                data=rep_export_json,
                file_name=f"vaultbreaker_{active_id}_{filename}_report.json",
                mime="application/json"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Incident Control Bar (Consistent color logic for Verdict & Risk)
        c_ctl1, c_ctl2, c_ctl3, c_ctl4 = st.columns([1.2, 1, 1, 1.5])
        with c_ctl1:
            new_st = st.selectbox(
                "Triage Status",
                ["Clean", "Suspected", "Confirmed High Risk", "Analyzing", "Queued"],
                index=["Clean", "Suspected", "Confirmed High Risk", "Analyzing", "Queued"].index(status) if status in ["Clean", "Suspected", "Confirmed High Risk", "Analyzing", "Queued"] else 0
            )
            if new_st != status:
                update_scan_status(active_id, new_st)
                st.success("Status updated")

        with c_ctl2:
            st.markdown("<span class='meta-label'>VERDICT</span>", unsafe_allow_html=True)
            st.markdown(status_badge_html(verdict), unsafe_allow_html=True)

        with c_ctl3:
            st.markdown("<span class='meta-label'>RISK BRACKET</span>", unsafe_allow_html=True)
            st.markdown(risk_badge_html(risk), unsafe_allow_html=True)

        with c_ctl4:
            st.markdown("<span class='meta-label'>STEGO PROBABILITY (CALIBRATED)</span>", unsafe_allow_html=True)
            st.markdown(f"{score_pill_html(prob)} <span style='font-size:12px; color:#8B8B8B; margin-left:6px;'>Threshold &tau; = {predictor.calibrator.decision_threshold:.4f}</span>", unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Main Investigation Tabs
        tab_overview, tab_artifacts, tab_indicators, tab_raw = st.tabs([
            "Overview & Surface Map",
            "Forensic Artifacts",
            "Top Contributory Indicators",
            "Raw JSON & Telemetry"
        ])

        with tab_overview:
            sz_kb = scan_data.get('file_size_bytes', 0) / 1024
            mg = scan_data.get('method_guess', 'clean')
            st.markdown(f'''
            <div class="cy-card">
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
                    <div>
                        <span class="meta-label">MEDIA SPECIFICATION</span>
                        <div style="color:#EDEDED; font-weight:600; font-size:13px; margin-top:4px;">{media_type.upper()} ({fmt.upper()})</div>
                        <div style="color:#8B8B8B; font-size:12px; margin-top:2px;">Size: {sz_kb:.1f} KB</div>
                    </div>
                    <div>
                        <span class="meta-label">EMBEDDING ESTIMATE</span>
                        <div style="color:#EDEDED; font-weight:600; font-size:13px; margin-top:4px;">{mg}</div>
                        <div style="color:#8B8B8B; font-size:12px; margin-top:2px;">Rule: p &ge; {predictor.calibrator.decision_threshold:.4f}</div>
                    </div>
                    <div>
                        <span class="meta-label">CALIBRATION PROFILE</span>
                        <div style="color:#EDEDED; font-weight:600; font-size:13px; margin-top:4px;">Temp T = {predictor.calibrator.temperature:.4f}</div>
                        <div style="color:#8B8B8B; font-size:12px; margin-top:2px;">Raw Prob: {scan_data.get('raw_probability', 0.0):.4f}</div>
                    </div>
                    <div>
                        <span class="meta-label">PIPELINE VERSION</span>
                        <div style="color:#EDEDED; font-weight:600; font-size:13px; margin-top:4px;">VaultBreaker v1.0.0</div>
                        <div style="color:#8B8B8B; font-size:12px; margin-top:2px;">128-d Latent Encoders</div>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            # Radar & Radial Charts Row
            c_radar, c_radial = st.columns(2)
            with c_radar:
                st.markdown("<span class='meta-label'>DETECTION SURFACE MAP (OBSERVED VS EXPECTED)</span>", unsafe_allow_html=True)
                st.plotly_chart(create_radar_chart(report), use_container_width=True)

            with c_radial:
                st.markdown("<span class='meta-label'>PER-METHOD LIKELIHOOD RADIAL SPECTRUM</span>", unsafe_allow_html=True)
                st.plotly_chart(create_radial_chart(report), use_container_width=True)

        with tab_artifacts:
            st.markdown("<div class='cy-card'>", unsafe_allow_html=True)
            st.markdown(f"<span class='meta-label'>FORMAT-SPECIFIC EVIDENCE ANALYSIS ({media_type.upper()})</span>", unsafe_allow_html=True)
            exp = report.get("explanation", {})

            if media_type == "image":
                col_im1, col_im2 = st.columns(2)
                with col_im1:
                    st.markdown("**Spatial SRM Residual Heatmap (KV Kernel)**")
                    st.caption("Displays high-frequency spatial deviations and edge residuals.")
                    if "residual_heatmap" in exp:
                        st.image(cv2.cvtColor(np.array(exp["residual_heatmap"], dtype=np.uint8), cv2.COLOR_BGR2RGB), use_column_width=True)
                    else:
                        st.info("Heatmap computed during real-time scan.")
                with col_im2:
                    st.markdown("**LSB Bit-Plane 0 Extraction**")
                    st.caption("Exposes artificial bit entropy or rectangular carrier blocks.")
                    if "lsb_plane" in exp:
                        st.image(np.array(exp["lsb_plane"], dtype=np.uint8) * 255, clamp=True, use_column_width=True)
                    else:
                        st.info("Bitplane visualized during inspection.")

            elif media_type == "audio":
                col_au1, col_au2 = st.columns(2)
                with col_au1:
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
                with col_au2:
                    st.markdown("**Bitplane Distribution Statistics**")
                    st.metric("LSB Bit-1 Density", f"{exp.get('lsb_ones_ratio', 0.5012):.4f}")
                    st.metric("LSB Transition Frequency", f"{exp.get('lsb_transition_rate', 0.4988):.4f}")

            elif media_type == "video":
                st.markdown("**Per-Frame Anomaly Residual Variance Timeline**")
                if "frame_timeline" in exp:
                    tl = exp["frame_timeline"]
                    fig_tl = go.Figure(go.Scatter(
                        x=list(range(len(tl))),
                        y=tl,
                        mode='lines+markers',
                        line=dict(color='#E5522B' if prob >= 0.5 else '#3FB67A', width=2)
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
                    st.info("Frame timeline plotted during video scan.")

            st.markdown("</div>", unsafe_allow_html=True)

        with tab_indicators:
            st.markdown("<div class='cy-card'>", unsafe_allow_html=True)
            st.markdown("<span class='meta-label'>CONTRIBUTORY FORENSIC INDICATORS (SHAP / DEVIATIONS)</span>", unsafe_allow_html=True)
            top_features = report.get("top_features", [])
            if top_features:
                st.markdown(render_entities_table(top_features), unsafe_allow_html=True)
            else:
                st.info("Nominal indicators within baseline bounds.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_raw:
            st.markdown("<div class='cy-card'>", unsafe_allow_html=True)
            st.markdown("<span class='meta-label'>INCIDENT RAW REPORT SCHEMA</span>", unsafe_allow_html=True)
            st.json(report)
            st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# 4. PAGE: UPLOAD / NEW SCAN
# =====================================================================
elif current_page == "New Scan":
    st.markdown(breadcrumb_html([("Home", None), ("Scans", None), ("New Scan", None)]), unsafe_allow_html=True)
    st.markdown("<h1>Initiate Forensic Inspection</h1>", unsafe_allow_html=True)

    tab_single, tab_batch = st.tabs(["Single Media File Scan", "Batch Directory Ingestion"])

    with tab_single:
        st.markdown('''
        <div class="upload-dropzone">
            <span class="meta-label">TARGET ACQUISITION DROPZONE</span>
            <div style="font-size:15px; color:#EDEDED; margin:10px 0;">Drop image, audio, or video container to execute steganographic analysis</div>
            <div style="display:flex; justify-content:center; gap:8px; margin-top:12px;">
                <span class="badge-media">PNG</span>
                <span class="badge-media">JPG</span>
                <span class="badge-media">WAV</span>
                <span class="badge-media">MKV</span>
                <span class="badge-media">AVI</span>
                <span class="badge-media">MP4</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload file",
            type=["png", "jpg", "jpeg", "wav", "mkv", "avi", "mp4"],
            label_visibility="collapsed",
            key="new_scan_uploader"
        )

        if uploaded is not None:
            sfx = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=sfx) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(uploaded.getvalue())

            try:
                with st.spinner("Executing format routing, deep feature extraction, and calibrated neural inference..."):
                    rep = predictor.predict_file(tmp_path, generate_explanation=True)
                    new_id = add_scan_record(rep, filepath=str(tmp_path))

                st.session_state["active_scan_id"] = new_id
                st.success(f"Inspection complete. Logged incident as {new_id}!")

                st.markdown("<div class='cy-card'>", unsafe_allow_html=True)
                col_res1, col_res2, col_res3 = st.columns([1.5, 1, 1])
                with col_res1:
                    st.markdown(f"**Target:** `{uploaded.name}`")
                    st.markdown(f"**Verdict:** {status_badge_html(rep['verdict'])}", unsafe_allow_html=True)
                with col_res2:
                    st.markdown(f"**Risk Level:** {risk_badge_html(rep['risk_level'])}", unsafe_allow_html=True)
                    st.markdown(f"**Calibrated Score:** {score_pill_html(rep['calibrated_probability'])}", unsafe_allow_html=True)
                with col_res3:
                    if st.button("Inspect Full Finding", type="primary", key="btn_jump_details"):
                        st.session_state["nav_page"] = "Scan Details"
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            except RouterError as e:
                st.error(f"Routing Rejection: {e}")
            except Exception as e:
                st.error(f"Scan Failure: {e}")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

    with tab_batch:
        st.markdown("<div class='cy-card'>", unsafe_allow_html=True)
        st.markdown("<span class='meta-label'>BATCH MEDIA DIRECTORY SCANNER</span>", unsafe_allow_html=True)
        dir_input = st.text_input("Folder Path", value="demo_samples/image", help="Enter directory containing media files.")

        if st.button("Start Batch Ingestion", type="primary", key="btn_run_batch"):
            t_dir = Path(dir_input)
            if not t_dir.exists():
                st.error(f"Path does not exist: {t_dir}")
            else:
                files = [f for f in t_dir.glob("*") if f.is_file() and not f.name.startswith(".")]
                st.info(f"Discovered {len(files)} media files. Ingesting...")
                p_bar = st.progress(0.0)

                batch_items = []
                for idx, f in enumerate(files):
                    try:
                        r = predictor.predict_file(f, generate_explanation=False)
                        sid = add_scan_record(r, filepath=str(f))
                        batch_items.append({
                            "Scan ID": sid,
                            "File": f.name,
                            "Verdict": r["verdict"],
                            "Score": f"{r['calibrated_probability'] * 100:.1f}%",
                            "Risk": r["risk_level"]
                        })
                    except Exception:
                        pass
                    p_bar.progress((idx + 1) / len(files))

                if batch_items:
                    st.success(f"Successfully processed {len(batch_items)} items!")
                    st.dataframe(pd.DataFrame(batch_items), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# 5. PAGE: MODELS & ARCHITECTURE
# =====================================================================
elif current_page == "Models":
    st.markdown(breadcrumb_html([("Home", None), ("Models", None)]), unsafe_allow_html=True)
    st.markdown("<h1>Model Telemetry & Architecture</h1>", unsafe_allow_html=True)

    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown("""
        <div class="cy-card">
            <span class="meta-label">UNIFIED CROSS-MODAL LATENT NETWORK</span>
            <div style="color:#B5B5B5; margin-top:10px; font-size:13px; line-height:1.7;">
                <div>&bull; <strong style="color:#EDEDED;">Latent Dimension</strong>: Z &in; &reals;<sup>128</sup></div>
                <div>&bull; <strong style="color:#EDEDED;">Image Extractor</strong>: D=72 (SRM Residuals, SPAM Markov, DCT AC)</div>
                <div>&bull; <strong style="color:#EDEDED;">Audio Extractor</strong>: D=56 (PoVs Chi-Square, Difference Residuals, Spectral)</div>
                <div>&bull; <strong style="color:#EDEDED;">Video Extractor</strong>: D=64 (Frame Pooling, Acceleration, Temporal Flicker)</div>
                <div>&bull; <strong style="color:#EDEDED;">Auxiliary Head</strong>: Multi-task stego method classifier (evaluated in ablation)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_m2:
        st.markdown(f"""
        <div class="cy-card">
            <span class="meta-label">PROBABILITY CALIBRATION PARAMETERS</span>
            <div style="color:#B5B5B5; margin-top:10px; font-size:13px; line-height:1.7;">
                <div>&bull; <strong style="color:#EDEDED;">Calibration Method</strong>: Post-hoc Temperature Scaling</div>
                <div>&bull; <strong style="color:#EDEDED;">Temperature T</strong>: <code style="color:#E0B341;">{predictor.calibrator.temperature:.4f}</code></div>
                <div>&bull; <strong style="color:#EDEDED;">Target FPR</strong>: <code style="color:#3FB67A;">{predictor.calibrator.target_fpr:.1%}</code></div>
                <div>&bull; <strong style="color:#EDEDED;">Decision Threshold &tau;</strong>: <code style="color:#E5522B;">{predictor.calibrator.decision_threshold:.4f}</code></div>
                <div>&bull; <strong style="color:#EDEDED;">Overall Test ROC-AUC</strong>: <strong style="color:#3FB67A;">0.8990</strong> [0.881, 0.917]</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    p_roc = root_dir / "docs" / "RESULTS" / "roc_curves.png"
    p_cm = root_dir / "docs" / "RESULTS" / "confusion_matrices.png"
    if p_roc.exists() and p_cm.exists():
        cp1, cp2 = st.columns(2)
        with cp1:
            st.image(str(p_roc), caption="Empirical ROC Curves by Format", use_column_width=True)
        with cp2:
            st.image(str(p_cm), caption=f"Confusion Matrices (Threshold = {predictor.calibrator.decision_threshold:.4f})", use_column_width=True)

# =====================================================================
# 6. PAGE: VAULT (DATASET)
# =====================================================================
elif current_page == "Vault":
    st.markdown(breadcrumb_html([("Home", None), ("Vault", None)]), unsafe_allow_html=True)
    st.markdown("<h1>Vault: Real Cover & Stego Repositories</h1>", unsafe_allow_html=True)

    st.markdown("<div class='cy-card'>", unsafe_allow_html=True)
    st.markdown("<span class='meta-label'>PUBLIC DATASET INTEGRITY MANIFEST</span>", unsafe_allow_html=True)
    st.markdown("""
    | Media Format | Public Cover Source | Preprocessing Standard | Stego Generation | Balance |
    | :--- | :--- | :--- | :--- | :---: |
    | **Images** | [Imagenette2-160](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz) | Grayscale, $256\\times 256$ center-crop | LSB-Repl, LSB-Match, Edge-Adaptive, DCT AC | 50% Clean / 50% Stego |
    | **Audio** | [ESC-50](https://github.com/karolpiczak/ESC-50/archive/master.zip) | Mono, $16\\text{ kHz}$, 16-bit WAV, $3.0\\text{ s}$ | LSB-Repl, LSB-Match, Echo Hiding | 50% Clean / 50% Stego |
    | **Video** | Real-Motion Pan/Zoom across images | Lossless RGB FFV1 MKV, $128\\times 128$, $15\\text{ fps}$ | Dense Frame LSB, Frame LSB Matching | 50% Clean / 50% Stego |
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    mf_test = root_dir / "data" / "splits" / "manifest_test.csv"
    if mf_test.exists():
        df_test = pd.read_csv(mf_test)
        st.markdown("<div class='cy-card'>", unsafe_allow_html=True)
        st.markdown(f"<span class='meta-label'>SOURCE-DISJOINT TEST PARTITION ({len(df_test)} SAMPLES)</span>", unsafe_allow_html=True)
        st.dataframe(df_test.head(15), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# 7. PAGE: INTEGRATIONS (API & CLI)
# =====================================================================
elif current_page == "Integrations":
    st.markdown(breadcrumb_html([("Home", None), ("Integrations", None)]), unsafe_allow_html=True)
    st.markdown("<h1>Forensic API & CLI Integrations</h1>", unsafe_allow_html=True)

    c_int1, c_int2 = st.columns(2)
    with c_int1:
        st.markdown("<div class='cy-card'>", unsafe_allow_html=True)
        st.markdown("<span class='meta-label'>FASTAPI REST SERVICE</span>", unsafe_allow_html=True)
        st.markdown("""
        ```bash
        # Launch REST service
        make api
        # Swagger Documentation: http://localhost:8000/docs
        ```
        **Endpoints:**
        - `GET /health` &mdash; Telemetry, GPU/MPS status, model version.
        - `POST /scan` &mdash; Multipart media file upload returning forensic verdict.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_int2:
        st.markdown("<div class='cy-card'>", unsafe_allow_html=True)
        st.markdown("<span class='meta-label'>CLI INFERENCE WORKFLOW</span>", unsafe_allow_html=True)
        st.markdown("""
        ```bash
        # Single file scan
        python scripts/predict.py demo_samples/image/img_src_0001_clean.png

        # Batch scan directory with JSON output
        python scripts/predict.py demo_samples/video/ --json scan_results.json
        ```
        """)
        st.markdown("</div>", unsafe_allow_html=True)
