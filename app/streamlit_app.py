import os
import sys
import tempfile
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

# --- PAGE CONFIGURATION & DARK FORENSIC STYLING ---
st.set_page_config(
    page_title="VaultBreaker | Steganography Detection Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Security Tool Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;600;700&display=swap');

    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    header, .stHeader {
        background: transparent !important;
    }

    h1, h2, h3, h4 {
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.5px;
    }

    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(55, 65, 81, 0.6);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 16px;
    }

    .badge-clean {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-stego {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-format {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        margin-right: 6px;
    }

    .terminal-box {
        background: #030712;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL INITIALIZATION WITH CACHE ---
@st.cache_resource
def get_predictor():
    return VaultBreakerPredictor()

predictor = get_predictor()

# --- PROBABILITY GAUGE HELPER ---
def plot_gauge(prob: float, threshold: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "STEGO PROBABILITY (%)", 'font': {'size': 14, 'color': '#94a3b8', 'family': 'JetBrains Mono'}},
        number={'font': {'size': 36, 'color': '#ef4444' if prob >= threshold else '#10b981', 'family': 'JetBrains Mono'}, 'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#374151"},
            'bar': {'color': "#ef4444" if prob >= threshold else "#10b981"},
            'bgcolor': "#111827",
            'borderwidth': 1,
            'bordercolor': "#374151",
            'steps': [
                {'range': [0, threshold * 100], 'color': "rgba(16, 185, 129, 0.15)"},
                {'range': [threshold * 100, 100], 'color': "rgba(239, 68, 68, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "#f59e0b", 'width': 3},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=220,
        margin=dict(l=20, r=20, t=30, b=10)
    )
    return fig

# --- TOP FEATURES CHART ---
def plot_features_bar(top_features):
    names = [f[0] for f in top_features][::-1]
    vals = [f[1] for f in top_features][::-1]
    colors = ['#ef4444' if v > 0 else '#3b82f6' for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation='h',
        marker=dict(color=colors)
    ))
    fig.update_layout(
        title="Top Contributory Indicators (Standardized Deviations)",
        xaxis_title="Standard Deviations from Normal (σ)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='JetBrains Mono'),
        height=260,
        margin=dict(l=10, r=10, t=35, b=20),
        xaxis=dict(gridcolor='#1f2937')
    )
    return fig

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("### 🛡️ VAULTBREAKER")
    st.markdown("*Multi-Modal Steganography Detection*")
    st.markdown("---")
    
    mode = st.radio("Operating Mode", ["Single File Scan", "Batch Directory Scan", "Architecture & Diagnostics"])
    
    st.markdown("---")
    st.markdown("**Engine Status:**")
    st.markdown("<span class='badge-clean'>ONLINE (PyTorch CPU/MPS)</span>", unsafe_allow_html=True)
    st.markdown(f"**Target FPR:** `{predictor.calibrator.target_fpr:.1%}`")
    st.markdown(f"**Calibrated Thresh:** `{predictor.calibrator.decision_threshold:.4f}`")
    st.markdown(f"**Temperature T:** `{predictor.calibrator.temperature:.3f}`")
    st.markdown("---")
    st.caption("VaultBreaker v1.0.0 | Pure Python ML Pipeline")

# --- MODE 1: SINGLE FILE SCAN ---
if mode == "Single File Scan":
    st.title("Forensic Media Inspection")
    st.markdown("Drop an **Image (PNG/JPG)**, **Audio (WAV)**, or **Video (MKV/AVI/MP4)** file to detect covert steganographic channels.")

    uploaded_file = st.file_uploader(
        "Upload Media File",
        type=["png", "jpg", "jpeg", "wav", "mkv", "avi", "mp4"],
        help="Protected by magic-byte verification and file sanitization."
    )

    if uploaded_file is not None:
        # Save temp file
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(uploaded_file.getvalue())

        try:
            with st.spinner("Executing multi-modal feature extraction and neural forward pass..."):
                report = predictor.predict_file(tmp_path, generate_explanation=True)

            # Executive Summary Bar
            st.markdown("---")
            col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

            verdict = report["verdict"]
            is_stego = "STEGO" in verdict

            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"**Target File:** `{report['file_name']}`")
                st.markdown(
                    f"<span class='badge-format'>{report['media_type'].upper()}</span>"
                    f"<span class='badge-format'>{report['format'].upper()}</span>",
                    unsafe_allow_html=True
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                badge_class = "badge-stego" if is_stego else "badge-clean"
                st.markdown(f"**Forensic Verdict:**")
                st.markdown(f"<span class='{badge_class}'>{verdict}</span>", unsafe_allow_html=True)
                st.markdown(f"**Risk Bracket:** `{report['risk_level']}`")
                st.markdown("</div>", unsafe_allow_html=True)

            with col3:
                st.plotly_chart(plot_gauge(report["calibrated_probability"], report["decision_threshold"]), use_container_width=True)

            # TABS FOR INVESTIGATION
            tab_explain, tab_features, tab_raw = st.tabs(["🔬 Forensic Explanation", "📊 Feature Analysis", "⚙️ Raw Metrics & JSON"])

            with tab_explain:
                st.subheader(f"Format-Specific Forensic Artifacts ({report['media_type'].capitalize()})")
                exp = report.get("explanation", {})

                if report["media_type"] == "image":
                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.markdown("**Spatial SRM Residual Heatmap (KV Kernel)**")
                        st.caption("Highlights high-frequency spatial anomalies and artificial noise patterns.")
                        if "residual_heatmap" in exp:
                            st.image(cv2.cvtColor(exp["residual_heatmap"], cv2.COLOR_BGR2RGB), use_column_width=True)
                        else:
                            st.image(uploaded_file, use_column_width=True)
                    with c_right:
                        st.markdown("**LSB Bit-Plane Visualization (Bit 0)**")
                        st.caption("Reveals abnormal bit parity, uniformity, or rectangular carrier blocks.")
                        if "lsb_plane" in exp:
                            st.image(exp["lsb_plane"], clamp=True, use_column_width=True)

                elif report["media_type"] == "audio":
                    st.audio(uploaded_file)
                    c_aud1, c_aud2 = st.columns(2)
                    with c_aud1:
                        st.markdown("**Short-Time Fourier Transform (STFT) Spectrogram**")
                        if "spectrogram_db" in exp:
                            fig_spec = px.imshow(
                                exp["spectrogram_db"],
                                origin='lower',
                                labels=dict(x="Time Frame", y="Frequency Bin", color="dB"),
                                color_continuous_scale="Viridis"
                            )
                            fig_spec.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
                            st.plotly_chart(fig_spec, use_container_width=True)
                    with c_aud2:
                        st.markdown("**LSB Bit Distribution Metrics**")
                        st.metric("LSB Bit-1 Ratio", f"{exp.get('lsb_ones_ratio', 0.5):.4f}")
                        st.metric("LSB Transition Frequency", f"{exp.get('lsb_transition_rate', 0.5):.4f}")

                elif report["media_type"] == "video":
                    st.video(uploaded_file)
                    if "frame_timeline" in exp:
                        st.markdown("**Per-Frame Anomaly Timeline**")
                        timeline = exp["frame_timeline"]
                        fig_time = go.Figure(go.Scatter(
                            x=list(range(len(timeline))),
                            y=timeline,
                            mode='lines+markers',
                            line=dict(color='#ef4444' if is_stego else '#3b82f6', width=2)
                        ))
                        fig_time.update_layout(
                            title="Frame Index vs. Noise Residual Variance",
                            xaxis_title="Sampled Frame Index",
                            yaxis_title="Residual Variance",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#94a3b8', family='JetBrains Mono')
                        )
                        st.plotly_chart(fig_time, use_container_width=True)

            with tab_features:
                st.subheader("Statistical Steganalysis Features")
                if "top_features" in report:
                    st.plotly_chart(plot_features_bar(report["top_features"]), use_container_width=True)

            with tab_raw:
                st.subheader("Inspection Metadata & Export")
                
                # Create serializable report
                export_report = report.copy()
                if "explanation" in export_report:
                    exp_clean = {}
                    for k, v in export_report["explanation"].items():
                        if isinstance(v, (int, float, str, list)):
                            exp_clean[k] = v
                    export_report["explanation"] = exp_clean

                st.json(export_report)
                st.download_button(
                    label="📥 Download Forensic Report (JSON)",
                    data=json.dumps(export_report, indent=2),
                    file_name=f"vaultbreaker_{report['file_name']}_report.json",
                    mime="application/json"
                )

        except RouterError as e:
            st.error(f"Security Alert: File rejected by MediaRouter: {e}")
        except Exception as e:
            st.error(f"Inference Failure: {e}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

# --- MODE 2: BATCH DIRECTORY SCAN ---
elif mode == "Batch Directory Scan":
    st.title("Batch Media Directory Scanner")
    st.markdown("Scan all files in a folder to detect steganographic payloads at scale.")

    dir_input = st.text_input("Folder Path", value="data/generated/images", help="Enter directory path containing media files.")

    if st.button("🚀 Start Batch Scan"):
        target_dir = Path(dir_input)
        if not target_dir.exists():
            st.error(f"Directory not found: {target_dir}")
        else:
            files = [f for f in target_dir.glob("*") if f.is_file() and not f.name.startswith(".")]
            st.info(f"Discovered {len(files)} candidate files. Scanning...")

            batch_results = []
            progress_bar = st.progress(0.0)

            for i, f in enumerate(files):
                try:
                    res = predictor.predict_file(f, generate_explanation=False)
                    batch_results.append({
                        "Filename": f.name,
                        "Format": f"{res['media_type']} ({res['format']})",
                        "Verdict": res["verdict"],
                        "Risk": res["risk_level"],
                        "Calibrated Prob": f"{res['calibrated_probability']:.4f}",
                        "Top Feature": res["top_features"][0][0] if res["top_features"] else "N/A"
                    })
                except Exception as e:
                    pass
                progress_bar.progress((i + 1) / len(files))

            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                
                # Summary Counters
                n_total = len(df_batch)
                n_stego = len(df_batch[df_batch["Verdict"].str.contains("STEGO")])
                n_clean = n_total - n_stego

                c1, c2, c3 = st.columns(3)
                c1.metric("Scanned Files", n_total)
                c2.metric("Clean Covers", n_clean)
                c3.metric("Suspected Stego", n_stego, delta=f"{n_stego/n_total:.1%}")

                st.dataframe(df_batch, use_container_width=True)

                st.download_button(
                    label="📥 Download Batch Results (CSV)",
                    data=df_batch.to_csv(index=False),
                    file_name="vaultbreaker_batch_results.csv",
                    mime="text/csv"
                )

# --- MODE 3: ARCHITECTURE & DIAGNOSTICS ---
else:
    st.title("System Architecture & Diagnostic Telemetry")
    st.markdown("""
    ### Unified Multi-Modal Steganography Detection
    VaultBreaker processes heterogeneous media formats through dedicated per-format extractors
    and MLP encoders that map high-dimensional signals into a **shared 128-dimensional latent space**.
    """)

    st.code("""
    +-------------------+    +--------------------+    +--------------------+
    | Image (D=72)      |    | Audio (D=56)       |    | Video (D=64)       |
    | SRM, SPAM, DCT    |    | LSB, MFCC, LPC     |    | Spatial, Temporal  |
    +---------+---------+    +---------+----------+    +---------+----------+
              |                        |                         |
              v                        v                         v
    +-------------------+    +--------------------+    +--------------------+
    | ImageEncoder      |    | AudioEncoder       |    | VideoEncoder       |
    | 72 -> 256 -> 128  |    | 56 -> 256 -> 128   |    | 64 -> 256 -> 128   |
    +---------+---------+    +---------+----------+    +---------+----------+
              \\                        |                        /
               \\_______________________|_______________________/
                                       |
                         Shared Latent Space (Z ∈ R^128)
                                       |
                                       v
                     +-----------------------------------+
                     | Shared Head: 128 -> 64 -> 1       |
                     | + Auxiliary Method Head           |
                     +-----------------------------------+
    """, language="text")

    # Load and display metrics summary
    metrics_path = Path("docs/RESULTS/metrics.json")
    if metrics_path.exists():
        st.subheader("Active Model Metrics (from docs/RESULTS/metrics.json)")
        with open(metrics_path) as f:
            m_data = json.load(f)
        st.json(m_data.get("overall", {}))
