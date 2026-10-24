"""Reusable Glass UI Component Library for VaultBreaker.

Provides clean glassmorphism components with zero fallback numbers and strictly
audited provenance across all presentation surfaces.
"""

import html
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import plotly.graph_objects as go
from .icons import get_svg_icon

# --- COLOR PALETTE TOKENS ---
ACCENT_ORANGE = "#E5522B"
ACCENT_ORANGE_HOVER = "#F0643C"
ACCENT_ORANGE_BG = "rgba(229, 82, 43, 0.15)"
STATUS_CLEAN = "#3FB67A"
STATUS_AMBER = "#E0B341"
STATUS_STEGO = "#E5522B"
TEXT_PRIMARY = "#EDEDED"
TEXT_SECONDARY = "#A0A0A0"
TEXT_MUTED = "#707070"


def glass_card(content_html: str, title: Optional[str] = None, subtitle: Optional[str] = None) -> str:
    """Renders a standard glassmorphic card container."""
    header_html = ""
    if title:
        sub_html = f'<div style="font-size:12px; color:var(--text-muted); margin-top:2px;">{html.escape(subtitle)}</div>' if subtitle else ""
        header_html = f'''
        <div style="margin-bottom: 14px; border-bottom: 1px solid var(--glass-border); padding-bottom: 8px;">
            <div style="font-size:15px; font-weight:600; color:var(--text-primary);">{html.escape(title)}</div>
            {sub_html}
        </div>
        '''
    return f'''
    <div class="glass-card">
        {header_html}
        {content_html}
    </div>
    '''


def stat_card(title: str, value: str, subtext: str = "", help_text: Optional[str] = None, icon_name: str = "activity") -> str:
    """Renders a glass stat card with title, large value, and help tooltip."""
    icon_svg = get_svg_icon(icon_name, size=18, color=TEXT_SECONDARY)
    title_attr = f'title="{html.escape(help_text)}"' if help_text else ""
    return f'''
    <div class="stat-card" {title_attr}>
        <div class="stat-header">
            <span class="stat-title">{html.escape(title)}</span>
            {icon_svg}
        </div>
        <div class="stat-value">{html.escape(value)}</div>
        <div class="stat-subtext">{html.escape(subtext)}</div>
    </div>
    '''


def verdict_badge(is_stego: bool, text: Optional[str] = None, is_unsupported: bool = False) -> str:
    """Renders a styled verdict chip (clean, stego, or unsupported)."""
    if is_unsupported or (text and "unsupported" in text.lower()):
        lbl = text or "Unsupported Format"
        return f'''
        <div class="verdict-badge" style="background: rgba(140, 140, 140, 0.15); color: #B0B0B0; border: 1px solid rgba(140, 140, 140, 0.3);">
            <span class="verdict-dot" style="background: #A0A0A0; box-shadow: 0 0 8px rgba(160, 160, 160, 0.4);"></span>
            <span>{html.escape(lbl)}</span>
        </div>
        '''
    if is_stego:
        lbl = text or "Suspected hidden data"
        return f'''
        <div class="verdict-badge verdict-stego">
            <span class="verdict-dot dot-stego"></span>
            <span>{html.escape(lbl)}</span>
        </div>
        '''
    lbl = text or "No hidden data detected"
    return f'''
    <div class="verdict-badge verdict-clean">
        <span class="verdict-dot dot-clean"></span>
        <span>{html.escape(lbl)}</span>
    </div>
    '''


def prob_gauge(prob: float, threshold: float = 0.5) -> str:
    """Renders a horizontal glass probability gauge with alert threshold marker."""
    pct = max(0.0, min(100.0, prob * 100.0))
    t_pct = max(0.0, min(100.0, threshold * 100.0))
    bar_color = STATUS_STEGO if prob >= threshold else STATUS_CLEAN
    
    return f'''
    <div style="margin: 8px 0 14px 0;">
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px;">
            <span style="font-size:12px; color:var(--text-secondary);">Calibrated Stego Score</span>
            <span style="font-size:16px; font-weight:600; font-family:'JetBrains Mono'; color:{bar_color};">{pct:.1f}%</span>
        </div>
        <div class="gauge-container" title="Alert threshold at {t_pct:.1f}%">
            <div class="gauge-threshold-line" style="left: {t_pct}%;"></div>
            <div class="gauge-fill" style="width: {pct}%; background-color: {bar_color};"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--text-muted);">
            <span>0% (Clean)</span>
            <span style="color:var(--status-amber);">Alert threshold: {t_pct:.1f}%</span>
            <span>100% (Payload)</span>
        </div>
    </div>
    '''


def ground_truth_strip(
    real_label: int,
    method: str,
    payload_rate: float,
    pred_stego: bool
) -> str:
    """Renders Ground Truth verification strip showing honest model correctness."""
    actual_stego = (real_label == 1)
    is_correct = (pred_stego == actual_stego)
    
    strip_cls = "gt-correct" if is_correct else "gt-wrong"
    verdict_text = "CORRECT VERDICT" if is_correct else "INCORRECT VERDICT"
    verdict_color = STATUS_CLEAN if is_correct else STATUS_AMBER
    
    label_desc = "Hidden data present" if actual_stego else "Clean verified media"
    method_desc = method.replace("_", " ").title() if method != "clean" else "Baseline Clean"
    payload_str = f"{payload_rate:.2f} bpp / rate" if payload_rate > 0 else "None (0.0)"
    
    return f'''
    <div class="ground-truth-strip {strip_cls}">
        <div>
            <div class="gt-label">Ground Truth Verification</div>
            <div class="gt-value" style="color:{verdict_color};">{verdict_text}</div>
        </div>
        <div>
            <div class="gt-label">Actual Condition</div>
            <div class="gt-value">{label_desc}</div>
        </div>
        <div>
            <div class="gt-label">Hiding Algorithm</div>
            <div class="gt-value">{html.escape(method_desc)}</div>
        </div>
        <div>
            <div class="gt-label">Payload Injection Rate</div>
            <div class="gt-value">{payload_str}</div>
        </div>
    </div>
    '''


def score_pill_html(prob: float, threshold: Optional[float] = None) -> str:
    """Return calibrated probability percentage badge for tables and cards."""
    score_val = f"{prob * 100:.1f}%"
    t = threshold if threshold is not None else 0.5
    if prob >= t:
        color = STATUS_STEGO
        bg = "rgba(229, 82, 43, 0.15)"
        cls = "score-high"
    else:
        color = STATUS_CLEAN
        bg = "rgba(63, 182, 122, 0.15)"
        cls = "score-clean"
    return f'<span class="score-pill {cls}" style="padding: 3px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; font-family: \'JetBrains Mono\'; color: {color}; background: {bg}; border: 1px solid {color}40;">{score_val}</span>'


def file_preview(media_type: str, file_path: Optional[str] = None, thumb_path: Optional[str] = None) -> str:
    """Renders thumbnail preview box or media icon placeholder."""
    icon_name = "image" if media_type == "image" else "music" if media_type == "audio" else "video"
    icon_svg = get_svg_icon(icon_name, size=28, color=TEXT_SECONDARY)
    
    if thumb_path and Path(thumb_path).exists():
        # Encode or link thumbnail
        return f'<div class="thumb-box"><img src="{thumb_path}" class="thumb-img" alt="Preview"/></div>'
    
    return f'<div class="thumb-box">{icon_svg}</div>'


def empty_state(title: str, subtext: str, button_label: Optional[str] = None, icon_name: str = "search") -> str:
    """Renders a friendly glass empty state container."""
    icon_svg = get_svg_icon(icon_name, size=32, color=TEXT_SECONDARY)
    btn_html = f'<div style="margin-top: 14px;"><span class="badge-media">{html.escape(button_label)}</span></div>' if button_label else ""
    return f'''
    <div class="empty-state-box">
        {icon_svg}
        <div class="empty-title">{html.escape(title)}</div>
        <div class="empty-subtext">{html.escape(subtext)}</div>
        {btn_html}
    </div>
    '''


def render_entities_table(top_features: List[Tuple[str, float]]) -> str:
    """Renders table of top forensic measurements furthest from normal."""
    rows = []
    for rank, (name, dev) in enumerate(top_features, 1):
        dev_val = f"{dev:+.4f} σ"
        dev_color = STATUS_STEGO if dev > 0 else STATUS_CLEAN
        
        # Categorize domain
        fn = name.lower()
        if "lsb" in fn or "chi" in fn:
            domain = "Bitplane Parity"
        elif "srm" in fn or "kv" in fn or "laplacian" in fn:
            domain = "Spatial Residual"
        elif "spam" in fn or "transition" in fn:
            domain = "Markov Co-occurrence"
        elif "spectral" in fn or "mfcc" in fn or "stft" in fn:
            domain = "Frequency Distribution"
        elif "flicker" in fn or "frame" in fn or "temporal" in fn:
            domain = "Temporal Dynamics"
        elif "dct" in fn:
            domain = "Frequency Transform"
        else:
            domain = "Acoustic / Visual"

        impact = "Elevating Risk" if dev > 1.5 else "Nominal" if abs(dev) < 1.0 else "Suppressing Risk"

        rows.append(f'''
        <tr>
            <td style="font-family:'JetBrains Mono'; color:#8B8B8B;">#{rank:02d}</td>
            <td style="font-weight:600; color:#EDEDED;">{html.escape(name)}</td>
            <td><span style="font-size:11px; padding:2px 8px; border-radius:4px; background:rgba(255,255,255,0.06); color:#A0A0A0;">{domain}</span></td>
            <td style="font-family:'JetBrains Mono'; color:{dev_color}; font-weight:600;">{dev_val}</td>
            <td style="color:#B5B5B5; font-size:12px;">{impact}</td>
        </tr>
        ''')

    return f'''
    <table class="glass-table">
        <thead>
            <tr>
                <th style="width:50px;">Rank</th>
                <th>Measurement</th>
                <th>Domain</th>
                <th>Furthest from Normal (Deviations σ)</th>
                <th>Forensic Impact</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    '''


def render_format_card(title: str, badge: str, m_dict: Optional[Dict[str, Any]], help_auc: str, help_acc: str) -> str:
    """Renders real benchmark card or 'Not available yet' empty state without numeric fallbacks."""
    if m_dict and "roc_auc" in m_dict and "accuracy" in m_dict and "unique_sources" in m_dict:
        auc_str = f"{m_dict['roc_auc']:.4f}"
        acc_str = f"{m_dict['accuracy'] * 100:.1f}%"
        src_str = f"{m_dict['unique_sources']} distinct source files"
        
        warning_html = ""
        n_src = m_dict.get("unique_sources")
        if (n_src is not None and n_src < 30) or m_dict.get("status") == "too few samples to trust" or m_dict.get("warning") == "too few samples to trust":
            warning_html = '<div style="font-size: 10px; color: #E0B341; margin-top: 4px; font-weight: 500;">⚠️ too few samples to trust (&lt; 30 sources)</div>'

        body = (
            f'<div style="display:flex; justify-content:space-between; align-items:baseline; margin: 8px 0 2px 0;">'
            f'<div title="{html.escape(help_auc)}"><div class="stat-value">{auc_str}</div><div style="font-size:11px; color:#8B8B8B;">Area under ROC curve</div></div>'
            f'<div style="text-align:right;" title="{html.escape(help_acc)}"><div style="font-size:18px; font-weight:600; font-family:\'JetBrains Mono\'; color:#EDEDED;">{acc_str}</div><div style="font-size:11px; color:#8B8B8B;">Overall Accuracy</div></div>'
            f'</div>'
            f'<div style="font-size: 11px; color: #8B8B8B; margin-top: 6px;" title="Independent media sources in evaluation partition">{src_str}</div>'
            f'{warning_html}'
        )
    else:
        body = (
            '<div style="margin: 14px 0 8px 0;">'
            '<div style="font-size: 14px; font-weight: 600; color: #E0B341;">Not available yet</div>'
            '<div style="font-size: 11px; color: #8B8B8B; margin-top: 4px;">Run make eval to compute test metrics</div>'
            '</div>'
        )

    return (
        f'<div class="stat-card">'
        f'<div class="stat-header"><span class="stat-title">{html.escape(title)}</span>'
        f'<span style="font-size:10px; font-weight:600; padding:2px 8px; border-radius:9999px; background:rgba(229,82,43,0.15); color:#E5522B; border:1px solid rgba(229,82,43,0.3);">{html.escape(badge)}</span></div>'
        f'{body}'
        f'</div>'
    )


def load_real_metrics(results_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load test evaluation metrics from docs/RESULTS/metrics_deep.json, metrics_handcrafted.json, or metrics.json.
    Returns None if missing or invalid.
    """
    import json
    results_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "RESULTS"
    candidates = []
    if results_path:
        candidates.append(results_path)
    else:
        candidates = [
            results_dir / "metrics_deep.json",
            results_dir / "metrics.json",
            results_dir / "metrics_handcrafted.json",
            results_dir / "smoke" / "metrics_deep.json",
            results_dir / "smoke" / "metrics.json",
        ]

    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and data:
                        if "smoke" in str(p) or data.get("is_smoke"):
                            data["is_smoke"] = True
                        return data
            except Exception:
                continue
    return None


def load_comparison_results(results_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load side-by-side comparison results from docs/RESULTS/comparison.json."""
    import json
    c_path = results_path if results_path else (Path(__file__).resolve().parent.parent.parent / "docs" / "RESULTS" / "comparison.json")
    if not c_path.exists():
        smoke_c_path = Path(__file__).resolve().parent.parent.parent / "docs" / "RESULTS" / "smoke" / "comparison.json"
        if smoke_c_path.exists():
            c_path = smoke_c_path
    if c_path.exists():
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
        except Exception:
            pass
    return None


def create_activity_trend_chart(scans: List[Dict[str, Any]], threshold: Optional[float] = None) -> go.Figure:
    """Bar trend chart of recent scans using actual calibrated threshold."""
    if not scans:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=220)
        return fig

    t = threshold if threshold is not None else 0.5
    sample_scans = scans[:12][::-1]
    ids = [s.get("scan_id") or "" for s in sample_scans]
    probs = []
    for s in sample_scans:
        val = s.get("probability")
        probs.append(float(val) if val is not None else 0.0)
    colors = [STATUS_STEGO if p >= t else STATUS_CLEAN for p in probs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ids,
        y=probs,
        marker=dict(
            color=colors,
            opacity=0.9,
            line=dict(color='rgba(255, 255, 255, 0.1)', width=1)
        ),
        name="Calibrated Stego Score"
    ))

    fig.add_hline(
        y=t,
        line_dash="dot",
        line_color="#E0B341",
        annotation_text=f"Alert threshold ({t:.1%})",
        annotation_font=dict(size=10, color="#E0B341"),
        annotation_position="bottom right"
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B8B8B', family='Inter'),
        yaxis=dict(
            range=[0, 1.05],
            gridcolor='rgba(255, 255, 255, 0.05)',
            zerolinecolor='rgba(255, 255, 255, 0.05)',
            tickfont=dict(size=10)
        ),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.05)',
            tickfont=dict(size=10)
        ),
        margin=dict(l=25, r=15, t=15, b=25),
        height=220,
        showlegend=False
    )
    return fig
