"""Reusable UI Component Library for VaultBreaker (CyFocus Design System)."""

import html
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .icons import get_svg_icon

# --- COLOR PALETTE TOKENS ---
ACCENT_ORANGE = "#E5522B"
ACCENT_ORANGE_HOVER = "#F0643C"
ACCENT_ORANGE_BG = "rgba(229, 82, 43, 0.2)"
STATUS_CLEAN = "#3FB67A"
STATUS_SUSPICIOUS = "#E0B341"
STATUS_HIGH = "#E5522B"
STATUS_GREY = "#6E6E6E"
TEXT_MUTED = "#8B8B8B"
TEXT_PRIMARY = "#EDEDED"

def status_marker(status_or_verdict: str) -> str:
    """Return colored square dot marker HTML."""
    s = str(status_or_verdict).lower()
    if "critical" in s or "high" in s or "stego" in s or "confirmed" in s:
        color = STATUS_HIGH
    elif "suspect" in s or "medium" in s or "triage" in s or "analyzing" in s:
        color = STATUS_SUSPICIOUS
    elif "clean" in s or "low" in s or "resolved" in s or "benign" in s:
        color = STATUS_CLEAN
    else:
        color = STATUS_GREY
    return f'<span class="severity-marker" style="background-color: {color};"></span>'

def status_badge_html(status: str) -> str:
    """Return styled status chip with marker and label."""
    s = str(status).lower()
    if "critical" in s or "high" in s or "stego" in s or "confirmed" in s:
        color = STATUS_HIGH
        cls = "badge-status-high"
    elif "suspect" in s or "medium" in s or "triage" in s or "analyzing" in s:
        color = STATUS_SUSPICIOUS
        cls = "badge-status-suspicious"
    elif "clean" in s or "low" in s or "resolved" in s:
        color = STATUS_CLEAN
        cls = "badge-status-clean"
    else:
        color = STATUS_GREY
        cls = "badge-media"
    marker = f'<span class="severity-marker" style="background-color: {color};"></span>'
    return f'<span class="badge-status {cls}">{marker}{html.escape(status)}</span>'

def risk_badge_html(risk: str) -> str:
    """Return styled risk bracket badge with matching colors."""
    r = str(risk).lower()
    if "critical" in r or "high" in r or "stego" in r:
        color = STATUS_HIGH
        cls = "badge-status-high"
    elif "medium" in r or "suspect" in r:
        color = STATUS_SUSPICIOUS
        cls = "badge-status-suspicious"
    else:
        color = STATUS_CLEAN
        cls = "badge-status-clean"
    marker = f'<span class="severity-marker" style="background-color: {color};"></span>'
    return f'<span class="badge-status {cls}">{marker}{html.escape(risk)}</span>'

def score_pill_html(prob: float) -> str:
    """Return calibrated probability percentage badge."""
    score_val = f"{prob * 100:.1f}%"
    if prob >= 0.6272:
        cls = "score-high"
    elif prob >= 0.40:
        cls = "score-suspicious"
    else:
        cls = "score-clean"
    return f'<span class="score-pill {cls}">{score_val}</span>'

def breadcrumb_html(items: List[Tuple[str, Optional[str]]]) -> str:
    """
    Renders top navigation breadcrumb.
    items: list of (label, url_or_none)
    """
    parts = []
    for i, (label, link) in enumerate(items):
        is_last = (i == len(items) - 1)
        if is_last:
            parts.append(f'<span class="crumb-active">{html.escape(label)}</span>')
        else:
            parts.append(f'<span class="crumb-item">{html.escape(label)}</span>')
    sep = ' <span class="crumb-sep">/</span> '
    return f'<div class="cy-breadcrumb">{sep.join(parts)}</div>'

def stat_card_html(title: str, value: str, subtext: str = "", delta: Optional[str] = None, delta_type: str = "good", icon_name: str = "activity") -> str:
    """Renders CyFocus stat card."""
    delta_html = ""
    if delta:
        delta_cls = "delta-good" if delta_type == "good" else "delta-bad"
        delta_html = f'<span class="{delta_cls}">{html.escape(delta)}</span> '

    icon_svg = get_svg_icon(icon_name, size=16, color="#8B8B8B")

    return f'''
    <div class="stat-card">
        <div class="stat-header">
            <span class="stat-title">{html.escape(title)}</span>
            {icon_svg}
        </div>
        <div class="stat-value">{html.escape(value)}</div>
        <div class="stat-subtext">
            {delta_html}<span>{html.escape(subtext)}</span>
        </div>
    </div>
    '''

def kanban_card_html(scan: Dict[str, Any], is_selected: bool = False) -> str:
    """Renders interactive Kanban finding card."""
    scan_id = scan.get("scan_id", "SCN-0000")
    filename = scan.get("filename", "unknown_file")
    prob = float(scan.get("probability", 0.0))
    fmt = scan.get("format", "bin").upper()
    created_at = scan.get("created_at", "")[:10]
    initials = "".join([part[0].upper() for part in filename.split("_")[:2]]) or "VB"

    border_style = 'border-color: var(--accent-orange);' if is_selected else ''

    return f'''
    <div class="kanban-card" style="{border_style}">
        <div class="card-top">
            <span class="card-id">{scan_id}</span>
            {score_pill_html(prob)}
        </div>
        <div class="card-file" title="{html.escape(filename)}">{html.escape(filename)}</div>
        <div class="card-footer">
            <span class="badge-media">{fmt}</span>
            <div style="display:flex; align-items:center; gap:6px;">
                <span class="card-date">{created_at}</span>
                <span class="card-avatar">{initials}</span>
            </div>
        </div>
    </div>
    '''

# --- PLOTLY CYBERSECURITY CHARTS ---

def create_radar_chart(report: Dict[str, Any]) -> go.Figure:
    """
    Detection Surface Map Radar Chart:
    Feature groups: LSB Statistics, SRM Residuals, SPAM Markov, Spectral, Temporal, DCT.
    Overlays: Observed vs. Expected (Clean Baseline) in warm orange tones.
    """
    categories = [
        "LSB Statistics",
        "SRM Residuals",
        "SPAM Markov",
        "Spectral Energy",
        "Temporal Flicker",
        "DCT Blockiness"
    ]

    prob = float(report.get("calibrated_probability", 0.1))
    top_feats = report.get("top_features", [])

    # Derive observed values from report features
    observed_scores = [0.15, 0.18, 0.20, 0.14, 0.16, 0.15]
    for feat_name, dev in top_feats:
        fn = feat_name.lower()
        dev_norm = min(1.0, max(0.1, abs(dev) / 4.0))
        if "lsb" in fn or "chi" in fn:
            observed_scores[0] = max(observed_scores[0], dev_norm)
        elif "srm" in fn or "kv" in fn or "laplacian" in fn:
            observed_scores[1] = max(observed_scores[1], dev_norm)
        elif "spam" in fn or "transition" in fn:
            observed_scores[2] = max(observed_scores[2], dev_norm)
        elif "spectral" in fn or "mfcc" in fn or "stft" in fn or "lpc" in fn:
            observed_scores[3] = max(observed_scores[3], dev_norm)
        elif "temporal" in fn or "flicker" in fn or "frame" in fn or "variance" in fn:
            observed_scores[4] = max(observed_scores[4], dev_norm)
        elif "dct" in fn:
            observed_scores[5] = max(observed_scores[5], dev_norm)

    # Scale by overall stego probability
    if prob > 0.5:
        observed_scores = [min(1.0, s * (0.8 + 0.6 * prob)) for s in observed_scores]

    # Baseline (clean expected) is flat nominal ~0.20
    expected_scores = [0.20, 0.18, 0.22, 0.19, 0.18, 0.20]

    # Close loops for radar
    categories_closed = categories + [categories[0]]
    observed_closed = observed_scores + [observed_scores[0]]
    expected_closed = expected_scores + [expected_scores[0]]

    fig = go.Figure()

    # Clean Expected Baseline
    fig.add_trace(go.Scatterpolar(
        r=expected_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(255, 255, 255, 0.04)',
        line=dict(color='rgba(255, 255, 255, 0.25)', width=1, dash='dot'),
        name='Expected Clean Baseline'
    ))

    # Observed Sample Profile
    fill_color = 'rgba(229, 82, 43, 0.35)' if prob >= 0.5 else 'rgba(63, 182, 122, 0.25)'
    line_color = ACCENT_ORANGE if prob >= 0.5 else STATUS_CLEAN

    fig.add_trace(go.Scatterpolar(
        r=observed_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor=fill_color,
        line=dict(color=line_color, width=2.5),
        name='Observed Incident Surface'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                showticklabels=False,
                linecolor='rgba(255, 255, 255, 0.08)',
                gridcolor='rgba(255, 255, 255, 0.06)'
            ),
            angularaxis=dict(
                linecolor='rgba(255, 255, 255, 0.08)',
                gridcolor='rgba(255, 255, 255, 0.06)',
                tickfont=dict(size=12, color='#EDEDED', family='Inter')
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=30, b=30),
        height=400,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.18,
            xanchor='center',
            x=0.5,
            font=dict(size=11, color='#8B8B8B')
        )
    )
    return fig

def create_radial_chart(report: Dict[str, Any]) -> go.Figure:
    """
    Proper Polar Bar Chart of Per-Method Stego Likelihoods.
    Rendered with distinct bar widths and legible typography without clipping.
    """
    methods = [
        "LSB Replacement",
        "LSB Matching",
        "Edge Adaptive",
        "DCT AC Coeffs",
        "Frame LSB",
        "Echo Hiding"
    ]

    prob = float(report.get("calibrated_probability", 0.05))
    top_feats = report.get("top_features", [])
    
    # Synthetic method likelihood weights derived from top indicators
    scores = [0.1, 0.1, 0.1, 0.1, 0.1, 0.05]
    if prob >= 0.5:
        for fn, dev in top_feats:
            f = fn.lower()
            if "dct" in f:
                scores[3] += 0.65
            elif "lsb" in f and "trans" in f:
                scores[1] += 0.55
            elif "chi" in f or "pairs" in f:
                scores[0] += 0.60
            elif "srm" in f or "spam" in f:
                scores[2] += 0.50
            elif "flicker" in f or "frame" in f:
                scores[4] += 0.70
            elif "echo" in f or "autocorr" in f:
                scores[5] += 0.40

    total = sum(scores)
    # Convert to percentage [0..100]
    pct_scores = [round(min(100.0, (s / max(0.01, total)) * (30.0 + 70.0 * prob)), 1) for s in scores]

    fig = go.Figure(go.Barpolar(
        r=pct_scores,
        theta=methods,
        width=[48] * len(methods),
        marker=dict(
            color=pct_scores,
            colorscale=[[0, '#2A1810'], [0.4, '#8F3419'], [1.0, '#E5522B']],
            cmin=0,
            cmax=100,
            line=dict(color='#E5522B', width=1.5)
        ),
        opacity=0.9
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticksuffix='%',
                tickfont=dict(size=10, color='#8B8B8B', family='JetBrains Mono'),
                gridcolor='rgba(255, 255, 255, 0.06)'
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.06)',
                tickfont=dict(size=12, color='#EDEDED', family='Inter')
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=30, b=30),
        height=400,
        showlegend=False
    )
    return fig

def create_media_donut_chart(scans: List[Dict[str, Any]]) -> go.Figure:
    """Donut chart showing breakdown by media type (image, audio, video)."""
    counts = {"Image": 0, "Audio": 0, "Video": 0}
    for s in scans:
        m = str(s.get("media_type", "image")).capitalize()
        if m in counts:
            counts[m] += 1
        else:
            counts["Image"] += 1

    labels = list(counts.keys())
    values = list(counts.values())

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.68,
        marker=dict(
            colors=['#E5522B', '#E0B341', '#3FB67A'],
            line=dict(color='#1B1B1B', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=11, color='#EDEDED', family='Inter'),
        hoverinfo='label+value'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        annotations=[dict(
            text=f"<b>{sum(values)}</b><br><span style='font-size:10px; color:#8B8B8B;'>SCANS</span>",
            x=0.5, y=0.5,
            font=dict(size=18, color='#EDEDED', family='JetBrains Mono'),
            showarrow=False
        )]
    )
    return fig

def create_activity_trend_chart(scans: List[Dict[str, Any]]) -> go.Figure:
    """Bar/line trend chart of recent scans."""
    if not scans:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=240)
        return fig

    # Group by scan ID order or timestamps
    sample_scans = scans[:12][::-1]
    ids = [s.get("scan_id", f"S-{i}") for i, s in enumerate(sample_scans)]
    probs = [float(s.get("probability", 0.0)) for s in sample_scans]
    colors = [STATUS_HIGH if p >= 0.5 else STATUS_CLEAN for p in probs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ids,
        y=probs,
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(color='rgba(255, 255, 255, 0.1)', width=1)
        ),
        name="Calibrated Stego Score"
    ))

    # Add 0.5 threshold line
    fig.add_hline(
        y=0.6389,
        line_dash="dot",
        line_color="#E0B341",
        annotation_text="Decision Threshold (0.639)",
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
        height=240,
        showlegend=False
    )
    return fig

def render_entities_table(top_features: List[Tuple[str, float]]) -> str:
    """Renders CyFocus entities-style table of top contributory indicators."""
    rows = []
    for rank, (name, dev) in enumerate(top_features, 1):
        dev_val = f"{dev:+.4f} σ"
        dev_color = STATUS_HIGH if dev > 0 else STATUS_CLEAN
        
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
            <td><span class="badge-media">{domain}</span></td>
            <td style="font-family:'JetBrains Mono'; color:{dev_color}; font-weight:600;">{dev_val}</td>
            <td style="color:#B5B5B5; font-size:12px;">{impact}</td>
        </tr>
        ''')

    return f'''
    <table class="cy-table">
        <thead>
            <tr>
                <th style="width:40px;">Rank</th>
                <th>Forensic Indicator</th>
                <th>Domain</th>
                <th>Deviation (σ)</th>
                <th>Risk Impact</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    '''
