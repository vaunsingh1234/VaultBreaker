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


def dedent_html(html_str: str) -> str:
    """Ensure HTML string contains no lines with leading indentation or blank lines that cause CommonMark code block parsing."""
    if not html_str:
        return ""
    lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    return "\n".join(lines)


def glass_card(content_html: str, title: Optional[str] = None, subtitle: Optional[str] = None) -> str:
    """Renders a standard glassmorphic card container."""
    header_html = ""
    if title:
        sub_html = f'<div style="font-size:12px; color:var(--text-muted); margin-top:2px;">{html.escape(subtitle)}</div>' if subtitle else ""
        header_html = (
            '<div style="margin-bottom: 14px; border-bottom: 1px solid var(--glass-border); padding-bottom: 8px;">'
            f'<div style="font-size:15px; font-weight:600; color:var(--text-primary);">{html.escape(title)}</div>'
            f'{sub_html}</div>'
        )
    return dedent_html(f'''
    <div class="glass-card">
        {header_html}
        {content_html}
    </div>
    ''')


def stat_card(
    title: str,
    value: str,
    subtext: str = "",
    help_text: Optional[str] = None,
    icon_name: str = "activity",
    items: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Renders a glass stat card with title, large value or small item list, and help tooltip."""
    icon_svg = get_svg_icon(icon_name, size=18, color=TEXT_SECONDARY)
    title_attr = f'title="{html.escape(help_text)}"' if help_text else ""
    
    if items:
        rows_html = "".join(
            f'<div class="stat-item-row"><span class="stat-item-label">{html.escape(lbl)}</span>'
            f'<span class="stat-item-val">{html.escape(val)}</span></div>'
            for lbl, val in items
        )
        val_html = f'<div class="stat-items-list">{rows_html}</div>'
    else:
        val_html = f'<div class="stat-value">{html.escape(value)}</div>'

    subtext_html = f'<div class="stat-subtext">{html.escape(subtext)}</div>' if subtext else ""
    
    return dedent_html(
        f'<div class="stat-card" {title_attr}>\n'
        f'<div class="stat-header">\n'
        f'<span class="stat-title">{html.escape(title)}</span>\n'
        f'{icon_svg}\n'
        f'</div>\n'
        f'{val_html}\n'
        f'{subtext_html}\n'
        f'</div>'
    )


def verdict_badge(is_stego: bool, text: Optional[str] = None, is_unsupported: bool = False) -> str:
    """Renders a styled verdict chip (clean, stego, or unsupported)."""
    if is_unsupported or (text and "unsupported" in text.lower()):
        lbl = text or "Unsupported Format"
        return (
            '<div class="verdict-badge" style="background: rgba(140, 140, 140, 0.15); color: #B0B0B0; border: 1px solid rgba(140, 140, 140, 0.3);">'
            '<span class="verdict-dot" style="background: #A0A0A0; box-shadow: 0 0 8px rgba(160, 160, 160, 0.4);"></span>'
            f'<span>{html.escape(lbl)}</span></div>'
        )
    if is_stego:
        lbl = text or "Suspected hidden data"
        return (
            '<div class="verdict-badge verdict-stego">'
            '<span class="verdict-dot dot-stego"></span>'
            f'<span>{html.escape(lbl)}</span></div>'
        )
    lbl = text or "No hidden data detected"
    return (
        '<div class="verdict-badge verdict-clean">'
        '<span class="verdict-dot dot-clean"></span>'
        f'<span>{html.escape(lbl)}</span></div>'
    )


def prob_gauge(prob: float, threshold: float = 0.5) -> str:
    """Renders a horizontal glass probability gauge with alert threshold marker."""
    pct = max(0.0, min(100.0, prob * 100.0))
    t_pct = max(0.0, min(100.0, threshold * 100.0))
    bar_color = STATUS_STEGO if prob >= threshold else STATUS_CLEAN
    
    return dedent_html(f'''
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
    ''')


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
    
    return dedent_html(f'''
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
    ''')


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
    return dedent_html(f'''
    <div class="empty-state-box">
        {icon_svg}
        <div class="empty-title">{html.escape(title)}</div>
        <div class="empty-subtext">{html.escape(subtext)}</div>
        {btn_html}
    </div>
    ''')


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

        rows.append(
            f'<tr>'
            f'<td style="font-family:\'JetBrains Mono\'; color:#8B8B8B;">#{rank:02d}</td>'
            f'<td style="font-weight:600; color:#EDEDED;">{html.escape(name)}</td>'
            f'<td><span style="font-size:11px; padding:2px 8px; border-radius:4px; background:rgba(255,255,255,0.06); color:#A0A0A0;">{domain}</span></td>'
            f'<td style="font-family:\'JetBrains Mono\'; color:{dev_color}; font-weight:600;">{dev_val}</td>'
            f'<td style="color:#B5B5B5; font-size:12px;">{impact}</td>'
            f'</tr>'
        )

    return dedent_html(f'''
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
    ''')


def render_format_card(
    title: str,
    badge: str,
    m_dict: Optional[Dict[str, Any]],
    help_auc: str,
    help_acc: str,
    is_smoke: bool = False
) -> str:
    """Renders real benchmark card or 'Not available yet' empty state without numeric fallbacks."""
    smoke_badge = '<div class="badge-smoke">Smoke test, not real performance</div>\n' if is_smoke else ""
    if m_dict and "roc_auc" in m_dict and "accuracy" in m_dict and "unique_sources" in m_dict:
        auc_str = f"{m_dict['roc_auc']:.4f}"
        acc_str = f"{m_dict['accuracy'] * 100:.1f}%"
        src_str = f"{m_dict['unique_sources']:,} distinct source files"
        
        warning_html = ""
        n_src = m_dict.get("unique_sources")
        if (n_src is not None and n_src < 30) or m_dict.get("status") == "too few samples to trust" or m_dict.get("warning") == "too few samples to trust":
            warning_html = '<div style="font-size: 10px; color: #E0B341; margin-top: 4px; font-weight: 500;">⚠️ too few samples to trust (&lt; 30 sources)</div>'

        body = (
            f'{smoke_badge}'
            f'<div style="display:flex; justify-content:space-between; align-items:baseline; margin: 6px 0 2px 0;">\n'
            f'<div title="{html.escape(help_auc)}"><div class="stat-value">{auc_str}</div><div style="font-size:11px; color:#8B8B8B;">Area under ROC curve</div></div>\n'
            f'<div style="text-align:right;" title="{html.escape(help_acc)}"><div style="font-size:18px; font-weight:600; font-family:\'JetBrains Mono\'; color:#EDEDED;">{acc_str}</div><div style="font-size:11px; color:#8B8B8B;">Overall Accuracy</div></div>\n'
            f'</div>\n'
            f'<div style="font-size: 11px; color: #8B8B8B; margin-top: 4px;" title="Independent media sources in evaluation partition">{src_str}</div>\n'
            f'{warning_html}'
        )
    else:
        body = (
            f'{smoke_badge}'
            f'<div style="margin: 12px 0 6px 0;">\n'
            f'<div style="font-size: 14px; font-weight: 600; color: #E0B341;">Not available yet</div>\n'
            f'<div style="font-size: 11px; color: #8B8B8B; margin-top: 4px;">Run make eval to compute test metrics</div>\n'
            f'</div>'
        )

    return dedent_html(
        f'<div class="stat-card">\n'
        f'<div class="stat-header"><span class="stat-title">{html.escape(title)}</span>\n'
        f'<span style="font-size:10px; font-weight:600; padding:2px 8px; border-radius:9999px; background:rgba(229,82,43,0.15); color:#E5522B; border:1px solid rgba(229,82,43,0.3);">{html.escape(badge)}</span></div>\n'
        f'{body}\n'
        f'</div>'
    )


def build_metrics_table_html(metrics_data: Dict[str, Any], is_smoke: bool = False) -> str:
    """Build single dedented HTML table for evaluation test-set benchmark metrics.
    Ensures zero lines start with 4+ spaces and no unrendered code blocks occur.
    """
    badge_html = '<div class="badge-smoke">Smoke test, not real performance</div>\n' if is_smoke else ""
    per_fmt = metrics_data.get("per_format") or {}
    overall_m = metrics_data.get("overall") or {}
    
    rows_data = [
        ("Image", per_fmt.get("image") or {}),
        ("Audio", per_fmt.get("audio") or {}),
        ("Video", per_fmt.get("video") or {}),
        ("Overall", overall_m),
    ]
    
    lines = [
        badge_html + '<table class="glass-table">',
        '<thead>',
        '<tr>',
        '<th>Format</th>',
        '<th>Accuracy</th>',
        '<th>ROC-AUC (95% CI)</th>',
        '<th>Recall @1% FPR</th>',
        '<th>Recall @5% FPR</th>',
        '<th>Recall @10% FPR</th>',
        '<th>Test files</th>',
        '</tr>',
        '</thead>',
        '<tbody>',
    ]
    
    for name, m in rows_data:
        trust_badge = ""
        n_src = m.get("unique_sources")
        if (n_src is not None and n_src < 30) or m.get("status") == "too few samples to trust" or m.get("warning") == "too few samples to trust":
            trust_badge = ' <span style="font-size:10px; color:#E0B341; padding:2px 6px; border-radius:4px; background:rgba(224,179,65,0.15); border:1px solid rgba(224,179,65,0.3); margin-left:6px; font-weight:normal;">too few samples to trust</span>'

        raw_acc = m.get("accuracy")
        acc = f"{float(raw_acc) * 100:.1f}%" if raw_acc is not None else "N/A"
        
        raw_auc = m.get("roc_auc")
        auc = f"{float(raw_auc):.4f}" if raw_auc is not None else "N/A"
        ci = m.get("roc_auc_ci95")
        ci_str = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if isinstance(ci, list) and len(ci) == 2 else ""
        auc_cell = f"{auc}<span style='font-size:11px; color:#8B8B8B;'>{ci_str}</span>" if raw_auc is not None else "N/A"

        rfpr = m.get("recalls_at_fpr") or {}
        r1 = rfpr.get("recall_at_1pct_fpr", m.get("recall_1pct_fpr"))
        r1_str = f"{float(r1) * 100:.1f}%" if r1 is not None else "N/A"
        
        r5 = rfpr.get("recall_at_5pct_fpr", m.get("recall_5pct_fpr"))
        r5_str = f"{float(r5) * 100:.1f}%" if r5 is not None else "N/A"
        
        r10 = rfpr.get("recall_at_10pct_fpr", m.get("recall_10pct_fpr"))
        r10_str = f"{float(r10) * 100:.1f}%" if r10 is not None else "N/A"

        n_files = m.get("sample_count")
        if n_files is not None and n_src is not None:
            src_str = f"{n_files:,} files ({n_src:,} sources)"
        elif n_files is not None:
            src_str = f"{n_files:,} files"
        else:
            src_str = "N/A"

        lines.append('<tr>')
        lines.append(f'<td style="font-weight:600; color:#EDEDED;">{name}{trust_badge}</td>')
        lines.append(f'<td style="font-family:\'JetBrains Mono\'; font-weight:600; color:#EDEDED;">{acc}</td>')
        lines.append(f'<td style="font-family:\'JetBrains Mono\'; color:#E0B341;">{auc_cell}</td>')
        lines.append(f'<td style="font-family:\'JetBrains Mono\'; color:#3FB67A;">{r1_str}</td>')
        lines.append(f'<td style="font-family:\'JetBrains Mono\'; color:#3FB67A;">{r5_str}</td>')
        lines.append(f'<td style="font-family:\'JetBrains Mono\'; color:#3FB67A;">{r10_str}</td>')
        lines.append(f'<td style="color:#B5B5B5; font-size:12px;">{src_str}</td>')
        lines.append('</tr>')

    lines.append('</tbody>')
    lines.append('</table>')
    return "\n".join(lines)


def build_comparison_table_html(comparison_data: Dict[str, Any], is_smoke: bool = False) -> str:
    """Build single dedented HTML table for side-by-side model comparison on identical test split."""
    badge_html = '<div class="badge-smoke">Smoke test, not real performance</div>\n' if is_smoke else ""
    comp_dict = comparison_data.get("comparison") if isinstance(comparison_data.get("comparison"), dict) else comparison_data
    
    lines = [
        badge_html + '<table class="glass-table">',
        '<thead>',
        '<tr>',
        '<th>Format</th>',
        '<th>Deep model</th>',
        '<th>Handcrafted MLP</th>',
        '<th>Gradient boosting</th>',
        '</tr>',
        '</thead>',
        '<tbody>',
    ]

    for fmt_key in ["image", "audio", "video"]:
        fmt_data = comp_dict.get(fmt_key, {})
        fmt_label = fmt_key.capitalize()

        def _fmt_cell(val):
            if isinstance(val, dict):
                auc_val = val.get("roc_auc")
                r5_val = val.get("recall_at_5pct_fpr")
                acc_val = val.get("accuracy")
                auc_s = f"{float(auc_val):.4f}" if auc_val is not None else "N/A"
                r5_s = f"{float(r5_val) * 100:.1f}%" if r5_val is not None else "N/A"
                acc_s = f"{float(acc_val) * 100:.1f}%" if acc_val is not None else "N/A"
                return (
                    f'<div style="font-weight:600; color:var(--text-primary);">'
                    f'ROC-AUC: <span style="color:var(--status-amber);">{auc_s}</span>'
                    f'</div>'
                    f'<div style="font-size:11px; color:var(--text-muted); margin-top:2px;">'
                    f'Acc: {acc_s} · R@5%: {r5_s}'
                    f'</div>'
                )
            elif val == "not run" or not val:
                return '<span style="color:#8B8B8B; font-style:italic;">not run</span>'
            return str(val)

        deep_cell = _fmt_cell(fmt_data.get("deep_model"))
        mlp_cell = _fmt_cell(fmt_data.get("handcrafted_mlp"))
        gb_cell = _fmt_cell(fmt_data.get("gradient_boosting"))

        lines.append('<tr>')
        lines.append(f'<td style="font-weight:600; color:#EDEDED;">{fmt_label}</td>')
        lines.append(f'<td style="font-family:\'JetBrains Mono\'; font-size:12px;">{deep_cell}</td>')
        lines.append(f'<td style="font-family:\'JetBrains Mono\'; font-size:12px;">{mlp_cell}</td>')
        lines.append(f'<td style="font-family:\'JetBrains Mono\'; font-size:12px;">{gb_cell}</td>')
        lines.append('</tr>')

    lines.append('</tbody>')
    lines.append('</table>')
    lines.append('</div>')
    return "\n".join(lines)


def build_history_table_html(scans: List[Dict[str, Any]], decision_threshold: float = 0.5) -> str:
    """Build dedented HTML table for historical scans.
    Ensures zero lines start with 4+ spaces so CommonMark never treats rows as code blocks.
    """
    lines = [
        '<div class="glass-table-container">',
        '<table class="glass-table">',
        '<thead>',
        '<tr>',
        '<th>Scan ID</th>',
        '<th>Timestamp</th>',
        '<th>File Name</th>',
        '<th>Modality</th>',
        '<th>Calibrated Score</th>',
        '<th>Ground Truth Verification</th>',
        '</tr>',
        '</thead>',
        '<tbody>',
    ]
    for s in scans:
        sid = s.get("scan_id") or ""
        dt = (s.get("created_at") or "")[:19]
        fn = s.get("filename") or "unknown"
        mtype = (s.get("media_type") or "image").upper()
        prob_val = s.get("probability")
        prob = float(prob_val) if prob_val is not None else 0.0
        v_chip = score_pill_html(prob, threshold=decision_threshold)

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

        lines.append('<tr>')
        lines.append(f'<td style="font-family:\'JetBrains Mono\'; font-weight:600; color:#EDEDED;">{sid}</td>')
        lines.append(f'<td style="color:#A0A0A0; font-size:12px;">{dt}</td>')
        lines.append(f'<td style="font-weight:600; color:#EDEDED;">{html.escape(fn)}</td>')
        lines.append(f'<td><span class="badge-media">{mtype}</span></td>')
        lines.append(f'<td>{v_chip}</td>')
        lines.append(f'<td>{gt_display}</td>')
        lines.append('</tr>')

    lines.append('</tbody>')
    lines.append('</table>')
    lines.append('</div>')
    return "\n".join(lines)


def load_real_metrics(results_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load test evaluation metrics from docs/RESULTS/metrics_deep.json, metrics_handcrafted.json, or metrics.json.
    Strictly rejects any smoke runs, smoke profiles, or files in smoke directories.
    Returns None if missing, smoke, or invalid.
    """
    import json
    results_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "RESULTS"
    if results_path:
        candidates = [Path(results_path)]
    else:
        candidates = [
            results_dir / "metrics_deep.json",
            results_dir / "metrics.json",
            results_dir / "metrics_handcrafted.json",
        ]

    for p in candidates:
        if p.exists():
            # Never load files from a smoke directory
            if "smoke" in p.parts:
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and data:
                        # Reject if marked as smoke or smoke profile
                        if data.get("is_smoke") is True or str(data.get("profile", "")).lower() == "smoke":
                            continue
                        return data
            except Exception:
                continue
    return None


def load_display_metrics(results_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load test metrics for UI presentation.
    First checks for real training runs (laptop/gpu profile, is_smoke=False).
    If none found, checks for smoke runs and returns them marked with is_smoke=True.
    """
    real = load_real_metrics(results_path)
    if real is not None:
        real["is_smoke"] = False
        return real

    # Check for smoke run to allow displaying smoke badge
    import json
    results_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "RESULTS"
    if results_path:
        candidates = [Path(results_path)]
    else:
        candidates = [
            results_dir / "metrics_deep.json",
            results_dir / "metrics.json",
            results_dir / "smoke" / "metrics_deep.json",
            results_dir / "smoke" / "metrics.json",
        ]

    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and data:
                        is_smoke = bool(data.get("is_smoke")) or str(data.get("profile", "")).lower() == "smoke" or not data.get("profile")
                        data["is_smoke"] = is_smoke
                        return data
            except Exception:
                continue
    return None


def load_comparison_results(results_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load side-by-side comparison results from docs/RESULTS/comparison.json.
    Strictly rejects any smoke run comparison file.
    """
    import json
    c_path = Path(results_path) if results_path else (Path(__file__).resolve().parent.parent.parent / "docs" / "RESULTS" / "comparison.json")
    if c_path.exists():
        if "smoke" in c_path.parts:
            return None
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    if data.get("is_smoke") is True or str(data.get("profile", "")).lower() == "smoke":
                        return None
                    return data
        except Exception:
            pass
    return None


def load_display_comparison_results(results_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load side-by-side comparison results for UI presentation."""
    real = load_comparison_results(results_path)
    if real is not None:
        real["is_smoke"] = False
        return real

    import json
    results_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "RESULTS"
    c_path = Path(results_path) if results_path else (results_dir / "comparison.json")
    if not c_path.exists():
        c_path = results_dir / "smoke" / "comparison.json"
        
    if c_path.exists():
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    is_smoke = bool(data.get("is_smoke")) or str(data.get("profile", "")).lower() == "smoke" or not data.get("profile")
                    data["is_smoke"] = is_smoke
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


def get_active_checkpoint_rows(root_dir: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """Load actual active model details per format from model_selection.json, calibrator, and model files.
    Returns:
        (rows, target_fpr, cal_per_fmt_dict)
    """
    import json
    from datetime import datetime
    base_dir = root_dir if root_dir else (Path(__file__).resolve().parent.parent.parent)
    models_dir = base_dir / "models_saved"
    selection_file = models_dir / "model_selection.json"
    calibrator_file = models_dir / "calibrator_per_format.json"
    
    selection_data = {}
    if selection_file.exists():
        try:
            with open(selection_file, "r", encoding="utf-8") as f:
                selection_data = json.load(f)
        except Exception:
            pass
            
    cal_data = {}
    if calibrator_file.exists():
        try:
            with open(calibrator_file, "r", encoding="utf-8") as f:
                cal_data = json.load(f)
        except Exception:
            pass

    formats_info = selection_data.get("formats", {})
    rows = []
    
    # Target FPR dynamically from calibrator or per-format calibration (default 0.05)
    target_fpr = 0.05
    
    for fmt in ["image", "audio", "video"]:
        fmt_sel = formats_info.get(fmt, {})
        sel_key = fmt_sel.get("selected_model", "deep_model")
        
        # Determine actual file
        if sel_key == "deep_model":
            model_file = models_dir / "deep_model.pt"
            display_model = "deep_model.pt"
        elif sel_key == "gradient_boosting":
            model_file = models_dir / f"baseline_{fmt}_GradientBoosting.joblib"
            display_model = model_file.name if model_file.exists() else f"baseline_{fmt}_GBDT.joblib"
        elif sel_key == "handcrafted_mlp":
            model_file = models_dir / "unified_model.pt"
            display_model = "unified_model.pt"
        else:
            model_file = models_dir / "deep_model.pt"
            display_model = model_file.name

        # File date
        if model_file.exists():
            mod_dt = datetime.fromtimestamp(model_file.stat().st_mtime).strftime("%Y-%m-%d")
        else:
            mod_dt = str(selection_data.get("timestamp", "N/A"))[:10]

        # Threshold from calibrator
        fmt_cal = cal_data.get(fmt, {})
        thresh = fmt_cal.get("decision_threshold")
        if thresh is None:
            thresh = 0.5
        if "target_fpr" in fmt_cal:
            target_fpr = fmt_cal["target_fpr"]

        rows.append({
            "format": fmt.capitalize(),
            "model": display_model,
            "threshold": float(thresh),
            "date": mod_dt,
        })
    return rows, target_fpr, cal_data
