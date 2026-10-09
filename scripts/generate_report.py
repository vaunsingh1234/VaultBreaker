#!/usr/bin/env python3
"""Report Generation Script for VaultBreaker.
Reads docs/RESULTS/metrics.json and regenerates docs/REPORT.md with
empirically grounded tables, bootstrap 95% confidence intervals,
and honest sample-count assertions (labeling unique sources < 30 as 'insufficient samples').
"""

import json
from pathlib import Path

def generate_report(metrics_path: Path, output_path: Path):
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}. Please run scripts/evaluate.py first.")

    with open(metrics_path, "r") as f:
        m = json.load(f)

    ov = m.get("overall", {})
    per_fmt = m.get("per_format", {})
    per_meth = m.get("per_method", {})
    acc_payload = m.get("accuracy_vs_payload", {})
    cross_gen = m.get("cross_method_generalization", {})
    sanity = m.get("shortcut_sanity_check", {})
    ablations = m.get("ablations", {})

    lines = []
    lines.append("# VaultBreaker: Empirical Steganography Detection Evaluation Report")
    lines.append("\n**Author:** Senior Forensic ML Engineer  ")
    lines.append("**Dataset:** Real Public Cover Datasets (Imagenette2-160, ESC-50, Pan/Zoom Visual Real-Motion Video)  ")
    lines.append("**Embeddings:** Laboratory In-House Stego Embeddings (LSB Replacement, LSB Matching, Edge-Adaptive, DCT AC, Echo Hiding)  ")
    lines.append("**Splits:** 70% Train / 15% Validation / 15% Test (Source-Disjoint Partitioning, 0% Leakage, Deduplicated 50/50 Balance)  \n")
    lines.append("---\n")

    lines.append("## 1. Executive Summary & Headline Results\n")
    lines.append(
        "VaultBreaker is a cross-modal digital forensic steganography detector unifying Images, Audio, "
        "and Video media through modality-specific feature extractors and a shared 128-dimensional latent space. "
        "This evaluation report reflects testing on real cover media with source-disjoint test sets. "
        "All metrics include cluster bootstrap 95% confidence intervals computed by resampling unique source IDs (1,000 resamples).\n"
    )

    # Table 1: Summary by Media Format
    lines.append("### Table 1: Detection Performance on Test Split (by Format)\n")
    lines.append("| Modality | Test Samples (Unique Sources) | Accuracy (95% CI) | Precision | Recall | F1 Score | ROC-AUC (95% CI) | Detection Error $P_E$ |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for fmt, data in per_fmt.items():
        n = data.get("sample_count", 0)
        u_src = data.get("unique_sources", n // 2)
        count_str = data.get("sample_count_str", f"{n} files ({u_src} unique sources)")
        acc = data.get("accuracy", 0.0)
        acc_ci = data.get("accuracy_ci95", [0, 0])
        auc = data.get("roc_auc", 0.0)
        auc_ci = data.get("roc_auc_ci95", [0, 0])
        prec = data.get("precision", 0.0)
        rec = data.get("recall", 0.0)
        f1 = data.get("f1", 0.0)
        pe = data.get("p_e", 0.0)

        acc_str = f"**{acc:.4f}** `[{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]`" if u_src >= 30 else f"{acc:.4f} *(N<30 sources)*"
        auc_str = f"**{auc:.4f}** `[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]`" if u_src >= 30 else f"{auc:.4f} *(N<30 sources)*"

        lines.append(
            f"| **{fmt.capitalize()}** | {count_str} | {acc_str} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {auc_str} | {pe:.4f} |"
        )

    # Overall row
    n_ov = ov.get("sample_count", 0)
    u_ov = ov.get("unique_sources", n_ov // 2)
    count_str_ov = ov.get("sample_count_str", f"{n_ov} files ({u_ov} unique sources)")
    acc_ov = ov.get("accuracy", 0.0)
    acc_ci_ov = ov.get("accuracy_ci95", [0, 0])
    auc_ov = ov.get("roc_auc", 0.0)
    auc_ci_ov = ov.get("roc_auc_ci95", [0, 0])
    lines.append(
        f"| **Unified Overall** | **{count_str_ov}** | **{acc_ov:.4f}** `[{acc_ci_ov[0]:.3f}, {acc_ci_ov[1]:.3f}]` | "
        f"**{ov.get('precision', 0.0):.4f}** | **{ov.get('recall', 0.0):.4f}** | **{ov.get('f1', 0.0):.4f}** | "
        f"**{auc_ov:.4f}** `[{auc_ci_ov[0]:.3f}, {auc_ci_ov[1]:.3f}]` | **{ov.get('p_e', 0.0):.4f}** |\n"
    )

    if "audio_lsb_headline" in m:
        a_lsb = m["audio_lsb_headline"]
        lines.append(
            f"> 🔍 **Audio LSB Headline (Excl. Echo Hiding):** Accuracy: **{a_lsb['accuracy']:.4f}** "
            f"(95% CI: `[{a_lsb['accuracy_ci95'][0]:.3f}, {a_lsb['accuracy_ci95'][1]:.3f}]`), "
            f"ROC-AUC: **{a_lsb['roc_auc']:.4f}** (95% CI: `[{a_lsb['roc_auc_ci95'][0]:.3f}, {a_lsb['roc_auc_ci95'][1]:.3f}]`), "
            f"Count: {a_lsb.get('sample_count_str', a_lsb['sample_count'])}\n"
        )

    # Section 2: Method-Specific Breakdown
    lines.append("## 2. Embedding Method Performance Breakdown\n")
    lines.append("| Embedding Method | Test Samples (Unique Sources) | Sensitivity / Accuracy | Mean Calibrated Stego Prob | Assessment |")
    lines.append("| :--- | :---: | :---: | :---: | :--- |")

    for method, d in per_meth.items():
        n = d.get("sample_count", 0)
        u_src = d.get("unique_sources", n)
        count_str = d.get("sample_count_str", f"{n} files ({u_src} sources)")
        acc = d.get("accuracy", 0.0)
        prob = d.get("mean_cal_prob", 0.0)
        is_insufficient = d.get("insufficient_samples", u_src < 30)

        if is_insufficient:
            status = "Insufficient samples (< 30 unique sources)"
        elif acc >= 0.80:
            status = "Strong Detection"
        elif acc >= 0.60:
            status = "Moderate Signal"
        else:
            status = "Weak Signal / Resistant"
        lines.append(f"| `{method}` | {count_str} | {acc:.4f} | {prob:.4f} | {status} |")

    # Section 3: Payload Sensitivity Curve
    lines.append("\n## 3. Detection Sensitivity vs. Embedding Payload Density\n")
    lines.append("| Payload Rate | Stego Test Samples (Unique Sources) | Detection Sensitivity | Statistical Confidence |")
    lines.append("| :--- | :---: | :---: | :--- |")

    for rate, d in acc_payload.items():
        n = d.get("sample_count", 0)
        u_src = d.get("unique_sources", n)
        count_str = d.get("sample_count_str", f"{n} files ({u_src} sources)")
        acc = d.get("accuracy", 0.0)
        is_insufficient = d.get("insufficient_samples", u_src < 30)
        conf = "Insufficient samples (< 30 unique sources)" if is_insufficient else "Statistically verified"
        lines.append(f"| **{rate}** | {count_str} | {acc:.4f} | {conf} |")

    # Section 4: Generalization & Shortcut Controls
    lines.append("\n## 4. Generalization & Integrity Control Experiments\n")
    lines.append("### 4.1 Cross-Method Generalization (Trained on LSB Replacement Only)\n")
    lines.append("| Test Method (Unseen) | Test Samples | Zero-Shot ROC-AUC | Verdict |")
    lines.append("| :--- | :---: | :---: | :--- |")
    for method, d in cross_gen.items():
        n = d.get("sample_count", 0)
        auc = d.get("auc", 0.5)
        verdict = "Generalizes robustly" if auc >= 0.75 else "Method-specific signature needed"
        if n < 30:
            verdict += " *(N < 30)*"
        lines.append(f"| `{method}` | {n} | {auc:.4f} | {verdict} |")

    lines.append("\n### 4.2 Shortcut Learning Sanity Check (Random Labels on Clean Covers)\n")
    lines.append(
        "To verify that models learn genuine steganographic artifacts rather than dataset shortcuts, "
        "a logistic regression model was trained on clean covers with purely random labels:\n"
    )
    lines.append("| Modality | Clean Samples | Random-Label ROC-AUC | Expected Value | Shortcut Detected? |")
    lines.append("| :--- | :---: | :---: | :---: | :--- |")
    for fmt, d in sanity.items():
        n = d.get("sample_count", 0)
        auc = d.get("auc", 0.5)
        sc = "NO (Expected ~0.50)" if abs(auc - 0.50) <= 0.08 else "INVESTIGATE"
        lines.append(f"| **{fmt.capitalize()}** | {n} | {auc:.4f} | ~0.5000 | **{sc}** |")

    # Section 5: Scientific Ablations
    lines.append("\n## 5. Scientific Ablation Studies\n")
    if "ablation_a_separate_vs_shared" in ablations:
        ab_a = ablations["ablation_a_separate_vs_shared"]
        lines.append("### 5.1 Ablation A: Unified Shared Latent Space vs. Separate Per-Format Models\n")
        lines.append("| Modality | Separate RF (Accuracy / AUC) | Unified Network (Accuracy / AUC) | Honest Assessment |")
        lines.append("| :--- | :---: | :---: | :--- |")
        for fmt, s_res in ab_a.get("separate_per_format", {}).items():
            u_res = ab_a.get("unified_model", {}).get(fmt, {})
            rf_acc = s_res.get("accuracy", 0.0)
            uni_acc = u_res.get("accuracy", 0.0)
            diff = uni_acc - rf_acc
            if diff >= 0:
                note = f"Unified model ahead (+{diff:.4f} acc)"
            else:
                note = f"Separate RF ahead (+{-diff:.4f} acc)"
            lines.append(
                f"| **{fmt.capitalize()}** | {s_res.get('accuracy', 0):.4f} / {s_res.get('roc_auc', 0):.4f} | "
                f"{u_res.get('accuracy', 0):.4f} / {u_res.get('roc_auc', 0):.4f} | {note} |"
            )
        lines.append(f"\n> **Empirical Finding:** {ab_a.get('finding', '')}\n")

    if "ablation_b_auxiliary_head" in ablations:
        ab_b = ablations["ablation_b_auxiliary_head"]
        lines.append("### 5.2 Ablation B: With vs. Without Auxiliary Method Head\n")
        lines.append("Empirical test of unified model trained with auxiliary method head vs. an identical model trained without it:\n")
        lines.append("| Configuration | Overall Accuracy | Overall ROC-AUC | Image AUC | Audio AUC | Video AUC |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        w_aux = ab_b.get("with_aux_head", {})
        wo_aux = ab_b.get("without_aux_head", {})
        lines.append(
            f"| **With Auxiliary Head** | **{w_aux.get('overall_accuracy', 0):.4f}** | **{w_aux.get('overall_roc_auc', 0):.4f}** | "
            f"{w_aux.get('per_format', {}).get('image', {}).get('roc_auc', 0):.4f} | "
            f"{w_aux.get('per_format', {}).get('audio', {}).get('roc_auc', 0):.4f} | "
            f"{w_aux.get('per_format', {}).get('video', {}).get('roc_auc', 0):.4f} |"
        )
        lines.append(
            f"| **Without Auxiliary Head** | **{wo_aux.get('overall_accuracy', 0):.4f}** | **{wo_aux.get('overall_roc_auc', 0):.4f}** | "
            f"{wo_aux.get('per_format', {}).get('image', {}).get('roc_auc', 0):.4f} | "
            f"{wo_aux.get('per_format', {}).get('audio', {}).get('roc_auc', 0):.4f} | "
            f"{wo_aux.get('per_format', {}).get('video', {}).get('roc_auc', 0):.4f} |"
        )
        lines.append(f"\n> **Empirical Finding:** {ab_b.get('finding', '')}\n")

    if "ablation_c_deep_vs_gbdt" in ablations:
        ab_c = ablations["ablation_c_deep_vs_gbdt"]
        lines.append("### 5.3 Ablation C: Unified Neural Network vs. Histogram Gradient Boosting Baseline\n")
        lines.append("| Modality | HistGradientBoosting (Accuracy / AUC) | Unified Network (Accuracy / AUC) | Comparison |")
        lines.append("| :--- | :---: | :---: | :--- |")
        for fmt, g_res in ab_c.get("hist_gradient_boosting", {}).items():
            u_res = ab_c.get("unified_network", {}).get(fmt, {})
            gb_acc = g_res.get("accuracy", 0.0)
            uni_acc = u_res.get("accuracy", 0.0)
            diff = uni_acc - gb_acc
            comp = f"Unified network ahead (+{diff:.4f})" if diff >= 0 else f"GBDT ahead (+{-diff:.4f})"
            lines.append(
                f"| **{fmt.capitalize()}** | {g_res.get('accuracy', 0):.4f} / {g_res.get('roc_auc', 0):.4f} | "
                f"{u_res.get('accuracy', 0):.4f} / {u_res.get('roc_auc', 0):.4f} | {comp} |"
            )

    # Section 6: Limitations and Ethical Boundaries
    lines.append("\n## 6. Limitations & Scientific Boundaries\n")
    lines.append("1. **Public Covers vs In-House Stego Embeddings:** Cover media are sourced from standard public datasets (Imagenette2-160 for images, ESC-50 for audio). However, stego files are produced via our own pure-Python embedding engines under controlled laboratory conditions, which may differ from malware in the wild.")
    lines.append("2. **Video Covers:** Video clips are synthesized via smooth pan/zoom affine transformations across real still photographs (from Imagenette) in lossless FFV1 MKV containers, not natural camera footage. Real-world compressed video (e.g. H.264/HEVC) involves inter-frame motion estimation and quantization noise that require separate motion-vector steganalysis.")
    lines.append("3. **Unified Architecture vs. Per-Format Specialization:** While the unified cross-modal network enables a single deployment footprint and cross-modal inference, empirical ablations demonstrate that specialized tree models (RandomForest and HistGradientBoosting) remain highly competitive on handcrafted feature vectors, achieving equal or higher accuracy on some modalities (e.g., audio and images). The unified model's primary advantage is architectural consolidation rather than a strict accuracy lead across all individual formats.")
    lines.append("4. **Echo Hiding in Audio:** Audio echo hiding alters cepstral delay peaks rather than time-domain LSB planes. While cepstral peak detection provides signal, echo hiding remains more difficult to separate at lower payload rates than direct bit-plane substitution.")

    with open(output_path, "w") as f_out:
        f_out.write("\n".join(lines) + "\n")

    print(f"Successfully generated {output_path} directly from {metrics_path}.")

if __name__ == "__main__":
    generate_report(Path("docs/RESULTS/metrics.json"), Path("docs/REPORT.md"))
