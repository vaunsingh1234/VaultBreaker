"""SQLite Scan History and Ground Truth Matching Store for VaultBreaker."""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

DB_PATH = Path("data/scan_history.db")
_GT_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize scan history table schema and migrate missing columns."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                scan_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT,
                media_type TEXT NOT NULL,
                format TEXT NOT NULL,
                verdict TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                probability REAL NOT NULL,
                raw_probability REAL NOT NULL,
                status TEXT NOT NULL,
                file_size_bytes INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                report_json TEXT NOT NULL,
                ground_truth_label INTEGER,
                ground_truth_method TEXT,
                ground_truth_payload REAL,
                is_correct INTEGER
            )
        """)
        # Safe migration if table exists without new columns
        cursor = conn.execute("PRAGMA table_info(scans)")
        cols = {row["name"] for row in cursor.fetchall()}
        if "ground_truth_label" not in cols:
            conn.execute("ALTER TABLE scans ADD COLUMN ground_truth_label INTEGER")
        if "ground_truth_method" not in cols:
            conn.execute("ALTER TABLE scans ADD COLUMN ground_truth_method TEXT")
        if "ground_truth_payload" not in cols:
            conn.execute("ALTER TABLE scans ADD COLUMN ground_truth_payload REAL")
        if "is_correct" not in cols:
            conn.execute("ALTER TABLE scans ADD COLUMN is_correct INTEGER")
        conn.commit()


def get_ground_truth_lookup() -> Dict[str, Dict[str, Any]]:
    """Build or retrieve cached lookup table for ground-truth manifests."""
    global _GT_CACHE
    if _GT_CACHE is not None:
        return _GT_CACHE

    lookup: Dict[str, Dict[str, Any]] = {}
    
    # 1. Load demo_samples/ground_truth.csv
    demo_gt_path = Path("demo_samples/ground_truth.csv")
    if demo_gt_path.exists():
        try:
            df_demo = pd.read_csv(demo_gt_path)
            for _, r in df_demo.iterrows():
                raw_lbl = r.get("label")
                raw_rate = r.get("payload_rate")
                entry = {
                    "source_id": str(r.get("source_id") or ""),
                    "filename": str(r.get("filename") or ""),
                    "media_type": str(r.get("media_type") or ""),
                    "label": int(raw_lbl) if raw_lbl is not None else None,
                    "method": str(r.get("method") or "clean"),
                    "payload_rate": float(raw_rate) if raw_rate is not None else None,
                    "sha256": str(r.get("sha256") or ""),
                    "split": str(r.get("split") or "test"),
                    "demo_filepath": str(r.get("demo_filepath") or "")
                }
                fn = entry["filename"]
                sha = entry["sha256"]
                if fn:
                    lookup[fn] = entry
                if sha:
                    lookup[sha] = entry
        except Exception:
            pass

    # 2. Supplement with data/splits/manifest_all.csv if present
    all_manifest = Path("data/splits/manifest_all.csv")
    if all_manifest.exists():
        try:
            df_all = pd.read_csv(all_manifest)
            for _, r in df_all.iterrows():
                fn = str(r.get("filename") or "")
                sha = str(r.get("sha256") or "")
                if fn and fn not in lookup:
                    raw_lbl = r.get("label")
                    raw_rate = r.get("payload_rate")
                    entry = {
                        "source_id": str(r.get("source_id") or ""),
                        "filename": fn,
                        "media_type": str(r.get("media_type") or ""),
                        "label": int(raw_lbl) if raw_lbl is not None else None,
                        "method": str(r.get("method") or "clean"),
                        "payload_rate": float(raw_rate) if raw_rate is not None else None,
                        "sha256": sha,
                        "split": str(r.get("split") or "test"),
                        "filepath": str(r.get("filepath") or "")
                    }
                    lookup[fn] = entry
                    if sha:
                        lookup[sha] = entry
        except Exception:
            pass

    _GT_CACHE = lookup
    return lookup


def find_ground_truth(filename: Optional[str] = None, sha256: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Look up ground-truth metadata by filename or SHA-256."""
    lookup = get_ground_truth_lookup()
    if filename and filename in lookup:
        return lookup[filename]
    if sha256 and sha256 in lookup:
        return lookup[sha256]
    return None


def generate_scan_id() -> str:
    """Generate sequential SCN-XXXX identifier."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM scans")
        count = cursor.fetchone()[0]
        return f"SCN-{count + 1:04d}"


def add_scan_record(
    report: Dict[str, Any],
    filepath: Optional[str] = None,
    status: Optional[str] = None,
    threshold: Optional[float] = None
) -> str:
    """Add a completed scan report to the database with ground truth matching."""
    init_db()
    scan_id = generate_scan_id()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    verdict = report.get("verdict") or "CLEAN"
    prob_val = report.get("calibrated_probability")
    prob = float(prob_val) if prob_val is not None else 0.0
    raw_val = report.get("raw_probability")
    raw_prob = float(raw_val) if raw_val is not None else 0.0
    risk = report.get("risk_level") or "LOW RISK"

    if status is None:
        if "STEGO" in verdict:
            status = "Suspected"
        else:
            status = "Clean"

    file_name = report.get("file_name") or "unknown_file"
    file_sha = report.get("sha256")
    gt = find_ground_truth(filename=file_name, sha256=file_sha)

    gt_label = None
    gt_method = None
    gt_payload = None
    is_correct = None

    t = threshold if threshold is not None else 0.5
    pred_stego = (prob >= t)

    if gt is not None:
        gt_label = gt.get("label")
        gt_method = gt.get("method")
        gt_payload = gt.get("payload_rate")
        actual_stego = (gt_label == 1)
        is_correct = 1 if (pred_stego == actual_stego) else 0

    # Clean explanation report for serializing
    clean_rep = report.copy()
    if "explanation" in clean_rep and isinstance(clean_rep["explanation"], dict):
        exp_clean = {}
        for k, v in clean_rep["explanation"].items():
            if isinstance(v, (int, float, str, list, dict)):
                exp_clean[k] = v
        clean_rep["explanation"] = exp_clean

    file_size = 0
    if filepath and Path(filepath).exists():
        try:
            file_size = Path(filepath).stat().st_size
        except Exception:
            pass

    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO scans (
                scan_id, filename, filepath, media_type, format,
                verdict, risk_level, probability, raw_probability,
                status, file_size_bytes, created_at, report_json,
                ground_truth_label, ground_truth_method, ground_truth_payload, is_correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan_id,
            file_name,
            str(filepath) if filepath else None,
            report.get("media_type") or "image",
            report.get("format") or "bin",
            verdict,
            risk,
            prob,
            raw_prob,
            status,
            file_size,
            now_str,
            json.dumps(clean_rep),
            gt_label,
            gt_method,
            gt_payload,
            is_correct
        ))
        conn.commit()
    return scan_id


def get_all_scans(limit: int = 200) -> List[Dict[str, Any]]:
    """Retrieve all scans ordered by newest first."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM scans ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_scan_by_id(scan_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve single scan record by ID."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["report"] = json.loads(res["report_json"])
        except Exception:
            res["report"] = {}
        return res


def clear_all_scans():
    """Clear all records from scan history database."""
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM scans")
        conn.commit()


def scan_demo_sample(predictor, preferred_format: Optional[str] = None, threshold: Optional[float] = None) -> Optional[str]:
    """Scan a real sample from demo_samples/ and record it with ground truth."""
    init_db()
    demo_root = Path("demo_samples")
    if not demo_root.exists():
        return None

    candidates = [
        demo_root / "image" / "img_src_0001_clean.png",
        demo_root / "image" / "img_src_0001_stego_lsb_matching_0.10.png",
        demo_root / "audio" / "aud_src_0006_clean.wav",
        demo_root / "audio" / "aud_src_0006_stego_lsb_replacement_0.70.wav",
        demo_root / "video" / "vid_src_0002_clean.mkv",
        demo_root / "video" / "vid_src_0002_stego_frame_lsb_0.20.mkv",
    ]

    scanned_filenames = {s.get("filename") for s in get_all_scans(limit=500)}
    target_sample = None

    if preferred_format:
        fmt_candidates = [c for c in candidates if preferred_format.lower() in str(c)]
        for c in fmt_candidates:
            if c.exists() and c.name not in scanned_filenames:
                target_sample = c
                break

    if target_sample is None:
        for c in candidates:
            if c.exists() and c.name not in scanned_filenames:
                target_sample = c
                break

    if target_sample is None:
        for c in candidates:
            if c.exists():
                target_sample = c
                break

    if target_sample is None or not target_sample.exists():
        return None

    rep = predictor.predict_file(target_sample, generate_explanation=True)
    return add_scan_record(rep, filepath=str(target_sample), threshold=threshold)


def load_manifest_summary(manifest_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Compute summary statistics across dataset manifests."""
    m_dir = manifest_dir if manifest_dir is not None else Path("data/splits")
    m_path = m_dir / "manifest_all.csv"
    if not m_path.exists():
        return {
            "total_files": 0,
            "unique_sources": 0,
            "media_counts": {},
            "split_counts": {},
            "clean_count": 0,
            "stego_count": 0
        }
    df = pd.read_csv(m_path)
    total_files = len(df)
    unique_sources = int(df["source_id"].nunique()) if "source_id" in df.columns else 0
    media_counts = df["media_type"].value_counts().to_dict() if "media_type" in df.columns else {}
    split_counts = df["split"].value_counts().to_dict() if "split" in df.columns else {}
    clean_count = int((df["label"] == 0).sum()) if "label" in df.columns else 0
    stego_count = int((df["label"] == 1).sum()) if "label" in df.columns else 0

    return {
        "total_files": total_files,
        "unique_sources": unique_sources,
        "media_counts": media_counts,
        "split_counts": split_counts,
        "clean_count": clean_count,
        "stego_count": stego_count
    }


def load_explorer_dataset(
    manifest_path: Optional[Path] = None,
    demo_path: Optional[Path] = None,
    generated_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, bool]:
    """
    Load dataset for Data Explorer.
    If generated_dir (default data/generated) is missing, falls back to demo_samples.
    Returns (dataframe, is_fallback_to_demo).
    """
    gen_dir = generated_dir if generated_dir is not None else Path("data/generated")
    demo_file = demo_path if demo_path is not None else Path("demo_samples/ground_truth.csv")
    man_file = manifest_path if manifest_path is not None else Path("data/splits/manifest_all.csv")

    # If full generated dataset does not exist on disk, fall back to demo samples
    if not gen_dir.exists():
        if demo_file.exists():
            df_demo = pd.read_csv(demo_file)
            if "demo_filepath" in df_demo.columns:
                df_demo["resolved_filepath"] = df_demo["demo_filepath"]
            elif "filepath" in df_demo.columns:
                df_demo["resolved_filepath"] = df_demo["filepath"]
            return df_demo, True
        return pd.DataFrame(), True

    if man_file.exists():
        df_all = pd.read_csv(man_file)
        df_all["resolved_filepath"] = df_all.get("filepath")
        return df_all, False

    return pd.DataFrame(), False


def format_blind_mode(row: Dict[str, Any], is_blind: bool = True) -> Dict[str, Any]:
    """Format row fields depending on whether Blind Test Mode is active."""
    raw_label = row.get("label")
    raw_method = row.get("method")
    raw_payload = row.get("payload_rate")

    if is_blind:
        return {
            "label_str": "[Hidden in Blind Mode]",
            "method_str": "[Hidden in Blind Mode]",
            "payload_rate_str": "[Hidden in Blind Mode]",
            "is_blind": True
        }

    label_int = int(raw_label) if raw_label is not None else None
    label_str = "Clean" if label_int == 0 else "Hidden Data" if label_int == 1 else "Unknown"
    method_str = str(raw_method or "clean")
    payload_val = float(raw_payload) if raw_payload is not None else None
    payload_str = f"{payload_val:.2f}" if payload_val is not None else "None"

    return {
        "label_str": label_str,
        "method_str": method_str,
        "payload_rate_str": payload_str,
        "is_blind": False
    }

