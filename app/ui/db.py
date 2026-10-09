"""SQLite Scan History and Incident Findings Store for VaultBreaker."""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = Path("data/scan_history.db")

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize scan history table schema."""
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
                method_guess TEXT,
                file_size_bytes INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
        """)
        conn.commit()

def generate_scan_id() -> str:
    """Generate sequential SCN-XXXX identifier."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM scans")
        count = cursor.fetchone()[0]
        return f"SCN-{count + 1:04d}"

def add_scan_record(report: Dict[str, Any], filepath: Optional[str] = None, status: Optional[str] = None) -> str:
    """Add a completed scan report to the database."""
    init_db()
    scan_id = generate_scan_id()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    verdict = report.get("verdict", "CLEAN")
    prob = float(report.get("calibrated_probability", 0.0))
    raw_prob = float(report.get("raw_probability", 0.0))
    risk = report.get("risk_level", "LOW RISK")

    if status is None:
        if prob >= 0.85:
            status = "Confirmed High Risk"
        elif "STEGO" in verdict:
            status = "Suspected"
        else:
            status = "Clean"

    method_guess = "clean"
    if "STEGO" in verdict:
        # Determine likely method from indicators
        top_feats = report.get("top_features", [])
        if top_feats:
            fname = top_feats[0][0].lower()
            if "lsb" in fname or "chi" in fname:
                method_guess = "lsb_matching"
            elif "dct" in fname:
                method_guess = "dct_ac"
            elif "srm" in fname or "spam" in fname:
                method_guess = "edge_adaptive"
            elif "temporal" in fname or "flicker" in fname:
                method_guess = "frame_lsb"
            else:
                method_guess = "lsb_replacement"

    # Clean explanation report for serializing
    clean_rep = report.copy()
    if "explanation" in clean_rep:
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
                status, method_guess, file_size_bytes, created_at, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan_id,
            report.get("file_name", "unknown_file"),
            str(filepath) if filepath else None,
            report.get("media_type", "image"),
            report.get("format", "bin"),
            verdict,
            risk,
            prob,
            raw_prob,
            status,
            method_guess,
            file_size,
            now_str,
            json.dumps(clean_rep)
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

def update_scan_status(scan_id: str, new_status: str):
    """Update status of a scan (e.g. Clean, Suspected, Confirmed High Risk, Triaged)."""
    with get_connection() as conn:
        conn.execute("UPDATE scans SET status = ? WHERE scan_id = ?", (new_status, scan_id))
        conn.commit()

def clear_all_scans():
    """Clear all records from scan history database."""
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM scans")
        conn.commit()

def scan_demo_sample(predictor, preferred_format: Optional[str] = None) -> Optional[str]:
    """Scan a real sample from demo_samples/ and record it in the database."""
    init_db()
    demo_root = Path("demo_samples")
    if not demo_root.exists():
        return None

    # Candidate real files across formats
    candidates = [
        demo_root / "image" / "img_src_0001_clean.png",
        demo_root / "image" / "img_src_0001_stego_lsb_matching_0.10.png",
        demo_root / "audio" / "aud_src_0006_clean.wav",
        demo_root / "audio" / "aud_src_0006_stego_lsb_replacement_0.70.wav",
        demo_root / "video" / "vid_src_0002_clean.mkv",
        demo_root / "video" / "vid_src_0002_stego_frame_lsb_0.20.mkv",
    ]

    # Check which have already been scanned
    scanned_filenames = {s["filename"] for s in get_all_scans(limit=500)}
    
    target_sample = None
    # First try unscanned candidates matching preferred format if specified
    if preferred_format:
        fmt_candidates = [c for c in candidates if preferred_format.lower() in str(c)]
        for c in fmt_candidates:
            if c.exists() and c.name not in scanned_filenames:
                target_sample = c
                break

    # Otherwise try any unscanned candidate
    if target_sample is None:
        for c in candidates:
            if c.exists() and c.name not in scanned_filenames:
                target_sample = c
                break

    # If all scanned, pick first existing candidate
    if target_sample is None:
        for c in candidates:
            if c.exists():
                target_sample = c
                break

    if target_sample is None or not target_sample.exists():
        return None

    rep = predictor.predict_file(target_sample, generate_explanation=True)
    return add_scan_record(rep, filepath=str(target_sample))

