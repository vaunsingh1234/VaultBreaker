#!/usr/bin/env python3
"""Inference CLI for VaultBreaker.
Scans individual files or entire directories, provides calibrated verdicts,
risk classifications, and detailed forensic explanations.
"""

import argparse
from pathlib import Path
import json
import sys

from vaultbreaker.utils.logger import get_logger
from vaultbreaker.inference.predict import VaultBreakerPredictor
from vaultbreaker.inference.router import RouterError

logger = get_logger("PredictCLI")

def print_single_report(res: dict) -> None:
    """Format and print an executive forensic analysis report."""
    print("=" * 64)
    print(f"  VAULTBREAKER FORENSIC ANALYSIS: {res['file_name']}")
    print("=" * 64)
    print(f"  Media Type        : {res['media_type'].upper()} ({res['format'].upper()})")
    
    verdict = res["verdict"]
    color_prefix = "\033[91m" if "STEGO" in verdict else "\033[92m"
    reset_color = "\033[0m"
    print(f"  Verdict           : {color_prefix}{verdict}{reset_color}")
    print(f"  Risk Level        : {res['risk_level']}")
    print(f"  Calibrated Prob   : {res['calibrated_probability']:.4f} (Raw: {res['raw_probability']:.4f})")
    print(f"  Decision Thresh   : {res['decision_threshold']:.4f} (Temp T = {res['temperature']:.3f})")
    print("-" * 64)
    print("  Top Contributory Indicators:")
    for name, val in res["top_features"]:
        print(f"    - {name:<30} : {val:+.4f} σ")
    print("=" * 64)

def scan_path(target_path: Path, predictor: VaultBreakerPredictor, with_explain: bool = False) -> list:
    """Scan file or directory recursively."""
    results = []
    if target_path.is_file():
        files = [target_path]
    elif target_path.is_dir():
        # Discover files
        files = [f for f in target_path.rglob("*") if f.is_file()]
    else:
        logger.error(f"Path does not exist: {target_path}")
        return []

    for f in files:
        try:
            res = predictor.predict_file(f, generate_explanation=with_explain)
            # Remove raw array objects before JSON serialization
            if "explanation" in res:
                exp_clean = {}
                for k, v in res["explanation"].items():
                    if isinstance(v, (int, float, str, list)):
                        exp_clean[k] = v
                res["explanation"] = exp_clean
            results.append(res)
            print_single_report(res)
        except RouterError as e:
            logger.warning(f"Skipping {f.name}: {e}")
        except Exception as e:
            logger.error(f"Error scanning {f.name}: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(description="VaultBreaker Steganography Detection CLI")
    parser.add_argument("target", type=str, help="Path to media file or directory to scan")
    parser.add_argument("--json", type=str, default=None, help="Save scan results to JSON file")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--explain", action="store_true", help="Include explanation artifacts")
    args = parser.parse_args()

    predictor = VaultBreakerPredictor(config_path=args.config)
    target = Path(args.target)
    results = scan_path(target, predictor, with_explain=args.explain)

    if args.json and results:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved {len(results)} scan results to {out_path}")

if __name__ == "__main__":
    main()
