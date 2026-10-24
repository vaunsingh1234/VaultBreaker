#!/usr/bin/env python3
"""Check UI presentation layer for unexplained, fake, or fallback numbers.

Fails (exit code 1) if:
- Any numeric literal is used as a fallback in .get(...) calls for displayed values
- Any deleted synthetic chart (create_radar_chart, create_radial_chart) is imported or used
- Any deleted heuristic field (method_guess) is used
- Any typed-in telemetry literals (such as hardcoded ROC-AUC "0.8990", "v1.0.0", "128-d") are found
"""

import ast
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ROOT_DIR / "app"

DELETED_CHART_NAMES = {"create_radar_chart", "create_radial_chart"}
FORBIDDEN_DISPLAY_STRINGS = [
    "0.8990", "0.9372", "0.6044", "0.9972", "0.8002",
    "26.5%", "45.6%", "v1.0.0", "128-d"
]
FORBIDDEN_EVALUATE_FALLBACKS = [
    "0.88", "0.80", "0.18", "0.85", "0.82", "0.89", "0.78", "0.75", "0.81"
]

def _rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return path.name

def check_ast_fallbacks(file_path: Path):
    errors = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        tree = ast.parse(content, filename=str(file_path))
    except Exception as e:
        return [f"AST parse error in {file_path.name}: {e}"]

    disp_name = _rel_path(file_path)

    for node in ast.walk(tree):
        # Check function/method calls: obj.get(key, fallback_literal)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                if len(node.args) >= 2:
                    fallback_arg = node.args[1]
                    # Check if fallback is a Constant number (int or float)
                    if isinstance(fallback_arg, ast.Constant) and isinstance(fallback_arg.value, (int, float)):
                        key_name = ""
                        if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                            key_name = node.args[0].value
                        errors.append(
                            f"{disp_name}: Line {node.lineno} uses numeric fallback literal {fallback_arg.value} in .get('{key_name}', ...)"
                        )

        # Check imports for deleted charts
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in DELETED_CHART_NAMES:
                        errors.append(
                            f"{disp_name}: Line {node.lineno} imports deleted chart '{alias.name}'"
                        )
            else:
                for alias in node.names:
                    if alias.name in DELETED_CHART_NAMES:
                        errors.append(
                            f"{disp_name}: Line {node.lineno} imports deleted chart '{alias.name}'"
                        )

    return errors

def check_forbidden_patterns(file_path: Path):
    errors = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    disp_name = _rel_path(file_path)

    for idx, line in enumerate(lines, 1):
        for s in FORBIDDEN_DISPLAY_STRINGS:
            if s in line:
                errors.append(
                    f"{disp_name}: Line {idx} contains forbidden hardcoded display string '{s}'"
                )
        if "method_guess" in line:
            errors.append(
                f"{disp_name}: Line {idx} contains forbidden deleted field 'method_guess'"
            )
        for dc in DELETED_CHART_NAMES:
            if dc in line:
                errors.append(
                    f"{disp_name}: Line {idx} references deleted chart '{dc}'"
                )

    return errors

def check_evaluate_script(eval_path: Path):
    errors = []
    if not eval_path.exists():
        return errors
    disp_name = _rel_path(eval_path)
    with open(eval_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        tree = ast.parse(content, filename=str(eval_path))
    except Exception as e:
        return [f"AST parse error in {eval_path.name}: {e}"]

    # Inspect build_side_by_side_comparison
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_side_by_side_comparison":
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, (int, float)):
                    # Allow 0, 1, 4 (rounding digits), but no hardcoded metric numbers
                    if child.value not in (0, 1, 4, 0.0, 1.0):
                        errors.append(
                            f"{disp_name}: build_side_by_side_comparison contains hardcoded numeric constant {child.value}"
                        )
    return errors

def main():
    py_files = list(APP_DIR.rglob("*.py"))
    all_errors = []

    for pf in py_files:
        all_errors.extend(check_ast_fallbacks(pf))
        all_errors.extend(check_forbidden_patterns(pf))

    eval_script = ROOT_DIR / "scripts" / "evaluate.py"
    all_errors.extend(check_evaluate_script(eval_script))

    if all_errors:
        print(f"❌ UI Numbers Check Failed ({len(all_errors)} violations found):")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print(f"✅ UI Numbers Check Passed: Clean presentation layer across {len(py_files)} files.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
