"""Structured reporting for pipeline runs (console + JSON)."""
from __future__ import annotations
import json
import os


def print_report(ctx):
    print("\n" + "=" * 70)
    print("AGENT EXECUTION TRACE")
    print("=" * 70)
    total = 0.0
    for step in ctx.trace:
        total += step.get("seconds", 0)
        print(f"[{step['status']:>6}] {step['agent']:<32} "
              f"{step.get('seconds', 0):>8.4f}s  {step.get('detail', '')}")
    print(f"{'':>9}{'TOTAL':<32} {total:>8.4f}s")

    print("\n" + "=" * 70)
    print("PIPELINE DECISIONS")
    print("=" * 70)
    print(f"data_kind          : {ctx.data_kind}")
    print(f"file_types         : {ctx.file_types}")
    print(f"anonymized columns : {list(ctx.anonymization_report.get('masked_columns', {}))}")
    print(f"PII types detected : {ctx.anonymization_report.get('detected_types')}")
    print(f"selected model     : {ctx.selected_model}")
    print(f"target             : {ctx.target}  (classes={ctx.n_classes})")
    print(f"valid data headers : {ctx.valid_headers}")

    if ctx.match_report.get("per_model"):
        print("\n" + "=" * 70)
        print("MODEL-DATA MATCHER (per model)")
        print("=" * 70)
        for m, info in ctx.match_report["per_model"].items():
            flag = "ELIGIBLE" if info["eligible"] else "rejected"
            print(f"  {m}: {info['n_matched']}/{info['n_required']} headers matched  [{flag}]")

    if ctx.column_types:
        print("\n" + "=" * 70)
        print("COLUMN TYPES (rule-based)")
        print("=" * 70)
        for c, t in ctx.column_types.items():
            print(f"  {c:<28} {t}")

    if ctx.metrics:
        print("\n" + "=" * 70)
        print("INFERENCE METRICS  (PerpetualBooster, held-out test set)")
        print("=" * 70)
        for k, v in ctx.metrics.items():
            print(f"  {k:<20}: {v}")

    if ctx.explanations.get("top10"):
        print("\n" + "=" * 70)
        print("EXPLAINABILITY  (SHAP top-10 features)")
        print("=" * 70)
        for i, (feat, val) in enumerate(ctx.explanations["top10"], 1):
            print(f"  {i:>2}. {feat:<30} {val}")
    elif ctx.explanations:
        print(f"\nexplanations: {ctx.explanations}")

    if ctx.predictions_path:
        print(f"\npredictions written to: {ctx.predictions_path}")


def save_report(ctx, outdir):
    os.makedirs(outdir, exist_ok=True)
    report = {
        "data_kind": ctx.data_kind,
        "file_types": ctx.file_types,
        "selected_model": ctx.selected_model,
        "target": ctx.target,
        "n_classes": ctx.n_classes,
        "valid_headers": ctx.valid_headers,
        "column_types": ctx.column_types,
        "anonymization": ctx.anonymization_report,
        "match_report": ctx.match_report,
        "metrics": ctx.metrics,
        "explanations": ctx.explanations,
        "trace": ctx.trace,
    }
    path = os.path.join(outdir, "pipeline_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"full report written to: {path}")
    return path
