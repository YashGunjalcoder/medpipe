"""
medpipe command-line interface.

Usage:
    python -m medpipe run DATA_FILE [options]
    python -m medpipe info DATA_FILE
    python -m medpipe demo

Examples:
    python -m medpipe run data/hope.sav --target anxiety
    python -m medpipe run mydata.csv --target outcome --threshold 0.55 --no-explain
    python -m medpipe info data/hope.sav
    python -m medpipe demo
"""
from __future__ import annotations
import argparse
import os
import sys

from .orchestrator import Orchestrator
from .data_loader import load_any
from .report import print_report, save_report


def _cmd_info(args):
    df, _meta = load_any(args.data_file)
    print(f"file      : {args.data_file}")
    print(f"shape     : {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("columns:")
    for c in df.columns:
        print(f"  - {c:<28} dtype={str(df[c].dtype):<8} "
              f"nunique={df[c].nunique():<6} n_missing={int(df[c].isna().sum())}")
    return 0


def _cmd_run(args):
    df, _meta = load_any(args.data_file)
    if args.target and args.target not in df.columns:
        print(f"error: --target '{args.target}' not found. Available columns:",
              file=sys.stderr)
        print("  " + ", ".join(map(str, df.columns)), file=sys.stderr)
        return 2

    orch = Orchestrator(
        output_dir=args.output_dir,
        threshold=args.threshold,
        explain=not args.no_explain,
        test_size=args.test_size,
        random_state=args.seed,
    )
    ctx = orch.run(input_path=args.data_file, dataframe=df.copy(),
                   force_target=args.target, target_mode=args.target_mode)
    print_report(ctx)
    save_report(ctx, args.output_dir)
    return 0


def _cmd_demo(args):
    from .synthetic import generate
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "anxiety_synthetic.csv")
    generate(n=600, seed=42).to_csv(csv_path, index=False)
    df, _ = load_any(csv_path)
    orch = Orchestrator(output_dir=args.output_dir, explain=not args.no_explain)
    ctx = orch.run(input_path=csv_path, dataframe=df.copy())
    print_report(ctx)
    save_report(ctx, args.output_dir)
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="medpipe",
        description="Agentic AI pipeline for end-to-end medical data inference.")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run the full 7-agent pipeline on a data file")
    pr.add_argument("data_file", help=".sav / .csv / .tsv / .xlsx")
    pr.add_argument("--target", help="target column (else inferred from model DB)")
    pr.add_argument("--target-mode", choices=["auto", "binary", "multiclass"],
                    default="auto", help="target framing (default: auto)")
    pr.add_argument("--threshold", type=float, default=0.6,
                    help="header-match cosine threshold (default: 0.6)")
    pr.add_argument("--test-size", type=float, default=0.2)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--no-explain", action="store_true", help="skip SHAP explanations")
    pr.add_argument("--output-dir", default="run_output")
    pr.set_defaults(func=_cmd_run)

    pi = sub.add_parser("info", help="print schema/columns of a data file")
    pi.add_argument("data_file")
    pi.set_defaults(func=_cmd_info)

    pdm = sub.add_parser("demo", help="run on bundled synthetic data (no file needed)")
    pdm.add_argument("--output-dir", default="run_output")
    pdm.add_argument("--no-explain", action="store_true")
    pdm.set_defaults(func=_cmd_demo)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
