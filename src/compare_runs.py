"""Comparison script for evaluation logs across baseline and distillation runs.

Lightweight, CPU-only analysis utility:
- Parses log.csv files from one or more root directories.
- Computes relative perplexity changes vs baseline.
- Computes percentage of teacher quality gap closed on C4.
- Generates publication-ready validation perplexity curves on a log y-axis.
- Outputs Markdown summary table.

NOTE: Uses only standard library + matplotlib (NO torch import).
"""
import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

# Prevent matplotlib cache permission warnings
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib.pyplot as plt


def parse_log_csv(file_path: Path):
    """
    Reads log.csv and extracts full curve history and last row data.
    Expected columns: step, train_loss, val_c4, val_ts, val_avg, ppl_c4, ppl_ts, lr
    """
    history = {"steps": [], "ppl_c4": [], "ppl_ts": []}
    last_row = None

    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(float(row["step"]))
                ppl_c4 = float(row["ppl_c4"])
                ppl_ts = float(row["ppl_ts"])
                history["steps"].append(step)
                history["ppl_c4"].append(ppl_c4)
                history["ppl_ts"].append(ppl_ts)
                last_row = {
                    "step": step,
                    "train_loss": float(row["train_loss"]),
                    "val_c4": float(row["val_c4"]),
                    "val_ts": float(row["val_ts"]),
                    "val_avg": float(row["val_avg"]),
                    "ppl_c4": ppl_c4,
                    "ppl_ts": ppl_ts,
                    "lr": float(row["lr"]),
                }
            except (ValueError, KeyError) as e:
                print(f"[WARN] Skipping malformed line in {file_path}: {e}")
                continue

    return history, last_row


def find_run_log(roots, run_name: str):
    """
    Searches for <root>/<run_name>/log.csv across roots in priority order.
    Returns the first matching path, or None if not found.
    """
    for root in roots:
        candidate = Path(root) / run_name / "log.csv"
        if candidate.is_file():
            return candidate
    return None


def plot_comparison_curves(runs_data: dict, out_file: Path):
    """
    Plots validation perplexity curves for C4 and TinyStories on a log scale.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Color palette
    colors = plt.cm.tab10.colors

    # Subplot 1: C4 Validation Perplexity
    ax0 = axes[0]
    for idx, (run_name, data) in enumerate(runs_data.items()):
        color = colors[idx % len(colors)]
        hist = data["history"]
        is_baseline = "baseline" in run_name.lower()
        ax0.plot(
            hist["steps"],
            hist["ppl_c4"],
            label=run_name,
            color=color,
            linewidth=2.2 if is_baseline else 1.6,
            linestyle="--" if is_baseline else "-",
            marker="o" if len(hist["steps"]) <= 15 else None,
            markersize=4,
        )
    ax0.set_title("C4 Validation Perplexity (Lower is Better)", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Steps", fontsize=11)
    ax0.set_ylabel("Perplexity (log scale)", fontsize=11)
    ax0.set_yscale("log")
    ax0.grid(True, which="both", linestyle=":", alpha=0.6)
    ax0.legend(frameon=True, fontsize=9)

    # Subplot 2: TinyStories Validation Perplexity
    ax1 = axes[1]
    for idx, (run_name, data) in enumerate(runs_data.items()):
        color = colors[idx % len(colors)]
        hist = data["history"]
        is_baseline = "baseline" in run_name.lower()
        ax1.plot(
            hist["steps"],
            hist["ppl_ts"],
            label=run_name,
            color=color,
            linewidth=2.2 if is_baseline else 1.6,
            linestyle="--" if is_baseline else "-",
            marker="o" if len(hist["steps"]) <= 15 else None,
            markersize=4,
        )
    ax1.set_title("TinyStories Validation Perplexity (Lower is Better)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Steps", fontsize=11)
    ax1.set_ylabel("Perplexity (log scale)", fontsize=11)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"[OK] Saved comparison curves to: {out_file}")


def generate_markdown_table(runs_data: dict, baseline_name: str, teacher_c4: float = None):
    """
    Constructs a markdown table summarizing final results, relative changes,
    and gap closed on C4. Warns on step mismatch.
    """
    has_teacher_c4 = teacher_c4 is not None
    base_data = runs_data.get(baseline_name, None)

    base_last_step = None
    base_c4_loss = None
    base_c4_ppl = None
    base_ts_loss = None
    base_ts_ppl = None

    if base_data and base_data.get("last_row"):
        b_row = base_data["last_row"]
        base_last_step = b_row["step"]
        base_c4_loss = b_row["val_c4"]
        base_c4_ppl = b_row["ppl_c4"]
        base_ts_loss = b_row["val_ts"]
        base_ts_ppl = b_row["ppl_ts"]

    # Table headers
    headers = [
        "Run",
        "Last Step",
        "C4 Loss",
        "C4 PPL",
        "TS Loss",
        "TS PPL",
        "C4 PPL Change (%)",
        "TS PPL Change (%)",
    ]
    if has_teacher_c4:
        headers.append("C4 Gap Closed (%)")

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    warnings = []

    for run_name, data in runs_data.items():
        row = data.get("last_row")
        if not row:
            continue

        step = row["step"]
        c4_loss = row["val_c4"]
        c4_ppl = row["ppl_c4"]
        ts_loss = row["val_ts"]
        ts_ppl = row["ppl_ts"]

        # Check step alignment with baseline
        step_str = str(step)
        if base_last_step is not None and step != base_last_step:
            step_str += " *"
            warnings.append(
                f"[WARN] Run '{run_name}' ended at step {step}, whereas baseline '{baseline_name}' ended at {base_last_step}. Comparison may not be matched compute!"
            )

        # Relative PPL changes
        if base_c4_ppl is not None:
            if run_name == baseline_name:
                c4_change_str = "baseline"
                ts_change_str = "baseline"
            else:
                c4_chg = ((c4_ppl - base_c4_ppl) / base_c4_ppl) * 100.0
                ts_chg = ((ts_ppl - base_ts_ppl) / base_ts_ppl) * 100.0
                c4_change_str = f"{c4_chg:+.2f}%"
                ts_change_str = f"{ts_chg:+.2f}%"
        else:
            c4_change_str = "N/A"
            ts_change_str = "N/A"

        # C4 Gap closed: (base_c4 - run_c4) / (base_c4 - teacher_c4)
        gap_str = "N/A"
        if has_teacher_c4:
            if base_c4_loss is not None:
                denom = base_c4_loss - teacher_c4
                if abs(denom) > 1e-6:
                    if run_name == baseline_name:
                        gap_str = "0.00%"
                    else:
                        gap = ((base_c4_loss - c4_loss) / denom) * 100.0
                        gap_str = f"{gap:+.2f}%"
                else:
                    gap_str = "N/A"
            else:
                gap_str = "N/A"

        cols = [
            f"`{run_name}`",
            step_str,
            f"{c4_loss:.4f}",
            f"{c4_ppl:.2f}",
            f"{ts_loss:.4f}",
            f"{ts_ppl:.2f}",
            c4_change_str,
            ts_change_str,
        ]
        if has_teacher_c4:
            cols.append(gap_str)

        lines.append("| " + " | ".join(cols) + " |")

    table_md = "\n".join(lines) + "\n"
    return table_md, warnings


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation logs and generate comparison figures/tables.")
    parser.add_argument(
        "--root",
        nargs="+",
        action="extend",
        required=True,
        help="One or more root directories containing run checkpoints folders (e.g. checkpoints/ or drive/checkpoints/)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of run names to compare (e.g. baseline_short kd_short_T1 kd_short_T2)",
    )
    parser.add_argument(
        "--baseline",
        default="baseline_short",
        help="Name of baseline run to compare against (default: baseline_short)",
    )
    parser.add_argument(
        "--teacher_c4",
        type=float,
        default=None,
        help="Teacher C4 validation loss (e.g. 3.047) to compute retention gap closed",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Directory to save comparison plot and table (default: results)",
    )

    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flatten roots list
    roots = [Path(r) for r in args.root]

    # Collect data for each requested run
    runs_data = {}
    missing_runs = []

    for run in args.runs:
        log_path = find_run_log(roots, run)
        if log_path is None:
            print(f"[WARN] Run '{run}' not found in any of the provided roots: {[str(r) for r in roots]}. Skipping.")
            missing_runs.append(run)
            continue

        history, last_row = parse_log_csv(log_path)
        if not history["steps"] or last_row is None:
            print(f"[WARN] Run '{run}' log file at '{log_path}' contains no valid rows. Skipping.")
            missing_runs.append(run)
            continue

        runs_data[run] = {
            "path": log_path,
            "history": history,
            "last_row": last_row,
        }

    if not runs_data:
        print("[ERROR] No valid runs found to compare. Exiting.")
        sys.exit(1)

    # Generate Markdown Table
    table_md, warnings = generate_markdown_table(runs_data, args.baseline, args.teacher_c4)

    # Print Warnings if any
    for w in warnings:
        print(w)

    # Print table to stdout
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON TABLE")
    print("=" * 80)
    print(table_md)

    # Save Markdown Table
    table_file = out_dir / "results_table.md"
    table_file.write_text(table_md, encoding="utf-8")
    print(f"[OK] Saved table markdown to: {table_file}")

    # Generate Plot
    plot_file = out_dir / "val_ppl_compare.png"
    plot_comparison_curves(runs_data, plot_file)


if __name__ == "__main__":
    main()

