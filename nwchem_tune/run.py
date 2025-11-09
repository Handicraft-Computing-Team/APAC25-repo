#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot cumulative-best curve from an Optuna SQLite study.
- Shows each trial's value (scatter)
- Draws the running minimum ("best-so-far") line
- Saves PNG and CSV next to the DB file

Usage:
  python3 plot_optuna_cummin.py --db /path/to/manual_tune.db
  # optionally specify a study:
  python3 plot_optuna_cummin.py --db /path/to/manual_tune.db --study manual_nwchem_visual
"""

import argparse
import math
from pathlib import Path

import optuna
import pandas as pd
import matplotlib.pyplot as plt


def load_trials(db_path: Path, study_name: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load completed trials (finite values) from the given Optuna SQLite DB."""
    storage = f"sqlite:///{db_path}"

    # If study_name not given, pick the most recently started study
    if study_name is None:
        summaries = optuna.study.get_all_study_summaries(storage)
        if not summaries:
            raise RuntimeError("No studies found in this DB.")
        study_name = sorted(summaries, key=lambda s: s.datetime_start)[-1].study_name

    study = optuna.load_study(study_name=study_name, storage=storage)

    trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and math.isfinite(t.value)
    ]
    if not trials:
        raise RuntimeError("No completed trials with finite values in this study.")

    trials.sort(key=lambda t: t.number)
    df = pd.DataFrame([{"trial": t.number, "value": t.value, **t.params} for t in trials])
    return df, study_name


def add_cummin(df: pd.DataFrame) -> pd.DataFrame:
    """Add a cumulative minimum (best-so-far) column to the DataFrame."""
    running_min = []
    cur = float("inf")
    for v in df["value"]:
        cur = v if v < cur else cur
        running_min.append(cur)
    df = df.copy()
    df["cummin"] = running_min
    return df


def plot_cummin(df: pd.DataFrame, out_png: Path, title: str) -> None:
    """Make a single-figure plot: scatter of values + line of best-so-far (orange)."""
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=16, fontweight="bold")
    # 大小
    plt.scatter(df["trial"], df["value"], label="Testing Value", color="blue", s=50)          # 蓝色散点
    plt.plot(df["trial"], df["cummin"], color="orange", linewidth=2,    # 橙色连线
             label="SOTA value (cumulative min)")                                      
    plt.xlabel("Testing number", fontsize=15)

    # x 轴是整数，每4个出现一次
    plt.xticks(df["trial"][::4])


    # 找出全局最低点
    best_row = df.loc[df["value"].idxmin()]
    best_trial = int(best_row["trial"])
    best_val = float(best_row["value"])

    # 标注最低值
    plt.scatter(best_trial, best_val, color="red", s=60, marker="*", label="best trial")
    plt.text(
        best_trial, best_val,
        f" trial {best_trial}: {best_val:.3f}",
        fontsize=10, color="red", ha="left", va="bottom", weight="bold"
    )

    plt.ylabel("Workload Running Time (smaller is better)", fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()



def main():
    ap = argparse.ArgumentParser(description="Plot Optuna cumulative best curve from SQLite DB.")
    ap.add_argument("--db", required=True, help="Path to Optuna SQLite DB (e.g., manual_tune.db)")
    ap.add_argument("--study", default=None, help="Study name; if omitted, pick the latest in DB")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    df, study_name = load_trials(db_path, args.study)
    df = add_cummin(df)

    out_dir = db_path.parent
    out_csv = out_dir / f"{study_name}_results.csv"
    out_png = out_dir / f"{study_name}_cummin.png"

    df.to_csv(out_csv, index=False)
    plot_cummin(df, out_png, f"Bayesian Optimization for NWChem Memory Allocation")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved PNG: {out_png}")


if __name__ == "__main__":
    main()
