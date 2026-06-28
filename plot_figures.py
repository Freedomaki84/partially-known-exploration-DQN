# make_figures_4_2.py
# Generates all Section 4.2 figures from your analysis_outputs CSVs.

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("analysis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

agg_csv = OUT_DIR / "aggregated_by_algo_env_buffer.csv"
raw_csv = OUT_DIR / "all_runs_raw.csv"

def robust_read_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[warn] failed to read {p}: {e}")
        return None

df_agg = robust_read_csv(agg_csv)
df_all = robust_read_csv(raw_csv)

def rebuild_from_all(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return None

    # Success proxy (use success if present, else reward)
    if "final_success" in df_all.columns:
        succ = pd.to_numeric(df_all["final_success"], errors="coerce")
    else:
        succ = pd.Series(np.nan, index=df_all.index)
    if "final_reward" in df_all.columns:
        succ = succ.fillna(pd.to_numeric(df_all["final_reward"], errors="coerce"))
    df_all["succ_proxy"] = succ

    # Make sure buffer exists (NaN OK if missing)
    if "buffer" not in df_all.columns:
        df_all["buffer"] = np.nan

    group_cols = ["algo", "env", "buffer"]

    # Possible columns present in all_runs_raw.csv
    metrics = {
        "succ_proxy": ["mean", "std"],
        "final_reward": ["mean", "std"],
        "exploration/unique_obs_mean_last100": ["mean", "std"],
        "exploration/frontier_rate_mean_last100": ["mean", "std"],
        "peak_replay_buffer_mb": ["mean", "std"],
        "peak_ram_mb": ["mean", "std"],
        "return_per_ram_gb": ["mean", "std"],
    }
    for col in list(metrics.keys()):
        if col not in df_all.columns:
            metrics.pop(col, None)

    g = df_all.groupby(group_cols, dropna=False).agg(metrics)
    g.columns = [f"{a}_{b}" for a,b in g.columns]
    g = g.reset_index()

    # Rename to expected names
    g = g.rename(columns={
        "succ_proxy_mean": "final_success_mean",
        "succ_proxy_std": "final_success_std",
        "final_reward_mean": "final_reward_mean",
        "final_reward_std": "final_reward_std",
        "exploration/unique_obs_mean_last100_mean": "exploration/unique_obs_mean_last100_mean",
        "exploration/unique_obs_mean_last100_std": "exploration/unique_obs_mean_last100_std",
        "exploration/frontier_rate_mean_last100_mean": "exploration/frontier_rate_mean_last100_mean",
        "exploration/frontier_rate_mean_last100_std": "exploration/frontier_rate_mean_last100_std",
        "peak_replay_buffer_mb_mean": "peak_replay_buffer_mb_mean",
        "peak_replay_buffer_mb_std": "peak_replay_buffer_mb_std",
        "peak_ram_mb_mean": "peak_ram_mb_mean",
        "peak_ram_mb_std": "peak_ram_mb_std",
        "return_per_ram_gb_mean": "return_per_ram_gb_mean",
        "return_per_ram_gb_std": "return_per_ram_gb_std",
    })
    return g

if df_agg is None or df_agg.empty:
    print("[info] Rebuilding aggregated_by_algo_env_buffer from all_runs_raw.csv…")
    df_agg = rebuild_from_all(df_all)

if df_agg is None or df_agg.empty:
    raise SystemExit("No usable data found. Make sure analysis_outputs/*.csv exist.")

def fmt_buf(v):
    try:
        return f"buf{int(v)//1000}k"
    except Exception:
        return "buf?"

df_agg["buffer_label"] = df_agg["buffer"].apply(fmt_buf)
df_agg["group"] = df_agg["algo"].astype(str) + " " + df_agg["buffer_label"].astype(str)

def save_bar(x, y, title, ylabel, fname):
    plt.figure(figsize=(10,6))
    plt.bar(x, y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ Saved:", OUT_DIR / fname)

# 1) Unique observations
col_unique = "exploration/unique_obs_mean_last100_mean"
if col_unique in df_agg.columns:
    save_bar(df_agg["group"], df_agg[col_unique],
             "Mean Unique Observations (last 100 episodes)",
             "unique_obs (mean)", "bar_unique_obs.png")
else:
    print("[warn] Missing:", col_unique)

# 2) Frontier rate
col_frontier = "exploration/frontier_rate_mean_last100_mean"
if col_frontier in df_agg.columns:
    save_bar(df_agg["group"], df_agg[col_frontier],
             "Mean Frontier Rate (last 100 episodes)",
             "frontier_rate (mean)", "bar_frontier_rate.png")
else:
    print("[warn] Missing:", col_frontier)

# 3) Final success (fallback to reward mean)
target_col = "final_success_mean" if "final_success_mean" in df_agg.columns else None
if target_col is None and "final_reward_mean" in df_agg.columns:
    target_col = "final_reward_mean"
if target_col:
    ttl = "Final Success (mean)" if target_col == "final_success_mean" else "Final Reward (mean)"
    save_bar(df_agg["group"], df_agg[target_col], ttl, target_col, "bar_final_success.png")
else:
    print("[warn] Missing final success/reward mean")

# 4) Peak replay-buffer MB
col_replay = "peak_replay_buffer_mb_mean"
if col_replay in df_agg.columns:
    save_bar(df_agg["group"], df_agg[col_replay],
             "Peak Replay Buffer (MB, mean)",
             "MB", "bar_peak_replay_mb.png")
else:
    print("[warn] Missing:", col_replay)

# 5) Scatter: memory vs performance
xcol = "peak_ram_mb_mean" if "peak_ram_mb_mean" in df_agg.columns else None
ycol = "final_success_mean" if "final_success_mean" in df_agg.columns else None
if xcol is None and "peak_replay_buffer_mb_mean" in df_agg.columns:
    xcol = "peak_replay_buffer_mb_mean"
if ycol is None and "final_reward_mean" in df_agg.columns:
    ycol = "final_reward_mean"

if xcol and ycol:
    plt.figure(figsize=(8,6))
    for _, r in df_agg.iterrows():
        x = r.get(xcol, np.nan)
        y = r.get(ycol, np.nan)
        if pd.notna(x) and pd.notna(y):
            plt.scatter(x, y)
            label = f'{r.get("algo","?")}-{fmt_buf(r.get("buffer", np.nan))}'
            plt.annotate(label, (x, y), xytext=(5,5), textcoords="offset points", fontsize=8)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title("Memory vs Performance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "scatter_memory_vs_success.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ Saved:", OUT_DIR / "scatter_memory_vs_success.png")
else:
    print("[warn] Could not plot scatter (missing x or y).")

print("✅ All done. Figures saved in:", OUT_DIR)
