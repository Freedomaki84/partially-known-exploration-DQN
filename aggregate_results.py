#!/usr/bin/env python3
"""
Aggregate and visualise results from multiple DQN experiments.

What this script does:
- Scans results/<ALGO>/<ENV>/<RUN_TAG>/ for:
    * logs/progress.csv  (or progress.csv at run root)
    * evaluation/evaluations.npz (SB3 EvalCallback)
    * memory_summary.csv (custom PeakMemoryCallback)
- Builds per-run records with:
    * best evaluation mean reward & success ratio (from evaluations.npz)
    * final CSV fallbacks when eval file is missing
    * memory footprint (peak RAM, replay buffer, model params, optimiser)
    * inferred buffer size and seed from run tag
- Aggregates by (algo, env, buffer) -> mean & std
- Saves:
    * analysis_outputs/all_runs_raw.csv
    * analysis_outputs/aggregated_by_algo_env_buffer.csv
    * analysis_outputs/bar_final_success.png
    * analysis_outputs/bar_peak_replay_mb.png
    * analysis_outputs/scatter_memory_vs_success.png
    * analysis_outputs/README_ANALYSIS.txt
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("results")                 # expects results/<ALGO>/<ENV>/<RUN_TAG>/
OUT  = Path("analysis_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------- Helpers -----------------------------

def parse_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """
    Extract algo/env/tag from path and try to parse seed + buffer from tag.
    Accepts tags such as: seed_222, seed222, s222, buf25k, buffer_100k, bf25K, bf50K.
    """
    meta: Dict[str, Any] = {
        "algo": None, "env": None, "tag": None, "seed": None, "buffer": None
    }
    parts = run_dir.parts
    try:
        i = parts.index("results")
        meta["algo"] = parts[i+1] if len(parts) > i+1 else None
        meta["env"]  = parts[i+2] if len(parts) > i+2 else None
        meta["tag"]  = parts[i+3] if len(parts) > i+3 else None
    except ValueError:
        pass

    tag = meta["tag"] or ""

    # seed: seed_000, seed000, s000
    m = re.search(r"(?:seed_?|s)(\d+)", tag, flags=re.IGNORECASE)
    if m:
        try:
            meta["seed"] = int(m.group(1))
        except Exception:
            meta["seed"] = None

    # buffer: buf25k, buffer_100k, bf25K, bf50K, "25k" somewhere
    m2 = re.search(r"(?:buf(?:fer)?|bf)_?(\d+)[kK]", tag)
    if not m2:
        m2 = re.search(r"\b(\d+)[kK]\b", tag)
    if m2:
        try:
            meta["buffer"] = int(m2.group(1)) * 1000
        except Exception:
            meta["buffer"] = None

    return meta


def robust_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Failed to read {path}: {e}")
        return None


def last_non_nan(series: Optional[pd.Series]) -> Optional[float]:
    if series is None:
        return None
    s = series.dropna()
    if len(s) == 0:
        return None
    return float(s.iloc[-1])


def read_best_eval(eval_dir: Path,
                   door_key_success_proxy: bool = True
                  ) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Read SB3 EvalCallback outputs from evaluation/evaluations.npz.
    Returns (best_timestep, best_mean_reward, best_success_ratio).

    Success ratio:
      - If door_key_success_proxy=True, success = (reward > 0) over the eval episodes
        (works for MiniGrid-DoorKey).
      - If False, returns None for success unless later overwritten by CSV success columns.
    """
    npz = eval_dir / "evaluations.npz"
    if not npz.exists():
        return None, None, None
    try:
        data = np.load(npz, allow_pickle=True)
        results = data["results"]     # shape: (n_evals, n_episodes)
        timesteps = data["timesteps"] # shape: (n_evals,)
        means = results.mean(axis=1)
        i = int(np.argmax(means))
        best_mean = float(means[i])
        best_ts   = int(timesteps[i])

        if door_key_success_proxy:
            best_success = float((results[i] > 0).mean())
        else:
            best_success = None

        return best_ts, best_mean, best_success
    except Exception as e:
        print(f"[warn] Failed to read {npz}: {e}")
        return None, None, None


def summarize_progress(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract helpful metrics from progress.csv (works across your naming variants).
    """
    summary: Dict[str, Any] = {}
    # steps / episodes
    summary["final_step"] = df["step"].iloc[-1] if "step" in df.columns and len(df) else np.nan
    summary["episodes"]   = df["episodes"].iloc[-1] if "episodes" in df.columns and len(df) else np.nan

    # Basic training stats (last logged)
    summary["final_train_ep_reward"] = last_non_nan(df.get("train/ep_reward"))
    summary["final_train_ep_len"]    = last_non_nan(df.get("train/ep_len"))
    # Some SB3 logs use "train/ep_len_mean"
    summary["final_ep_len_alt"]      = last_non_nan(df.get("train/ep_len_mean"))

    # Exploration (mean of last 100 if present)
    for col in ["exploration/unique_obs", "exploration/frontier_rate"]:
        if col in df.columns and len(df) > 0:
            summary[f"{col}_mean_last100"] = float(df[col].tail(min(100, len(df))).mean())
        else:
            summary[f"{col}_mean_last100"] = np.nan

    # Evaluation CSV columns (older / custom)
    # We still read them as *fallbacks* (best eval comes from evaluations.npz)
    for col, out in [
        ("eval_q/success", "eval_success_csv"),
        ("eval_q/mean_reward", "eval_mean_csv"),
        ("eval_masked/success", "eval_success_masked_csv"),
        ("eval_masked/mean_reward", "eval_mean_masked_csv"),
        ("eval/success", "eval_success_csv_old"),
        ("eval/mean_reward", "eval_mean_csv_old"),
        ("evaluation/mean_reward", "eval_mean_csv_alt"),
    ]:
        summary[out] = last_non_nan(df.get(col))

    return summary


def read_memory_summary(path: Path) -> Dict[str, Any]:
    """
    Read memory_summary.csv from SB3 runs (or an empty dict of NaNs if missing).
    """
    out_keys = [
        "peak_ram_mb", "peak_vram_mb",
        "peak_replay_buffer_mb", "model_params_mb", "optimizer_state_mb",
        "final_eval_mean", "return_per_ram_gb", "return_per_vram_gb"
    ]
    out = {k: np.nan for k in out_keys}
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path)
        for k in out_keys:
            if k in df.columns:
                v = df[k].iloc[0]
                out[k] = float(v) if pd.notna(v) else np.nan
    except Exception as e:
        print(f"[warn] Failed reading {path}: {e}")
    return out


def discover_runs(root: Path) -> List[Path]:
    run_dirs: List[Path] = []
    if not root.exists():
        return run_dirs
    for algo_dir in root.iterdir():
        if not algo_dir.is_dir(): continue
        for env_dir in algo_dir.iterdir():
            if not env_dir.is_dir(): continue
            for run_tag_dir in env_dir.iterdir():
                if run_tag_dir.is_dir():
                    run_dirs.append(run_tag_dir)
    return run_dirs

# ----------------------------- Main aggregation -----------------------------

records: List[Dict[str, Any]] = []
runs = discover_runs(ROOT)
print(f"Discovered {len(runs)} run directories under `{ROOT}`.")

for rd in runs:
    meta = parse_run_metadata(rd)

    # Locate progress.csv (either logs/progress.csv or run root)
    prog_path = rd / "logs" / "progress.csv"
    if not prog_path.exists():
        alt = rd / "progress.csv"
        prog_path = alt if alt.exists() else prog_path
    prog = robust_read_csv(prog_path) if prog_path.exists() else None

    # Memory summary
    mem_path = rd / "memory_summary.csv"
    mem  = read_memory_summary(mem_path)

    # EvalCallback results (best over time)
    eval_dir = rd / "evaluation"
    best_ts, best_mean, best_succ = read_best_eval(eval_dir, door_key_success_proxy=True)

    row: Dict[str, Any] = {"run_dir": str(rd), **meta, **mem}
    row["best_eval_timestep"] = best_ts
    row["final_reward"] = best_mean  # we define "final" as BEST over time for reporting
    row["final_success"] = best_succ

    # If evaluations.npz is absent, try to fallback to CSV eval columns
    if (best_mean is None) or (np.isnan(best_mean)):
        if prog is not None:
            summ = summarize_progress(prog)
            row.update(summ)

            # reward fallback precedence
            fallback_reward = None
            for k in ["eval_mean_csv", "eval_mean_csv_alt", "eval_mean_csv_old"]:
                v = summ.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    fallback_reward = v
                    break
            row["final_reward"] = fallback_reward

            # success fallback: if any success column exists, use it;
            # else, derive proxy: success = (final_reward > 0) on DoorKey
            fallback_success = None
            for k in ["eval_success_csv", "eval_success_masked_csv", "eval_success_csv_old"]:
                v = summ.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    fallback_success = v
                    break
            if fallback_success is None and fallback_reward is not None:
                try:
                    fallback_success = 1.0 if float(fallback_reward) > 0 else 0.0
                except Exception:
                    fallback_success = np.nan
            row["final_success"] = fallback_success

        else:
            row["final_step"] = np.nan
            row["episodes"]   = np.nan

    # Attach exploration means (if we had a CSV)
    if (prog is not None) and (len(prog) > 0):
        summ = summarize_progress(prog)
        for k in ["exploration/unique_obs_mean_last100", "exploration/frontier_rate_mean_last100",
                  "final_train_ep_reward", "final_train_ep_len", "final_ep_len_alt"]:
            row[k] = summ.get(k, np.nan)

    records.append(row)

df_all = pd.DataFrame.from_records(records)
df_all.to_csv(OUT / "all_runs_raw.csv", index=False, encoding="utf-8")
print(f"✔ Saved: {OUT/'all_runs_raw.csv'} ({len(df_all)} rows)")

# ----------------------------- Aggregate across seeds -----------------------------

group_cols = ["algo", "env", "buffer"]
metrics = [
    "final_success", "final_reward",
    "exploration/unique_obs_mean_last100", "exploration/frontier_rate_mean_last100",
    "peak_replay_buffer_mb", "peak_ram_mb", "return_per_ram_gb"
]

if len(df_all) > 0:
    mean_df = df_all.groupby(group_cols, dropna=False)[metrics].mean().reset_index()
    std_df  = df_all.groupby(group_cols, dropna=False)[metrics].std(ddof=1).reset_index()
    df_agg  = mean_df.merge(std_df, on=group_cols, suffixes=("_mean", "_std"))
else:
    cols = group_cols + [m+"_mean" for m in metrics] + [m+"_std" for m in metrics]
    df_agg = pd.DataFrame(columns=cols)

df_agg.to_csv(OUT / "aggregated_by_algo_env_buffer.csv", index=False, encoding="utf-8")
print(f"✔ Saved: {OUT/'aggregated_by_algo_env_buffer.csv'}")

# ----------------------------- Plots -----------------------------

def save_bar_from_pivot(df: pd.DataFrame, value_col: str, title: str, fname: str):
    try:
        if len(df) == 0 or value_col not in df.columns:
            print(f"[warn] Could not plot {fname}: no column {value_col}")
            return
        pivot = df.pivot_table(index=["algo","buffer"], values=value_col)
        if pivot.dropna().empty:
            print(f"[warn] Could not plot {fname}: no numeric data to plot")
            return
        ax = pivot.plot(kind="bar", legend=False, figsize=(9,5))
        ax.set_title(title)
        ax.set_ylabel(value_col)
        ax.set_xlabel("Algorithm / Buffer")
        plt.tight_layout()
        plt.savefig(OUT / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✔ Saved plot: {OUT/fname}")
    except Exception as e:
        print(f"[warn] Could not plot {fname}: {e}")

save_bar_from_pivot(df_agg, "final_success_mean",
                    "Best Evaluation Success (mean) by Algo & Buffer",
                    "bar_final_success.png")

save_bar_from_pivot(df_agg, "peak_replay_buffer_mb_mean",
                    "Peak Replay Buffer (MB, mean) by Algo & Buffer",
                    "bar_peak_replay_mb.png")

# Scatter: memory vs performance (mean)
try:
    if len(df_agg) > 0:
        pts = []
        xs, ys, labels = [], [], []
        for _, r in df_agg.iterrows():
            x = r.get("peak_ram_mb_mean", np.nan)
            y = r.get("final_success_mean", np.nan)
            if pd.notna(x) and pd.notna(y):
                xs.append(x); ys.append(y)
                labels.append(f'{r["algo"]}-buf{int(r["buffer"])//1000 if pd.notna(r["buffer"]) else "?"}k')
        if len(xs) > 0:
            plt.figure(figsize=(8,5))
            plt.scatter(xs, ys)
            for (x,y,l) in zip(xs,ys,labels):
                plt.annotate(l, (x,y), xytext=(5,5), textcoords="offset points", fontsize=8)
            plt.xlabel("Peak RAM (MB, mean)")
            plt.ylabel("Best Eval Success (mean)")
            plt.title("Memory vs Performance (mean by algo/buffer)")
            plt.tight_layout()
            plt.savefig(OUT / "scatter_memory_vs_success.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✔ Saved plot: {OUT/'scatter_memory_vs_success.png'}")
        else:
            print("[warn] Scatter: no numeric points to plot")
except Exception as e:
    print(f"[warn] Could not plot scatter: {e}")

# ----------------------------- README -----------------------------

(OUT / "README_ANALYSIS.txt").write_text(
"""Analysis outputs generated under analysis_outputs/:

- all_runs_raw.csv
    One row per run, with parsed metadata (algo/env/tag/seed/buffer),
    best evaluation reward & success (from evaluations.npz when present),
    CSV fallbacks, and memory statistics.

- aggregated_by_algo_env_buffer.csv
    Mean ± std per (algo, env, buffer) for success, reward, exploration, and memory.

- bar_final_success.png
    Best evaluation success (mean) by algorithm and buffer size.

- bar_peak_replay_mb.png
    Peak replay buffer (MB, mean) by algorithm and buffer size.

- scatter_memory_vs_success.png
    Peak RAM (mean) vs best evaluation success (mean); annotated by algo/buffer.

Notes:
- For MiniGrid-DoorKey-5x5, "success" is proxied as reward > 0 during evaluation episodes.
- "Best" metrics are taken from evaluation/evaluations.npz (SB3 EvalCallback).
  If that file is missing, the script falls back to CSV columns (eval/mean_reward, etc.).
- Buffer size is parsed from run tag tokens such as 'bf25K', 'buf100k', 'buffer_50K'.
""",
encoding="utf-8"
)
print(f"✔ Wrote: {OUT/'README_ANALYSIS.txt'}")
print("Done.")
