Analysis outputs generated under analysis_outputs/:

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
