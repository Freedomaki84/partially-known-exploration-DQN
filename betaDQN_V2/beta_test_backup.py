# train_minigrid_beta_dqn.py
# β-DQN on MiniGrid DoorKey-5x5 with per-seed results folders,
# logging ON (TensorBoard + CSV), eval ON, and NO video / NO rendering.

import os
import argparse
import gymnasium as gym
import numpy as np

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from stable_baselines3.common.monitor import Monitor

from beta_dqn_sb3_like import BetaDQN, BetaDQNConfig

import os, torch
torch.set_num_threads(max(1, os.cpu_count()//2))  # or 1


# ---------------------------
# Env helpers (MiniGrid, partial obs, CHW uint8)
# ---------------------------

def make_wrapped_env(env_id: str, seed: int = 0, render_mode=None):
    """
    render_mode=None => fastest (no frame generation).
    We still use RGBImgPartialObsWrapper + ImgObsWrapper to get small HWC images.
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env)  # H×W×3
    env = ImgObsWrapper(env)            # -> H×W×C uint8
    env = Monitor(env)                  # episode stats in info
    env.reset(seed=seed)
    return env


def to_channel_first_uint8(obs):
    if isinstance(obs, np.ndarray) and obs.ndim == 3:
        H, W, C = obs.shape
        return np.transpose(obs, (2, 0, 1))
    raise ValueError("Unexpected observation shape; expected HWC ndarray.")


class CHWWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old = env.observation_space
        assert len(old.shape) == 3 and old.dtype == np.uint8
        H, W, C = old.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(C, H, W), dtype=np.uint8)

    def observation(self, obs):
        return to_channel_first_uint8(obs)


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train β-DQN on MiniGrid DoorKey-5x5 (logs+eval ON, no video/render).")
    p.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-5x5-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--eval_every", type=int, default=50_000, help="Evaluate every N steps")
    p.add_argument("--results_root", type=str, default="./results", help="Root folder for all outputs")
    # Optional quick overrides for throttling knobs
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--entropy_every", type=int, default=None)
    p.add_argument("--csv_flush_every", type=int, default=None)
    return p.parse_args()


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()

    # Per-seed folder structure
    seed_tag = f"seed_{args.seed:03d}"
    run_root = os.path.join(args.results_root, seed_tag)
    videos_dir = os.path.join(run_root, "videos")  # kept for consistency, unused
    tb_dir     = os.path.join(run_root, "tb")
    logs_dir   = os.path.join(run_root, "logs")
    models_dir = os.path.join(run_root, "models")
    for d in [videos_dir, tb_dir, logs_dir, models_dir]:
        os.makedirs(d, exist_ok=True)

    # Envs — NO RENDER ANYWHERE, NO VIDEO WRAPPERS
    base_train_env = make_wrapped_env(args.env_id, seed=args.seed, render_mode=None)
    train_env = CHWWrapper(base_train_env)

    base_eval_env = make_wrapped_env(args.env_id, seed=args.seed + 100, render_mode=None)
    eval_env = CHWWrapper(base_eval_env)

    # Specs
    obs_shape = train_env.observation_space.shape
    n_actions = train_env.action_space.n

    # Config (logging ON, throttled)
    run_name = f"BetaDQN_{args.env_id}_{seed_tag}"
    cfg = BetaDQNConfig(
        seed=args.seed,
        tensorboard_log=tb_dir,                              # TensorBoard logs
        run_name=run_name,
        save_path=os.path.join(models_dir, "beta_dqn.pt"),
        csv_log_path=os.path.join(logs_dir, "progress.csv"),# progress.csv
        buffer_size=200_000,
        lr_q=2.5e-4,
        lr_beta=2.5e-4,
        learning_starts=10_000,
        target_update_freq=1_000,
        eps_decay_steps=1_000_000,
        cov_deltas=(0.05, 0.10),
        cor_alphas=tuple([0.1 * i for i in range(0, 11)]),
        meta_window=1000,
        plot_replay_ram=True,
        # Throttling (speed-friendly)
        device="cpu",          # force CPU
        train_freq=16,         # fewer optimizer steps
        batch_size=32,         # cheaper updates
        log_every=5000,        # TB less often
        entropy_every=100000,  # nearly off
        csv_flush_every=10000, # buffer CSV
    )

    # Optional CLI overrides
    if args.log_every is not None:
        cfg.log_every = int(args.log_every)
    if args.entropy_every is not None:
        cfg.entropy_every = int(args.entropy_every)
    if args.csv_flush_every is not None:
        cfg.csv_flush_every = int(args.csv_flush_every)

    # Train
    agent = BetaDQN(obs_shape=obs_shape, n_actions=n_actions, cfg=cfg)
    agent.learn(
        env=train_env,
        eval_env=eval_env,                 # eval ON (no video, no render)
        total_timesteps=args.steps,
        log_interval=1000,
        tb_log_name=run_name,
        reset_num_timesteps=True,
        progress_bar=True,                 # keep tqdm if you want a live ETA
        eval_every=args.eval_every
    )

    # Save & report
    agent.save(cfg.save_path)
    print("\n=== Run outputs ===")
    print("Seed:            ", args.seed)
    print("Results root:    ", run_root)
    print("Model:           ", cfg.save_path)
    print("Progress CSV:    ", cfg.csv_log_path)
    print("TensorBoard dir: ", os.path.join(tb_dir, run_name))
    print("Videos dir:      ", videos_dir, "(unused: no video)")

    # Short eval summary at the end (without rendering)
    mean_ret, succ = agent.evaluate(eval_env, episodes=10)
    print(f"Final evaluation — mean_return={mean_ret:.2f}, success_rate={succ:.2%}")


if __name__ == "__main__":
    main()
