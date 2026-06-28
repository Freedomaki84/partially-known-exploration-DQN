# train_doorKey5x5_dqn_exploration.py
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import os, psutil, hashlib
import torch
import numpy as np

# ----------------------
# Config
# ----------------------
ALGO      = "DQN"                         # or "DoubleDQN", "BetaDQN"
ENV_ID    = "MiniGrid-DoorKey-5x5-v0"
RUN_TAG   = "bf50K_seed_222"                    # e.g., "seed_123", "run_01", timestamp
TOTAL_STEPS = 1_000_000
EVAL_FREQ   = 10_000
EVAL_EPISODES = 5
SEED = 222

# Run directories
RUN_ROOT  = f"results/{ALGO}/{ENV_ID}/{RUN_TAG}"
TB_DIR    = f"{RUN_ROOT}/tb"
CSV_DIR   = f"{RUN_ROOT}/logs"
EVAL_DIR  = f"{RUN_ROOT}/evaluation"
MODEL_DIR = f"{RUN_ROOT}/models"
FINAL_MODEL_PATH = f"{MODEL_DIR}/{ALGO.lower()}_{ENV_ID}.zip"

for p in [RUN_ROOT, TB_DIR, CSV_DIR, EVAL_DIR, MODEL_DIR]:
    Path(p).mkdir(parents=True, exist_ok=True)

# ----------------------
# Memory logging callback (your original, unchanged)
# ----------------------
class PeakMemoryCallback(BaseCallback):
    def __init__(self, run_root: str, eval_dir: str, check_interval_steps: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.run_root = run_root
        self.eval_dir = eval_dir
        self.check_interval_steps = check_interval_steps
        self.process = psutil.Process(os.getpid())
        self.peak_rss_bytes = 0
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()
        self.peak_replay_mb = 0.0

    def _buffer_memory_mb(self):
        buf = getattr(self.model, "replay_buffer", None)
        if buf is None:
            return None
        total = 0
        for name in ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]:
            if hasattr(buf, name):
                arr = getattr(buf, name)
                if arr is not None and hasattr(arr, "nbytes"):
                    total += arr.nbytes
        return total / 1024**2

    def _model_and_optim_memory_mb(self):
        model_bytes = 0
        for p in self.model.policy.parameters():
            if p is not None:
                model_bytes += p.nelement() * p.element_size()
        model_mb = model_bytes / 1024**2
        optim_bytes = 0
        opt = getattr(self.model.policy, "optimizer", None)
        if opt is not None:
            for p, st in opt.state.items():
                for tensor in st.values():
                    if torch.is_tensor(tensor):
                        optim_bytes += tensor.nelement() * tensor.element_size()
        optim_mb = optim_bytes / 1024**2
        return model_mb, optim_mb

    def _on_step(self) -> bool:
        if self.n_calls % self.check_interval_steps == 0:
            rss = self.process.memory_info().rss
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
            self.logger.record("memory/ram_mb", rss / 1024**2)

            if self.cuda_available:
                self.logger.record("memory/vram_allocated_mb", torch.cuda.memory_allocated() / 1024**2)

            rb_mb = self._buffer_memory_mb()
            if rb_mb is not None:
                self.logger.record("memory/replay_buffer_mb", rb_mb)
                if rb_mb > self.peak_replay_mb:
                    self.peak_replay_mb = rb_mb

            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.dump(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        peak_ram_mb = self.peak_rss_bytes / 1024**2
        peak_vram_mb = None
        if self.cuda_available:
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

        final_eval_mean = None
        eval_npz = os.path.join(self.eval_dir, "evaluations.npz")
        if os.path.exists(eval_npz):
            try:
                data = np.load(eval_npz, allow_pickle=True)
                results = data.get("results")
                if results is not None and results.size > 0:
                    final_eval_mean = float(results[-1].mean())
            except Exception:
                pass

        model_mb, optim_mb = self._model_and_optim_memory_mb()

        ram_gb  = peak_ram_mb / 1024 if peak_ram_mb > 0 else np.nan
        vram_gb = (peak_vram_mb / 1024) if (peak_vram_mb is not None and peak_vram_mb > 0) else np.nan
        if final_eval_mean is not None and np.isfinite(ram_gb):
            self.logger.record("memory_norm/return_per_ram_gb", final_eval_mean / ram_gb)
        if final_eval_mean is not None and peak_vram_mb is not None and np.isfinite(vram_gb):
            self.logger.record("memory_norm/return_per_vram_gb", final_eval_mean / vram_gb)

        self.logger.record("memory/peak_ram_mb", float(f"{peak_ram_mb:.2f}"))
        if peak_vram_mb is not None:
            self.logger.record("memory/peak_vram_mb", float(f"{peak_vram_mb:.2f}"))
        if self.peak_replay_mb:
            self.logger.record("memory/peak_replay_buffer_mb", float(f"{self.peak_replay_mb:.2f}"))
        self.logger.record("memory/model_params_mb", float(f"{model_mb:.2f}"))
        self.logger.record("memory/optimizer_state_mb", float(f"{optim_mb:.2f}"))
        self.logger.record("memory/model_plus_optim_mb", float(f"{(model_mb+optim_mb):.2f}"))
        self.logger.dump(self.num_timesteps)

        os.makedirs(self.run_root, exist_ok=True)
        summary_path = os.path.join(self.run_root, "memory_summary.csv")
        with open(summary_path, "w") as f:
            f.write("peak_ram_mb,peak_vram_mb,peak_replay_buffer_mb,model_params_mb,optimizer_state_mb,final_eval_mean,return_per_ram_gb,return_per_vram_gb\n")
            vals = [
                f"{peak_ram_mb:.2f}",
                "" if peak_vram_mb is None else f"{peak_vram_mb:.2f}",
                f"{self.peak_replay_mb:.2f}" if self.peak_replay_mb else "",
                f"{model_mb:.2f}",
                f"{optim_mb:.2f}",
                "" if final_eval_mean is None else f"{final_eval_mean:.3f}",
                "" if (final_eval_mean is None or not np.isfinite(ram_gb)) else f"{final_eval_mean/ram_gb:.3f}",
                "" if (final_eval_mean is None or peak_vram_mb is None or not np.isfinite(vram_gb)) else f"{final_eval_mean/vram_gb:.3f}",
            ]
            f.write(",".join(vals) + "\n")

        if self.verbose:
            msg = f"[Memory] Peak RAM: {peak_ram_mb:.2f} MB"
            if peak_vram_mb is not None:
                msg += f" | Peak VRAM: {peak_vram_mb:.2f} MB"
            if self.peak_replay_mb:
                msg += f" | Peak Replay Buffer: {self.peak_replay_mb:.2f} MB"
            msg += f" | Model: {model_mb:.2f} MB | Optimizer: {optim_mb:.2f} MB"
            print(msg)
            if final_eval_mean is not None and np.isfinite(ram_gb):
                print(f"[Memory] Return per RAM GB: {final_eval_mean/ram_gb:.3f}")
            if final_eval_mean is not None and peak_vram_mb is not None and np.isfinite(vram_gb):
                print(f"[Memory] Return per VRAM GB: {final_eval_mean/vram_gb:.3f}")

# ----------------------
# NEW: Exploration metrics callback (unique obs & frontier rate)
# ----------------------
class ExplorationCallback(BaseCallback):
    """
    Logs per-episode exploration metrics:
      - exploration/unique_obs
      - exploration/frontier_rate
    Works with vectorized env (uses the first env index) and CHW obs (VecTransposeImage).
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.global_seen = set()
        self.ep_seen = set()
        self.ep_len = 0
        self.frontier_hits = 0

    @staticmethod
    def _hash_obs(obs_chw_uint8: np.ndarray) -> int:
        # obs is (C,H,W) uint8 due to VecTransposeImage
        return int.from_bytes(hashlib.blake2b(obs_chw_uint8.tobytes(), digest_size=8).digest(), "little")

    def _on_step(self) -> bool:
        obs_batch = self.locals.get("new_obs", None)  
        dones = self.locals.get("dones", None)        
        if obs_batch is None or dones is None:
            return True

        obs0 = obs_batch[0]
        h = self._hash_obs(obs0)
        self.ep_len += 1

        if h not in self.ep_seen:
            self.ep_seen.add(h)
            if h not in self.global_seen:
                self.global_seen.add(h)
                self.frontier_hits += 1

        if bool(dones[0]):  # end of episode for env 0
            unique_obs = len(self.ep_seen)
            frontier_rate = self.frontier_hits / max(1, self.ep_len)
            self.logger.record("exploration/unique_obs", unique_obs)
            self.logger.record("exploration/frontier_rate", frontier_rate)
            # reset episode stats
            self.ep_seen.clear()
            self.ep_len = 0
            self.frontier_hits = 0

        return True

# ----------------------
# Env builders (same wrappers)
# ----------------------
def make_wrapped_env(env_id: str, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env

train_env = DummyVecEnv([lambda: make_wrapped_env(ENV_ID)])
train_env = VecTransposeImage(train_env)
train_env.seed(SEED)

eval_env = DummyVecEnv([lambda: make_wrapped_env(ENV_ID)])
eval_env = VecTransposeImage(eval_env)
eval_env.seed(SEED + 100)

# ----------------------
# Logger (CSV + TensorBoard + stdout under RUN_ROOT)
# ----------------------
logger = configure(RUN_ROOT, ["stdout", "csv", "tensorboard"])

# ----------------------
# Evaluation callback
# ----------------------
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=EVAL_DIR,
    log_path=EVAL_DIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=1,
)

# ----------------------
# Model (CnnPolicy)
# ----------------------
model = DQN(
    "CnnPolicy",
    train_env,
    seed=SEED,
    verbose=1,
    batch_size=32,
    buffer_size=50_000,
    gamma=0.99,
    target_update_interval=1_000,
    train_freq=4,
    gradient_steps=1,             
    exploration_fraction=0.5,    
    exploration_final_eps=0.01,
    learning_rate=1e-4,
    learning_starts=10_000,
    device="auto",
)
model.set_logger(logger)

# ----------------------
# Train
# ----------------------
mem_callback = PeakMemoryCallback(RUN_ROOT, EVAL_DIR, check_interval_steps=1_000, verbose=1)
exp_callback = ExplorationCallback(verbose=0)  
callback = CallbackList([eval_callback, mem_callback, exp_callback])

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=callback,
    tb_log_name="dqn_doorKey5x5",
    progress_bar=True,
)

# ----------------------
# Save final model
# ----------------------
model.save(FINAL_MODEL_PATH)
print(f"✅ Training finished. Final model saved to: {FINAL_MODEL_PATH}")
print(f"👉 Best checkpoint (by eval reward) saved under: {EVAL_DIR}/best_model.zip")
print("📊 Open TensorBoard with:")
print(f"    tensorboard --logdir {RUN_ROOT}")
print(f"   (TB events and CSV logs are both under: {RUN_ROOT})")
