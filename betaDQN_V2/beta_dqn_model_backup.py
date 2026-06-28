# beta_dqn_sb3_like.py
# β-DQN for MiniGrid (discrete), SB3-like API, CNN encoder, TensorBoard logs, tqdm progress bar
# Now with correct per-episode logging:
#   - progress.csv now has an `ep_reward_finished` column written exactly at episode end
#   - a separate episodes.csv writes one row per finished episode (episode, return, length)
#   - TB also logs per-episode series keyed by episode index
#
# Paper alignment:
#  - Eq.(2) supervised β head (cross-entropy on (s,a) from replay)
#  - Eq.(5) masked TD target using β(a'|s')>ε, fallback if no valid action
#  - Eq.(6) masked greedy exploitation
#  - Eq.(7) coverage policy π_cov(δ)
#  - Eq.(8-9) correction policy π_cor(α)
#  - Eq.(10-12) meta-controller over policy set Π using sliding window of returns & exploration

import os
import csv
import random
from dataclasses import dataclass
from collections import deque
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ---------------------------
# Replay Buffer
# ---------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)      # store images as uint8 (0..255)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = action
        self.rew[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx])

    def ram_megabytes(self) -> float:
        bytes_total = (
            self.obs.nbytes + self.next_obs.nbytes +
            self.act.nbytes + self.rew.nbytes + self.done.nbytes
        )
        return bytes_total / (1024**2)


# ---------------------------
# Networks (CNN encoder + dual heads)
# ---------------------------

class CNNEncoder(nn.Module):
    """
    A small CNN for MiniGrid partial observations (C,H,W), C=3 in RGB wrappers, H=W ~ 7, 11, etc.
    """
    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
        )
        self.fc = None
        self.out_dim = out_dim

    def forward(self, x):
        z = self.conv(x)
        if self.fc is None:
            n_flat = z.shape[1] * z.shape[2] * z.shape[3]
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_flat, self.out_dim),
                nn.ReLU(),
            )
        return self.fc(z)


class BetaQNet(nn.Module):
    """
    Shared CNN trunk, two heads:
      - Q head: Q(s,·)
      - β head: categorical β(·|s) (softmax over logits)
    """
    def __init__(self, in_channels: int, n_actions: int, feat_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.enc = CNNEncoder(in_channels, out_dim=feat_dim)
        self.q_head = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
        self.beta_head = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)  # logits
        )

    def forward(self, x):
        f = self.enc(x)
        q = self.q_head(f)
        beta_logits = self.beta_head(f)
        beta_prob = torch.softmax(beta_logits, dim=-1)
        return q, beta_prob, beta_logits


# ---------------------------
# Config
# ---------------------------

@dataclass
class BetaDQNConfig:
    gamma: float = 0.99
    batch_size: int = 64
    lr_q: float = 2.5e-4
    lr_beta: float = 2.5e-4
    buffer_size: int = 200_000
    target_update_freq: int = 1_000
    train_freq: int = 4
    learning_starts: int = 10_000

    eps_start: float = 1.0
    eps_final: float = 0.01
    eps_decay_steps: int = 1_000_000

    beta_mask_eps: float = 0.05  # ε in paper (threshold in Eq.5 & Eq.6)

    cov_deltas: Tuple[float, ...] = (0.05, 0.10)  # δ for π_cov
    cor_alphas: Tuple[float, ...] = tuple([0.1 * i for i in range(0, 11)])  # α in {0.0..1.0}

    meta_window: int = 1000   # sliding window L
    eval_episodes: int = 5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    # Logging (paths)
    tensorboard_log: Optional[str] = "./tb_logs"
    run_name: str = "BetaDQN"
    save_path: Optional[str] = "./beta_dqn_ckpt.pt"   # torch.save state_dict
    save_every_steps: int = 50_000
    plot_replay_ram: bool = True
    csv_log_path: str = "./progress.csv"

    # NEW: throttling knobs (for speed)
    log_every: int = 200         # write TB scalars every N steps
    entropy_every: int = 5000    # compute beta entropy every N steps
    csv_flush_every: int = 2000  # flush progress.csv every N steps


# ---------------------------
# SB3-like Agent
# ---------------------------

class BetaDQN:
    """
    SB3-like API:
      - learn(total_timesteps, callback=None, log_interval=10, tb_log_name="BetaDQN", reset_num_timesteps=True, progress_bar=True)
      - save(path), load(path, env_specs)
    Expect Gymnasium env with obs in (C,H,W) uint8, actions discrete.
    """

    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int, cfg: BetaDQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.n_actions = n_actions
        c, h, w = obs_shape
        self.obs_shape = (c, h, w)

        # Nets
        self.q = BetaQNet(in_channels=c, n_actions=n_actions).to(self.device)
        self.q_target = BetaQNet(in_channels=c, n_actions=n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        # Separate optimizers (β vs rest)
        beta_params, q_params = [], []
        for name, p in self.q.named_parameters():
            if "beta_head" in name:
                beta_params.append(p)
            else:
                q_params.append(p)

        self.opt_q = torch.optim.Adam(q_params, lr=cfg.lr_q)
        self.opt_beta = torch.optim.Adam(beta_params, lr=cfg.lr_beta)

        # Replay
        self.replay = ReplayBuffer(cfg.buffer_size, self.obs_shape)

        # Policy set Π
        self.policies = []
        for d in cfg.cov_deltas:
            self.policies.append(("cov", d))
        for a in cfg.cor_alphas:
            self.policies.append(("cor", a))
        self.n_policies = len(self.policies)

        # Meta-controller stats (sliding window)
        self.window_policy_idx = deque(maxlen=cfg.meta_window)
        self.window_returns = deque(maxlen=cfg.meta_window)
        self.window_explore_ratio = deque(maxlen=cfg.meta_window)

        # Episode stats
        self.ep_reward = 0.0
        self.ep_len = 0
        self.ep_index = 0
        self.cur_policy_index = 0
        self.cur_policy_exploratory = 0
        self.cur_policy_actions = 0

        # Global
        self.global_step = 0
        self.epsilon = cfg.eps_start
        self.ram_over_steps = []  # for optional RAM plot

        # Logging
        self.writer: Optional[SummaryWriter] = None
        self._policy_pick_counts: Dict[int, int] = {i: 0 for i in range(self.n_policies)}
        self._csv_fp = None
        self._csv_writer: Optional[csv.DictWriter] = None

        # New: episodes.csv
        self._episodes_fp = None
        self._episodes_csv: Optional[csv.DictWriter] = None

        # Debug: track if any non-zero rewards have ever been seen (per-step)
        self._nonzero_reward_seen = 0

        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    # ---- Helpers ----

    @staticmethod
    def _to_float_tensor_uint8(obs_uint8: np.ndarray, device) -> torch.Tensor:
        if obs_uint8.ndim == 3:
            obs_uint8 = obs_uint8[None, ...]
        x = torch.from_numpy(obs_uint8).to(device=device, dtype=torch.uint8)
        return x.float().div_(255.0)

    def _decay_eps(self):
        t = min(self.global_step, self.cfg.eps_decay_steps)
        frac = 1.0 - t / float(self.cfg.eps_decay_steps)
        self.epsilon = self.cfg.eps_final + (self.cfg.eps_start - self.cfg.eps_final) * max(frac, 0.0)

    @torch.no_grad()
    def _pure_exploit(self, obs_t: torch.Tensor) -> torch.Tensor:
        q, beta_prob, _ = self.q(obs_t)
        mask = (beta_prob > self.cfg.beta_mask_eps).float()
        masked_q = q - 1e9 * (1.0 - mask)
        any_valid = mask.sum(-1) > 0
        greedy_all = q.argmax(dim=-1)
        greedy_masked = masked_q.argmax(dim=-1)
        return torch.where(any_valid, greedy_masked, greedy_all)

    @torch.no_grad()
    def _cov_action(self, obs_t: torch.Tensor, delta: float) -> torch.Tensor:
        _, beta_prob, _ = self.q(obs_t)
        low = (beta_prob <= delta)
        B = obs_t.shape[0]
        actions = []
        for i in range(B):
            idxs = torch.where(low[i])[0]
            if idxs.numel() > 0:
                a = idxs[torch.randint(0, idxs.numel(), (1,))].item()
            else:
                a = self._pure_exploit(obs_t[i:i+1]).item()
            actions.append(a)
        return torch.tensor(actions, device=obs_t.device)

    @torch.no_grad()
    def _cor_action(self, obs_t: torch.Tensor, alpha: float) -> torch.Tensor:
        q, beta_prob, _ = self.q(obs_t)
        min_q = q.min(dim=-1, keepdim=True).values
        keep = (beta_prob > self.cfg.beta_mask_eps).float()
        q_hat = keep * q + (1.0 - keep) * min_q
        combined = alpha * q + (1.0 - alpha) * q_hat
        return combined.argmax(dim=-1)

    @torch.no_grad()
    def _select_policy_index(self) -> int:
        counts = np.zeros(self.n_policies, dtype=int)
        sums = np.zeros(self.n_policies, dtype=float)
        bonus = np.zeros(self.n_policies, dtype=float)
        for pi_idx, ret, expl in zip(self.window_policy_idx, self.window_returns, self.window_explore_ratio):
            counts[pi_idx] += 1
            sums[pi_idx] += ret
            bonus[pi_idx] += expl

        for i in range(self.n_policies):
            if counts[i] == 0:
                return i

        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        b = np.divide(bonus, counts, out=np.zeros_like(bonus), where=counts > 0)
        score = means + b
        return int(np.argmax(score))

    @torch.no_grad()
    def act(self, obs_uint8: np.ndarray, warmup_eps: float) -> Tuple[int, bool]:
        if random.random() < warmup_eps:
            return random.randrange(self.n_actions), True

        obs_t = self._to_float_tensor_uint8(obs_uint8, self.device)
        kind, param = self.policies[self.cur_policy_index]
        if kind == "cov":
            a = self._cov_action(obs_t, delta=param).item()
            exploratory = True
        else:
            a_cor = self._cor_action(obs_t, alpha=param).item()
            a_exp = self._pure_exploit(obs_t).item()
            a = a_cor
            exploratory = (a != a_exp)
        return a, exploratory

    def _train_step(self):
        if self.replay.size < max(self.cfg.learning_starts, self.cfg.batch_size):
            return 0.0, 0.0

        if self.global_step % self.cfg.train_freq != 0:
            return 0.0, 0.0

        obs, act, rew, nxt, done = self.replay.sample(self.cfg.batch_size)
        obs_t = self._to_float_tensor_uint8(obs, self.device)
        nxt_t = self._to_float_tensor_uint8(nxt, self.device)
        act_t = torch.as_tensor(act, device=self.device, dtype=torch.long)
        rew_t = torch.as_tensor(rew, device=self.device, dtype=torch.float32)
        done_t = torch.as_tensor(done, device=self.device, dtype=torch.float32)

        # Optional debug: fraction of non-zero rewards in this batch
        if self.writer is not None and self.global_step % self.cfg.log_every == 0:
            nz = (rew_t != 0).float().mean().item()
            self.writer.add_scalar("debug/batch_frac_nonzero_rewards", nz, self.global_step)

        # β loss (Eq. 2) — supervised on current net
        _, _, beta_logits = self.q(obs_t)
        loss_beta = F.cross_entropy(beta_logits, act_t)

        # Q loss with masked target (Eq. 5)
        with torch.no_grad():
            # Q(s',·) from TARGET net (stable bootstrap)
            q_next, _, _ = self.q_target(nxt_t)
            # β(s',·) from CURRENT net (paper-consistent mask source)
            _, beta_next, _ = self.q(nxt_t)

            mask = (beta_next > self.cfg.beta_mask_eps).float()
            masked_q = q_next - 1e9 * (1.0 - mask)
            use_masked = mask.sum(-1) > 0

            max_all = q_next.max(dim=-1).values
            max_masked = masked_q.max(dim=-1).values
            q_next_max = torch.where(use_masked, max_masked, max_all)

            target = rew_t + self.cfg.gamma * (1.0 - done_t) * q_next_max

        q_pred, _, _ = self.q(obs_t)
        q_taken = q_pred.gather(1, act_t.view(-1, 1)).squeeze(1)
        loss_q = F.mse_loss(q_taken, target)

        # Optimizers
        self.opt_beta.zero_grad()
        loss_beta.backward()
        self.opt_beta.step()

        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()

        # Target update
        if self.global_step % self.cfg.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss_q.item()), float(loss_beta.item())

    def begin_episode(self):
        # meta-controller choice (Eq. 11-12)
        self.cur_policy_index = self._select_policy_index() if len(self.window_policy_idx) > 0 else 0
        self._policy_pick_counts[self.cur_policy_index] += 1
        self.ep_reward = 0.0
        self.ep_len = 0
        self.cur_policy_exploratory = 0
        self.cur_policy_actions = 0

    def end_episode(self):
        self.window_policy_idx.append(self.cur_policy_index)
        self.window_returns.append(self.ep_reward)
        ratio = (self.cur_policy_exploratory / max(1, self.cur_policy_actions))
        self.window_explore_ratio.append(ratio)

    def _log_step(self, env_step_return: Optional[float], loss_q: float, loss_beta: float):
        # --- TensorBoard (throttled) ---
        if self.writer is not None:
            # compute entropy occasionally (expensive extra forward)
            if (self.global_step % self.cfg.entropy_every) == 0 and self.replay.size > 0:
                with torch.no_grad():
                    idx = np.random.randint(0, self.replay.size, size=min(256, self.replay.size))
                    o = self._to_float_tensor_uint8(self.replay.obs[idx], self.device)
                    _, beta_prob, _ = self.q(o)
                    ent = (-beta_prob.clamp_min(1e-8).log() * beta_prob).sum(dim=-1).mean().item()
                    self.writer.add_scalar("beta/entropy", ent, self.global_step)

            # add these when _log_step is called (already throttled by caller)
            self.writer.add_scalar("train/loss_q", loss_q, self.global_step)
            self.writer.add_scalar("train/loss_beta", loss_beta, self.global_step)

            if env_step_return is not None:
                self.writer.add_scalar("rollout/ep_rew_last", env_step_return, self.global_step)

            # meta policy fractions
            total_picks = sum(self._policy_pick_counts.values())
            if total_picks > 0:
                for i, (kind, param) in enumerate(self.policies):
                    frac = self._policy_pick_counts[i] / total_picks
                    tag = f"meta/policy_frac_{i:02d}_{kind}({param})"
                    self.writer.add_scalar(tag, frac, self.global_step)

            # replay RAM (cheap)
            ram_mb = self.replay.ram_megabytes()
            self.ram_over_steps.append((self.global_step, ram_mb))
            self.writer.add_scalar("memory/replay_buffer_mb", ram_mb, self.global_step)

        # --- CSV (progress.csv) buffered ---
        if self._csv_writer is not None:
            kind, param = self.policies[self.cur_policy_index]
            self._csv_writer.writerow({
                "step": self.global_step,
                "episode": self.ep_index,
                "ep_reward": self.ep_reward,           # may be mid-episode or zero after reset
                "ep_reward_finished": "",              # filled only at real episode end
                "ep_len": self.ep_len,
                "epsilon": self.epsilon,
                "loss_q": loss_q,
                "loss_beta": loss_beta,
                "replay_mb": self.replay.ram_megabytes(),
                "policy_index": self.cur_policy_index,
                "policy_kind": kind,
                "policy_param": float(param),
            })
            # flush only every N steps (big speedup vs flush-every-step)
            if (self.global_step % self.cfg.csv_flush_every) == 0:
                self._csv_fp.flush()

    def _open_csv(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header = [
            "step", "episode",
            "ep_reward",              # mid-episode/reset value (kept for compatibility)
            "ep_reward_finished",     # reward of the just-finished episode
            "ep_len",
            "epsilon",
            "loss_q", "loss_beta",
            "replay_mb",
            "policy_index", "policy_kind", "policy_param",
            "eval_mean_reward", "eval_success_rate"
        ]
        file_exists = os.path.isfile(path)
        self._csv_fp = open(path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=header)
        if not file_exists:
            self._csv_writer.writeheader()
        else:
            # If you appended to an old CSV with a different header, consider deleting it first.
            pass

    def _close_csv(self):
        if self._csv_fp:
            self._csv_fp.flush()
            self._csv_fp.close()
            self._csv_fp = None
            self._csv_writer = None

    def _open_episodes_csv(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        is_new = not os.path.isfile(path)
        self._episodes_fp = open(path, "a", newline="", encoding="utf-8")
        self._episodes_csv = csv.DictWriter(self._episodes_fp, fieldnames=["episode", "return", "length"])
        if is_new:
            self._episodes_csv.writeheader()

    def _close_episodes_csv(self):
        if self._episodes_fp:
            self._episodes_fp.flush()
            self._episodes_fp.close()
            self._episodes_fp = None
            self._episodes_csv = None

    # ---- Public API ----

    def learn(self,
              env,
              eval_env=None,
              total_timesteps: int = 1_000_000,
              log_interval: int = 10,
              tb_log_name: Optional[str] = None,
              reset_num_timesteps: bool = True,
              progress_bar: bool = True,
              eval_every: int = 10_000):
        """
        env, eval_env: Gymnasium envs with:
          - observation: (C,H,W) uint8
          - action_space: Discrete
        """
        # Writer
        tb_dir = self.cfg.tensorboard_log or "./tb_logs"
        run_name = tb_log_name or self.cfg.run_name
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(tb_dir, run_name))

        # CSV (progress)
        self._open_csv(self.cfg.csv_log_path)

        # CSV (episodes.csv) — placed next to progress.csv
        try:
            csv_dir = os.path.dirname(self.cfg.csv_log_path) or "."
            self._open_episodes_csv(os.path.join(csv_dir, "episodes.csv"))
        except Exception as e:
            print(f"[warn] could not open episodes.csv: {e}")

        obs, info = env.reset(seed=self.cfg.seed)
        self.begin_episode()

        pbar = tqdm(total=total_timesteps, disable=not progress_bar, desc="Training", unit="step")
        last_logged_return = None

        while self.global_step < total_timesteps:
            # Act
            action, is_expl = self.act(obs, warmup_eps=self.epsilon)
            self.cur_policy_actions += 1
            if is_expl:
                self.cur_policy_exploratory += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # Store
            self.replay.add(obs, action, reward, next_obs, float(done))
            if reward != 0:
                self._nonzero_reward_seen += 1
                if self.writer is not None:
                    self.writer.add_scalar("debug/nonzero_reward_seen", self._nonzero_reward_seen, self.global_step)

            self.ep_reward += reward
            self.ep_len += 1

            # Train
            self.global_step += 1
            self._decay_eps()
            loss_q, loss_beta = self._train_step()

            # THROTTLED logging
            if (self.global_step % self.cfg.log_every) == 0:
                self._log_step(None, loss_q, loss_beta)

            # Step bookkeeping
            obs = next_obs
            if done:
                # success flag if env provides (MiniGrid often sets 'success' in info)
                success = float(info.get("success", 1.0 if self.ep_reward > 0 else 0.0))
                if self.writer is not None:
                    self.writer.add_scalar("rollout/ep_len", self.ep_len, self.global_step)
                    self.writer.add_scalar("rollout/ep_rew", self.ep_reward, self.global_step)
                    self.writer.add_scalar("rollout/success", success, self.global_step)

                last_logged_return = self.ep_reward

                # ---- Write finished episode to CSVs BEFORE reset ----
                # Mirror into progress.csv with explicit finished column
                if self._csv_writer is not None:
                    kind, param = self.policies[self.cur_policy_index]
                    self._csv_writer.writerow({
                        "step": self.global_step,
                        "episode": self.ep_index,
                        "ep_reward": "",  # left blank intentionally; use finished field
                        "ep_reward_finished": float(last_logged_return),
                        "ep_len": int(self.ep_len),
                        "epsilon": float(self.epsilon),
                        "loss_q": 0.0,
                        "loss_beta": 0.0,
                        "replay_mb": self.replay.ram_megabytes(),
                        "policy_index": self.cur_policy_index,
                        "policy_kind": kind,
                        "policy_param": float(param),
                        "eval_mean_reward": "",
                        "eval_success_rate": ""
                    })
                    if (self.global_step % self.cfg.csv_flush_every) == 0:
                        self._csv_fp.flush()

                # Append to episodes.csv (one clean row per episode)
                if self._episodes_csv is not None:
                    self._episodes_csv.writerow({
                        "episode": int(self.ep_index),
                        "return":  float(last_logged_return),
                        "length":  int(self.ep_len),
                    })
                    if (self.ep_index % 50) == 0:
                        self._episodes_fp.flush()

                # TB: also log by episode index
                if self.writer is not None:
                    self.writer.add_scalar("rollout/ep_reward_per_episode", last_logged_return, self.ep_index)
                    self.writer.add_scalar("rollout/ep_len_per_episode", self.ep_len, self.ep_index)

                # Finish and reset
                self.end_episode()
                self.ep_index += 1
                obs, info = env.reset()
                self.begin_episode()

            # Periodic eval (also mirrored into CSV)
            if eval_env is not None and (self.global_step % eval_every == 0):
                mean_ret, success_rate = self.evaluate(eval_env)
                if self.writer is not None:
                    self.writer.add_scalar("eval/mean_reward", mean_ret, self.global_step)
                    self.writer.add_scalar("eval/success_rate", success_rate, self.global_step)
                if self._csv_writer is not None:
                    self._csv_writer.writerow({
                        "step": self.global_step,
                        "episode": self.ep_index,
                        "ep_reward": self.ep_reward,
                        "ep_reward_finished": "",
                        "ep_len": self.ep_len,
                        "epsilon": self.epsilon,
                        "loss_q": 0.0,
                        "loss_beta": 0.0,
                        "replay_mb": self.replay.ram_megabytes(),
                        "policy_index": self.cur_policy_index,
                        "policy_kind": self.policies[self.cur_policy_index][0],
                        "policy_param": float(self.policies[self.cur_policy_index][1]),
                        "eval_mean_reward": float(mean_ret),
                        "eval_success_rate": float(success_rate),
                    })
                    if (self.global_step % self.cfg.csv_flush_every) == 0:
                        self._csv_fp.flush()

            # SB3-like extra logging cadence (optional)
            if self.global_step % log_interval == 0 and last_logged_return is not None:
                self._log_step(last_logged_return, loss_q, loss_beta)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                "ep_rew": f"{(last_logged_return if last_logged_return is not None else 0):.1f}",
                "eps": f"{self.epsilon:.3f}",
                "buf": f"{self.replay.size}/{self.replay.capacity}"
            })

        pbar.close()

        # Cleanup
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        self._close_csv()
        self._close_episodes_csv()

        # Optional RAM plot
        if self.cfg.plot_replay_ram and len(self.ram_over_steps) > 1:
            try:
                import matplotlib.pyplot as plt
                xs, ys = zip(*self.ram_over_steps)
                plt.figure()
                plt.plot(xs, ys)
                plt.xlabel("steps")
                plt.ylabel("replay RAM (MB)")
                plt.title("Replay Buffer RAM over Steps")
                out = "ram_replay_over_steps.png"
                plt.savefig(out, bbox_inches="tight")
                print(f"[info] Saved RAM plot to {out}")
            except Exception as e:
                print(f"[warn] Could not save RAM plot: {e}")

        return self

    @torch.no_grad()
    def evaluate(self, eval_env, episodes: Optional[int] = None):
        episodes = episodes or self.cfg.eval_episodes
        returns = []
        successes = 0
        for _ in range(episodes):
            obs, info = eval_env.reset()
            ret = 0.0
            for _ in range(2000):
                obs_t = self._to_float_tensor_uint8(obs, self.device)
                a = self._pure_exploit(obs_t).item()
                obs, r, term, trunc, info = eval_env.step(a)
                ret += r
                if term or trunc:
                    break
            returns.append(ret)
            successes += int(info.get("success", 1 if ret > 0 else 0))
        return float(np.mean(returns)), successes / max(1, episodes)

    def save(self, path: str):
        torch.save({
            "state_dict": self.q.state_dict(),
            "cfg": self.cfg.__dict__,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "global_step": self.global_step,
        }, path)

    @staticmethod
    def load(path: str, device: Optional[str] = None) -> "BetaDQN":
        import torch, numpy as np
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # (PyTorch 2.6) try safe loader first, fall back if needed
        try:
            from torch.serialization import add_safe_globals
            add_safe_globals([np.core.multiarray.scalar, np.dtype, type(np.float64(0)), type(np.int64(0))])
        except Exception:
            pass
        try:
            blob = torch.load(path, map_location=dev, weights_only=True)
        except Exception:
            blob = torch.load(path, map_location=dev, weights_only=False)

        cfg = BetaDQNConfig(**blob["cfg"])
        agent = BetaDQN(obs_shape=tuple(blob["obs_shape"]), n_actions=int(blob["n_actions"]), cfg=cfg)

        # ---- Robust state_dict load ----
        sd = blob["state_dict"]
        model_sd = agent.q.state_dict()

        # Keep only keys that exist AND have the same shape
        filtered = {k: v for k, v in sd.items() if (k in model_sd and model_sd[k].shape == v.shape)}
        missing = [k for k in model_sd.keys() if k not in filtered]
        unexpected = [k for k in sd.keys() if k not in model_sd]

        # Load non-strict, then report what happened
        agent.q.load_state_dict(filtered, strict=False)
        agent.q_target.load_state_dict(agent.q.state_dict(), strict=True)

        if unexpected or missing:
            print("[BetaDQN.load] Note:")
            if unexpected:
                print("  - Ignored unexpected keys from checkpoint:", unexpected[:10], ("...+%d more" % (len(unexpected)-10) if len(unexpected) > 10 else ""))
            if missing:
                print("  - Model had missing keys initialized randomly:", missing[:10], ("...+%d more" % (len(missing)-10) if len(missing) > 10 else ""))

        agent.global_step = int(blob.get("global_step", 0))
        return agent
