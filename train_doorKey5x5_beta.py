# train_doorKey5x5_beta.py
from pathlib import Path
import os, psutil, hashlib, time, csv, random
from collections import deque
from typing import Any, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

# ---- Config ----
ALGO      = "BetaDQN"
ENV_ID    = "MiniGrid-DoorKey-5x5-v0"
RUN_TAG   = "seed_222"
TOTAL_STEPS = 1_000_000
EVAL_FREQ   = 10_000
EVAL_EPISODES = 5
SEED = 222
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Dirs ----
RUN_ROOT  = f"results/{ALGO}/{ENV_ID}/{RUN_TAG}"
TB_DIR    = f"{RUN_ROOT}/tb"
CSV_DIR   = f"{RUN_ROOT}/logs"
EVAL_DIR  = f"{RUN_ROOT}/evaluation"
MODEL_DIR = f"{RUN_ROOT}/models"
FINAL_MODEL_PATH = f"{MODEL_DIR}/{ALGO.lower()}_{ENV_ID}.pth"
for p in [RUN_ROOT, TB_DIR, CSV_DIR, EVAL_DIR, MODEL_DIR]:
    Path(p).mkdir(parents=True, exist_ok=True)

# ---- Minimal TB logger (since we’re not using SB3 here) ----
try:
    from torch.utils.tensorboard import SummaryWriter
    WRITER = SummaryWriter(TB_DIR)
except Exception:
    WRITER = None

# ---- Env builders (same wrappers as SB3 scripts) ----
def make_wrapped_env(env_id: str, seed: int, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)   # obs is HWC uint8
    env.reset(seed=seed)
    return env

# ---- Utils ----
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _to_7x7x3(image: np.ndarray) -> np.ndarray:
    arr = np.array(image)
    if arr.ndim != 3: raise ValueError(f"obs must be HWC/CHW, got {arr.shape}")
    if arr.shape[0] in (3,4) and arr.shape[-1] not in (3,4):
        arr = arr.transpose(1,2,0)
    if arr.shape[-1] > 3: arr = arr[...,:3]
    if arr.shape[-1] < 3:
        pad=3-arr.shape[-1]
        arr=np.concatenate([arr,np.zeros((*arr.shape[:2],pad),dtype=arr.dtype)],axis=-1)
    H,W,_=arr.shape
    if (H,W)!=(7,7):
        r = np.linspace(0,H-1,7).round().astype(int)
        c = np.linspace(0,W-1,7).round().astype(int)
        arr = arr[r][:,c]
    return (arr.astype(np.float32)/10.0)

def obs_to_flat(obs: Any) -> np.ndarray:
    if isinstance(obs, dict) and "image" in obs:
        img = _to_7x7x3(obs["image"])
        chw = img.transpose(2,0,1).reshape(-1)  # 147
        # MiniGrid direction is internal; partial wrapper doesn’t include it.
        dir_onehot = np.zeros(4, dtype=np.float32)
        return np.concatenate([chw, dir_onehot])  # 151
    else:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 3:
            img = _to_7x7x3(arr)
            chw = img.transpose(2,0,1).reshape(-1)
            return np.concatenate([chw, np.zeros(4, np.float32)])
        flat = arr.flatten()
        if flat.size < 151:
            flat = np.pad(flat, (0, 151-flat.size))
        return flat

# ---- Replay ----
class Replay:
    def __init__(self, capacity:int=100_000):
        self.S=[]; self.A=[]; self.R=[]; self.S2=[]; self.D=[]; self.cap=capacity
    def push(self,s,a,r,s2,d):
        if len(self.S)==self.cap:
            i = np.random.randint(0,self.cap)
            self.S[i]=s; self.A[i]=a; self.R[i]=r; self.S2[i]=s2; self.D[i]=d
        else:
            self.S.append(s); self.A.append(a); self.R.append(r); self.S2.append(s2); self.D.append(d)
    def __len__(self): return len(self.S)
    def sample(self, n:int):
        idx = np.random.randint(0,len(self.S),size=n)
        return [ (self.S[i],self.A[i],self.R[i],self.S2[i],self.D[i]) for i in idx ]

# ---- Nets ----
class DQNNet(nn.Module):
    def __init__(self, n_actions:int, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), nn.ReLU(),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64+4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x):
        img, dir_ = x[:,:-4], x[:,-4:]
        B = img.shape[0]
        img = img.view(B,3,7,7)
        z = self.conv(img).view(B,-1)
        return self.fc(torch.cat([z, dir_],1))

class BetaNet(nn.Module):
    def __init__(self, n_actions:int, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), nn.ReLU(),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64+4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x):
        img, dir_ = x[:,:-4], x[:,-4:]
        B = img.shape[0]
        img = img.view(B,3,7,7)
        z = self.conv(img).view(B,-1)
        return self.fc(torch.cat([z, dir_],1))  # logits

# ---- BetaDQN agent ----
class BetaDQN:
    def __init__(self, n_actions:int, batch=64, lr_q=3e-4, lr_b=1e-3, gamma=0.99, epsilon_mask=0.02):
        self.n_actions=n_actions; self.batch=batch; self.gamma=gamma
        self.q = DQNNet(n_actions).to(DEVICE)
        self.tgt = DQNNet(n_actions).to(DEVICE)
        self.b = BetaNet(n_actions).to(DEVICE)
        self.tgt.load_state_dict(self.q.state_dict())
        self.qopt = optim.Adam(self.q.parameters(), lr=lr_q)
        self.bopt = optim.Adam(self.b.parameters(), lr=lr_b)
        self.replay = Replay(100_000)
        self.eps_start=1.0; self.eps_end=0.05; self.eps_decay=200_000
        self.steps=0; self.epsilon_mask=epsilon_mask
        self.loss_q_hist=[]; self.loss_b_hist=[]
    def epsilon(self):
        frac=max(0.0,1.0-self.steps/self.eps_decay)
        return self.eps_end+(self.eps_start-self.eps_end)*frac
    def preprocess(self, obs):
        flat = obs_to_flat(obs)
        return torch.from_numpy(flat).unsqueeze(0).to(DEVICE)
    @torch.no_grad()
    def act_masked_greedy(self, s):
        beta = self.b(s).softmax(-1).squeeze(0)
        q = self.q(s).squeeze(0)
        valid = torch.where(beta > self.epsilon_mask)[0]
        if valid.numel()>0:
            idx = valid[torch.argmax(q[valid])]
            return int(idx.item())
        return int(torch.argmax(q).item())
    def soft_update(self, tau=0.005):
        with torch.no_grad():
            for p,tp in zip(self.q.parameters(), self.tgt.parameters()):
                tp.data.mul_(1-tau).add_(tau*p.data)
    def update_beta(self, batch):
        S = torch.cat([self.preprocess(s) for s,_,_,_,_ in batch])
        A = torch.tensor([a for _,a,_,_,_ in batch], device=DEVICE)
        logits = self.b(S)
        loss = F.cross_entropy(logits, A)
        self.bopt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.b.parameters(), 1.0)
        self.bopt.step()
        self.loss_b_hist.append(float(loss.item()))
    def update_q(self, batch):
        S = torch.cat([self.preprocess(s) for s,_,_,_,_ in batch])
        A = torch.tensor([a for _,a,_,_,_ in batch], device=DEVICE).long().unsqueeze(1)
        R = torch.tensor([r for *_,r,_,_ in batch], device=DEVICE).float().unsqueeze(1)
        S2= torch.cat([self.preprocess(s2) for *_,s2,_ in batch])
        D = torch.tensor([d for *_,d in batch], device=DEVICE).float().unsqueeze(1)
        q_sa = self.q(S).gather(1, A)
        with torch.no_grad():
            q_next_online = self.q(S2)
            q_next_target = self.tgt(S2)
            # masked selector
            beta = self.b(S2).softmax(-1)
            valid = beta > self.epsilon_mask
            sel = q_next_online.clone(); sel[~valid] = -float("inf")
            a_star = torch.argmax(sel, dim=1, keepdim=True)
            # fallback if all masked
            all_masked = (~valid).all(dim=1, keepdim=True)
            a_fallback = torch.argmax(q_next_online, dim=1, keepdim=True)
            a_star = torch.where(all_masked, a_fallback, a_star)
            max_q = q_next_target.gather(1, a_star)
            target = R + (1.0 - D) * self.gamma * max_q
        loss = F.smooth_l1_loss(q_sa, target)
        self.qopt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.qopt.step()
        self.loss_q_hist.append(float(loss.item()))
    def update(self):
        if len(self.replay) < self.batch: return
        batch = self.replay.sample(self.batch)
        self.update_beta(batch); self.update_q(batch)

# ---- Callbacks (SB3-style metrics, but simple) ----
class PeakMemory:
    def __init__(self, run_root:str, eval_dir:str):
        self.proc = psutil.Process(os.getpid())
        self.peak_rss = 0
        self.cuda = torch.cuda.is_available()
        if self.cuda: torch.cuda.reset_peak_memory_stats()
        self.run_root = run_root; self.eval_dir = eval_dir
    def on_step(self, replay):
        rss = self.proc.memory_info().rss
        if rss > self.peak_rss: self.peak_rss = rss
    def on_end(self, final_eval_mean: float|None, model_bytes_mb: float, optim_bytes_mb: float, replay_mb: float):
        peak_ram_mb = self.peak_rss / 1024**2
        peak_vram_mb = (torch.cuda.max_memory_allocated()/1024**2) if torch.cuda.is_available() else None
        ram_gb = peak_ram_mb/1024 if peak_ram_mb>0 else np.nan
        vram_gb = (peak_vram_mb/1024) if peak_vram_mb else np.nan
        ret_per_ram = (final_eval_mean/ram_gb) if (final_eval_mean is not None and np.isfinite(ram_gb)) else ""
        ret_per_vram= (final_eval_mean/vram_gb) if (final_eval_mean is not None and peak_vram_mb and np.isfinite(vram_gb)) else ""
        # write CSV summary
        path = os.path.join(self.run_root,"memory_summary.csv")
        with open(path,"w") as f:
            f.write("peak_ram_mb,peak_vram_mb,peak_replay_buffer_mb,model_params_mb,optimizer_state_mb,final_eval_mean,return_per_ram_gb,return_per_vram_gb\n")
            f.write(f"{peak_ram_mb:.2f},{'' if peak_vram_mb is None else f'{peak_vram_mb:.2f}'},{replay_mb:.2f},{model_bytes_mb:.2f},{optim_bytes_mb:.2f},{'' if final_eval_mean is None else f'{final_eval_mean:.3f}'},{ret_per_ram if ret_per_ram=='' else f'{ret_per_ram:.3f}'},{ret_per_vram if ret_per_vram=='' else f'{ret_per_vram:.3f}'}\n")

class ExplorationMeter:
    def __init__(self):
        self.global_seen=set(); self.ep_seen=set(); self.ep_len=0; self.frontier_hits=0
    @staticmethod
    def hash_obs_hwc_uint8(obs_img: np.ndarray)->int:
        return int.from_bytes(hashlib.blake2b(obs_img.tobytes(), digest_size=8).digest(),"little")
    def step(self, obs_img: np.ndarray):
        h=self.hash_obs_hwc_uint8(obs_img); self.ep_len+=1
        if h not in self.ep_seen:
            self.ep_seen.add(h)
            if h not in self.global_seen:
                self.global_seen.add(h); self.frontier_hits+=1
    def end_episode(self) -> Tuple[int,float]:
        unique_obs = len(self.ep_seen)
        frontier = self.frontier_hits / max(1,self.ep_len)
        self.ep_seen.clear(); self.ep_len=0; self.frontier_hits=0
        return unique_obs, frontier

# ---- Eval ----
@torch.no_grad()
def evaluate(agent: BetaDQN, env: gym.Env, n_episodes:int=5) -> Tuple[float,float]:
    total=0.0; succ=0
    for _ in range(n_episodes):
        obs,_=env.reset(); done=False; ep_r=0.0
        while not done:
            s = agent.preprocess(obs)
            a = int(torch.argmax(agent.q(s)).item())  # greedy Q
            obs,r,term,trunc,_ = env.step(a)
            ep_r += float(r); done = term or trunc
        total += ep_r; succ += (1 if ep_r>0 else 0)
    return total/n_episodes, succ/n_episodes

# ---- Training loop ----
def main():
    set_seed(SEED)
    train_env = make_wrapped_env(ENV_ID, SEED)
    eval_env  = make_wrapped_env(ENV_ID, SEED+100)
    n_actions = train_env.action_space.n
    agent = BetaDQN(n_actions)

    # CSV progress (align with SB3-ish fields)
    prog_path = os.path.join(CSV_DIR, "progress_beta.csv")
    Path(CSV_DIR).mkdir(parents=True, exist_ok=True)
    csv_f = open(prog_path,"w",newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["time/total_timesteps","train/episode_reward","train/episode_length","exploration/unique_obs","exploration/frontier_rate","eval/mean_reward","eval/success","train/loss_q","train/loss_beta"])

    mem = PeakMemory(RUN_ROOT, EVAL_DIR)
    exp = ExplorationMeter()

    obs,_ = train_env.reset()
    ep_r=0.0; ep_len=0
    recent_q_losses=deque(maxlen=100); recent_b_losses=deque(maxlen=100)

    for step in range(1, TOTAL_STEPS+1):
        # ε-greedy over masked-greedy
        if random.random() < agent.epsilon():
            a = train_env.action_space.sample()
        else:
            a = agent.act_masked_greedy(agent.preprocess(obs))

        # exploration counters (HWC image available)
        exp.step(obs)

        obs2, r, term, trunc, _ = train_env.step(a)
        done = term or trunc
        agent.replay.push(obs, a, float(r), obs2, done)

        if step % 2 == 0 and len(agent.replay) >= agent.batch:
            agent.update()
            if agent.loss_q_hist: recent_q_losses.append(agent.loss_q_hist[-1])
            if agent.loss_b_hist: recent_b_losses.append(agent.loss_b_hist[-1])
        agent.soft_update(0.005)

        ep_r += float(r); ep_len += 1
        obs = obs2

        mem.on_step(agent.replay)

        if done:
            uq, fr = exp.end_episode()
            if WRITER:
                WRITER.add_scalar("train/episode_reward", ep_r, step)
                WRITER.add_scalar("train/episode_length", ep_len, step)
                WRITER.add_scalar("exploration/unique_obs", uq, step)
                WRITER.add_scalar("exploration/frontier_rate", fr, step)
                if recent_q_losses: WRITER.add_scalar("train/loss_q", np.mean(recent_q_losses), step)
                if recent_b_losses: WRITER.add_scalar("train/loss_beta", np.mean(recent_b_losses), step)
            writer.writerow([step, f"{ep_r:.4f}", ep_len, uq, f"{fr:.4f}", "", "", f"{np.mean(recent_q_losses) if recent_q_losses else ''}", f"{np.mean(recent_b_losses) if recent_b_losses else ''}"])
            ep_r=0.0; ep_len=0
            obs,_ = train_env.reset()

        if step % EVAL_FREQ == 0:
            mr, succ = evaluate(agent, eval_env, EVAL_EPISODES)
            if WRITER:
                WRITER.add_scalar("eval/mean_reward", mr, step)
                WRITER.add_scalar("eval/success", succ, step)
            # estimate sizes
            model_bytes = sum(p.numel()*p.element_size() for p in agent.q.parameters()) \
                        + sum(p.numel()*p.element_size() for p in agent.tgt.parameters()) \
                        + sum(p.numel()*p.element_size() for p in agent.b.parameters())
            opt_bytes   = 0
            for opt in (agent.qopt, agent.bopt):
                for st in opt.state.values():
                    for v in st.values():
                        if torch.is_tensor(v): opt_bytes += v.numel()*v.element_size()
            # rough replay size (python objects → underestimate/exclude overhead)
            replay_mb = 0.0
            if len(agent.replay)>0:
                # sample a few transitions to estimate bytes
                sample = agent.replay.sample(min(128,len(agent.replay)))
                def nbytes_of(x):
                    if isinstance(x, dict) and "image" in x:
                        return x["image"].nbytes
                    if hasattr(x,"nbytes"): return x.nbytes
                    if torch.is_tensor(x): return x.numel()*x.element_size()
                    return 0
                per = np.mean([ nbytes_of(s)+nbytes_of(s2)+8+8+1 for (s,a,r,s2,d) in sample ])
                replay_mb = (per * len(agent.replay)) / (1024**2)

            # write a small eval row for convenience
            with open(os.path.join(CSV_DIR,"eval_beta.csv"),"a",newline="") as ef:
                ec = csv.writer(ef)
                if ef.tell()==0: ec.writerow(["step","mean_reward","success","loss_q_mean","loss_b_mean","replay_mb"])
                ec.writerow([step, f"{mr:.4f}", f"{succ:.4f}", f"{np.mean(recent_q_losses) if recent_q_losses else np.nan:.6f}", f"{np.mean(recent_b_losses) if recent_b_losses else np.nan:.6f}", f"{replay_mb:.2f}"])

    # end loop
    # final eval
    mr, succ = evaluate(agent, eval_env, EVAL_EPISODES)
    model_bytes = sum(p.numel()*p.element_size() for p in agent.q.parameters()) \
                + sum(p.numel()*p.element_size() for p in agent.tgt.parameters()) \
                + sum(p.numel()*p.element_size() for p in agent.b.parameters())
    opt_bytes   = 0
    for opt in (agent.qopt, agent.bopt):
        for st in opt.state.values():
            for v in st.values():
                if torch.is_tensor(v): opt_bytes += v.numel()*v.element_size()

    # (rough) replay size again
    replay_mb = 0.0
    if len(agent.replay)>0:
        sample = agent.replay.sample(min(256,len(agent.replay)))
        def nbytes_of(x):
            if isinstance(x, dict) and "image" in x: return x["image"].nbytes
            if hasattr(x,"nbytes"): return x.nbytes
            if torch.is_tensor(x): return x.numel()*x.element_size()
            return 0
        per = np.mean([ nbytes_of(s)+nbytes_of(s2)+8+8+1 for (s,a,r,s2,d) in sample ])
        replay_mb = (per * len(agent.replay)) / (1024**2)

    mem.on_end(final_eval_mean=mr, model_bytes_mb=model_bytes/(1024**2), optim_bytes_mb=opt_bytes/(1024**2), replay_mb=replay_mb)

    # save checkpoint
    torch.save({
        "q": agent.q.state_dict(),
        "tgt": agent.tgt.state_dict(),
        "beta": agent.b.state_dict(),
        "q_opt": agent.qopt.state_dict(),
        "b_opt": agent.bopt.state_dict(),
    }, FINAL_MODEL_PATH)
    print(f"✅ Finished. Mean eval reward={mr:.3f}, success={succ:.2f}")
    print(f"💾 Saved: {FINAL_MODEL_PATH}")
    csv_f.close()
    if WRITER: WRITER.close()
    train_env.close(); eval_env.close()

if __name__ == "__main__":
    main()
