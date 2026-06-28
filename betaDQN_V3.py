"""
Strict + Faithful β-DQN (MiniGrid DoorKey-5x5)
------------------------------------------------
- ε decay: 1M → 0.05
- Sliding window L = 100
- Policy set: cov(0.05), cov(0.10), cor(0.3)
- Buffer size = 100k
- Random warm-up: 10k steps
- Strict β masking (no fallback)
- Train 5M steps
- Logs: exploration metrics, eval masked/Q, memory_summary.csv
"""

import os, csv, time, random, hashlib, psutil
from collections import defaultdict
from typing import Any
import gymnasium as gym
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TBOARD_OK = True
except Exception:
    TBOARD_OK = False

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Utils ----------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(env_id: str, seed: int = 0):
    env = gym.make(env_id)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env.reset(seed=seed)
    return env

def _to_7x7x3(image: np.ndarray) -> np.ndarray:
    arr = np.array(image)
    if arr.ndim == 3 and arr.shape[0] in (3,4) and arr.shape[-1] not in (3,4):
        arr = np.transpose(arr, (1,2,0))
    if arr.shape[-1] > 3: arr = arr[...,:3]
    if arr.shape[-1] < 3:
        pad = 3 - arr.shape[-1]
        arr = np.concatenate([arr, np.zeros((*arr.shape[:2], pad),dtype=arr.dtype)],axis=-1)
    H,W,_ = arr.shape
    if (H,W)!=(7,7):
        r_idx = np.linspace(0,H-1,7).round().astype(int)
        c_idx = np.linspace(0,W-1,7).round().astype(int)
        arr = arr[r_idx][:,c_idx]
    return arr.astype(np.float32)/10.0

def hash_obs(img: np.ndarray) -> int:
    return int.from_bytes(
        hashlib.blake2b(np.asarray(img,dtype=np.uint8).tobytes(),digest_size=8).digest(),
        "little"
    )

# ---------------- Replay ----------------
class ReplayMemory:
    def __init__(self, capacity: int):
        from collections import deque
        self.memory = deque(maxlen=capacity)
    def push(self,s,a,r,s2,d): self.memory.append((s,a,r,s2,d))
    def sample(self,n): return random.sample(self.memory,n)
    def __len__(self): return len(self.memory)

# ---------------- Nets ----------------
class QNetwork(nn.Module):
    def __init__(self,n_actions,hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),nn.ReLU(),
            nn.Conv2d(16,32,3,1,1),nn.ReLU(),
            nn.Conv2d(32,64,3,1,1),nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64+4,hidden),nn.ReLU(),
            nn.Linear(hidden,hidden),nn.ReLU(),
            nn.Linear(hidden,n_actions)
        )
    def forward(self,x):
        img, d = x[:,:-4], x[:,-4:]
        B = img.shape[0]; img = img.view(B,3,7,7)
        z = self.conv(img).view(B,-1)
        return self.fc(torch.cat([z,d],1))

class BetaNetwork(nn.Module):
    def __init__(self,n_actions,hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),nn.ReLU(),
            nn.Conv2d(16,32,3,1,1),nn.ReLU(),
            nn.Conv2d(32,64,3,1,1),nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64+4,hidden),nn.ReLU(),
            nn.Linear(hidden,hidden),nn.ReLU(),
            nn.Linear(hidden,n_actions)
        )
    def forward(self,x):
        img, d = x[:,:-4], x[:,-4:]
        B=img.shape[0]; img=img.view(B,3,7,7)
        z=self.conv(img).view(B,-1)
        return self.fc(torch.cat([z,d],1)) # logits

# ---------------- Agent ----------------
class BetaDQN:
    def __init__(self,n_actions,lr=3e-4,gamma=0.99,
                 eps_start=1.0,eps_end=0.05,eps_decay=1_000_000,
                 mem_size=100_000,batch_size=32,target_update=1000,
                 L=100,hidden=128,eps_mask=0.05):
        self.n_actions=n_actions; self.gamma=gamma
        self.eps_start=eps_start; self.eps_end=eps_end; self.eps_decay=eps_decay
        self.batch_size=batch_size; self.target_update=target_update
        self.eps_mask=eps_mask; self.L=L

        self.q=QNetwork(n_actions,hidden).to(device)
        self.tgt=QNetwork(n_actions,hidden).to(device)
        self.beta=BetaNetwork(n_actions,hidden).to(device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.q_opt=optim.Adam(self.q.parameters(),lr=lr)
        self.b_opt=optim.Adam(self.beta.parameters(),lr=lr)
        self.mem=ReplayMemory(mem_size)

        self.delta_vals=[0.05,0.10]; self.alpha_vals=[0.3]
        self.policies=self._make_policies()
        self.history=[]; self.steps=0

    def _make_policies(self):
        d={}
        for δ in self.delta_vals: d[f"cov({δ:.2f})"]={"type":"cov","δ":δ}
        for α in self.alpha_vals: d[f"cor({α:.1f})"]={"type":"cor","α":α}
        return d

    def epsilon(self):
        frac=max(0,1-self.steps/self.eps_decay)
        return self.eps_end+(self.eps_start-self.eps_end)*frac

    def preprocess(self,obs:Any):
        if isinstance(obs,dict) and "image" in obs:
            img=obs["image"]; d=int(obs.get("direction",0))
        else: img,d=obs,0
        one=np.zeros(4,np.float32); one[d%4]=1
        arr=_to_7x7x3(img); chw=np.transpose(arr,(2,0,1)).reshape(-1)
        flat=np.concatenate([chw,one]).astype(np.float32)
        return torch.from_numpy(flat).unsqueeze(0).to(device)

    def select_policy(self):
        if not self.history: return random.choice(list(self.policies))
        window=self.history[-self.L:]
        counts=defaultdict(int); R=defaultdict(float); E=defaultdict(float)
        for p,r,e in window: counts[p]+=1; R[p]+=r; E[p]+=e
        unused=[p for p in self.policies if counts[p]==0]
        if unused: return random.choice(unused)
        best,score=None,-1e9
        for p in self.policies:
            mu=R[p]/counts[p]; b=E[p]/counts[p]; s=mu+b
            if s>score: score, best=s,p
        return best

    @torch.no_grad()
    def act_cov(self,s,δ):
        beta=self.beta(s).softmax(-1).squeeze(0); q=self.q(s).squeeze(0)
        low=(beta<=δ).nonzero(as_tuple=True)[0]
        if len(low)==0: return int(torch.argmax(q))
        return int(low[torch.randint(0,len(low),(1,))])

    @torch.no_grad()
    def act_cor(self,s,α):
        beta=self.beta(s).softmax(-1).squeeze(0); q=self.q(s).squeeze(0)
        qh=q.clone(); qh[beta<=self.eps_mask]=torch.min(q)
        return int(torch.argmax(α*q+(1-α)*qh))

    def act_under_policy(self,s,policy):
        meta=self.policies[policy]
        return self.act_cov(s,meta["δ"]) if meta["type"]=="cov" else self.act_cor(s,meta["α"])

    def update(self):
        if len(self.mem)<self.batch_size: return None,None
        batch=self.mem.sample(self.batch_size)
        S=torch.cat([self.preprocess(x[0]) for x in batch])
        A=torch.tensor([x[1] for x in batch],device=device).long()
        R=torch.tensor([x[2] for x in batch],device=device).float()
        S2=torch.cat([self.preprocess(x[3]) for x in batch])
        D=torch.tensor([x[4] for x in batch],device=device).bool()

        # β update
        loss_b=F.cross_entropy(self.beta(S),A)
        self.b_opt.zero_grad(); loss_b.backward(); self.b_opt.step()

        # Q update (strict β mask; no fallback)
        q_sa=self.q(S).gather(1,A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next=self.tgt(S2)
            beta_next=self.beta(S2).softmax(-1)
            valid=beta_next>self.eps_mask
            masked=q_next.clone(); masked[~valid]=-1e9
            max_q,_=masked.max(1)
            max_q[D]=0.0
            target=R+self.gamma*max_q
        loss_q=F.smooth_l1_loss(q_sa,target)
        self.q_opt.zero_grad(); loss_q.backward(); self.q_opt.step()
        return float(loss_q.item()),float(loss_b.item())

    def update_target(self): self.tgt.load_state_dict(self.q.state_dict())

# ---------------- Train ----------------
def train_beta_dqn(env_id="MiniGrid-DoorKey-5x5-v0",total_steps=5_000_000,
                   log_interval=10_000,eval_interval=50_000,
                   seed=0,run_dir="results/BetaDQN_strict"):
    set_seed(seed)
    os.makedirs(run_dir,exist_ok=True)
    csv_path=os.path.join(run_dir,"progress.csv")
    csv_file=open(csv_path,"w",newline="")
    csv_writer=csv.writer(csv_file)
    csv_writer.writerow([
        "step","episodes","epsilon",
        "train/ep_reward","train/ep_len",
        "exploration/unique_obs","exploration/frontier_rate",
        "loss/q","loss/b","recent_policy",
        "eval_masked/mean_reward","eval_masked/success",
        "eval_q/mean_reward","eval_q/success"
    ])
    csv_file.flush()

    writer=SummaryWriter(run_dir) if TBOARD_OK else None
    env=make_env(env_id,seed); eval_env=make_env(env_id,seed+100)
    agent=BetaDQN(n_actions=env.action_space.n)
    proc=psutil.Process(os.getpid())
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    # Warm-up (random)
    obs,_=env.reset()
    for _ in range(10_000):
        a=env.action_space.sample()
        obs2,r,term,trunc,_=env.step(a)
        agent.mem.push(obs,a,float(r),obs2,term or trunc)
        if term or trunc: obs,_=env.reset()
        else: obs=obs2

    # Tracking
    global_seen=set(); ep_seen=set(); frontier_hits=0; ep_len=0
    obs,_=env.reset(); s=agent.preprocess(obs)
    ep_r=0.0; episodes=0; pol=agent.select_policy()
    last_log=0; last_eval=0
    pbar=tqdm(total=total_steps,desc="β-DQN strict")

    for step in range(1,total_steps+1):
        if random.random()<agent.epsilon():
            a,polname=env.action_space.sample(),"eps"
        else:
            a,polname=agent.act_under_policy(s,pol),pol

        img=obs["image"] if isinstance(obs,dict) else obs
        h=hash_obs(img); ep_len+=1
        if h not in ep_seen:
            ep_seen.add(h)
            if h not in global_seen:
                global_seen.add(h); frontier_hits+=1

        obs2,r,term,trunc,_=env.step(a)
        d=term or trunc
        agent.mem.push(obs,a,float(r),obs2,d)
        lq,lb=agent.update()
        if step%agent.target_update==0: agent.update_target()
        ep_r+=float(r); obs=obs2; s=agent.preprocess(obs); agent.steps=step

        if d:
            episodes+=1
            unique_obs=len(ep_seen)
            frontier_rate=frontier_hits/max(1,ep_len)
            agent.history.append((pol,ep_r,frontier_rate))
            if len(agent.history)>agent.L: agent.history.pop(0)
            csv_writer.writerow([
                step,episodes,f"{agent.epsilon():.4f}",
                f"{ep_r:.4f}",ep_len,
                unique_obs,f"{frontier_rate:.4f}",
                f"{lq:.6f}" if lq else "",
                f"{lb:.6f}" if lb else "",
                polname,"","","",""
            ]); csv_file.flush()
            if writer:
                writer.add_scalar("rollout/ep_reward",ep_r,step)
                writer.add_scalar("rollout/ep_len",ep_len,step)
                writer.add_scalar("exploration/unique_obs",unique_obs,step)
                writer.add_scalar("exploration/frontier_rate",frontier_rate,step)
                if lq: writer.add_scalar("train/loss_q",lq,step)
                if lb: writer.add_scalar("train/loss_b",lb,step)
            ep_r,ep_len,frontier_hits=0.0,0,0; ep_seen.clear()
            obs,_=env.reset(); s=agent.preprocess(obs); pol=agent.select_policy()

        # RAM log
        if (step-last_log)>=log_interval:
            rss_mb=proc.memory_info().rss/1024**2
            if writer: writer.add_scalar("memory/ram_mb",rss_mb,step)
            last_log=step

        # Eval (both masked + greedy)
        if (step-last_eval)>=eval_interval:
            masked_rewards,masked_succ,q_rewards,q_succ=[],[],[],[]
            for _ in range(10):
                # masked eval
                o,_=eval_env.reset(); done=False; ep=0.0
                while not done:
                    with torch.no_grad():
                        s2=agent.preprocess(o)
                        beta=agent.beta(s2).softmax(-1).squeeze(0)
                        q=agent.q(s2).squeeze(0)
                        valid=torch.where(beta>agent.eps_mask)[0]
                        if len(valid)>0:
                            a=int(valid[torch.argmax(q[valid])].item())
                        else: a=int(torch.argmax(q).item())
                    o,r,t,tr,_=eval_env.step(a); ep+=float(r); done=t or tr
                masked_rewards.append(ep); masked_succ.append(1 if ep>0 else 0)

                # q-greedy eval
                o,_=eval_env.reset(); done=False; ep=0.0
                while not done:
                    with torch.no_grad():
                        a=int(torch.argmax(agent.q(agent.preprocess(o)).squeeze(0)).item())
                    o,r,t,tr,_=eval_env.step(a); ep+=float(r); done=t or tr
                q_rewards.append(ep); q_succ.append(1 if ep>0 else 0)

            mean_mr=np.mean(masked_rewards); mean_ms=np.mean(masked_succ)
            mean_qr=np.mean(q_rewards); mean_qs=np.mean(q_succ)
            csv_writer.writerow([
                step,episodes,"","","","","","","","",
                f"{mean_mr:.4f}",f"{mean_ms:.4f}",
                f"{mean_qr:.4f}",f"{mean_qs:.4f}"
            ]); csv_file.flush()
            if writer:
                writer.add_scalar("eval/masked_reward",mean_mr,step)
                writer.add_scalar("eval/masked_success",mean_ms,step)
                writer.add_scalar("eval/q_reward",mean_qr,step)
                writer.add_scalar("eval/q_success",mean_qs,step)
            last_eval=step

        pbar.update(1)

    pbar.close(); csv_file.close()
    if writer: writer.close()

    # Memory summary
    peak_ram_mb=proc.memory_info().rss/1024**2
    peak_vram_mb=(torch.cuda.max_memory_allocated()/1024**2) if torch.cuda.is_available() else None
    with open(os.path.join(run_dir,"memory_summary.csv"),"w") as f:
        f.write("peak_ram_mb,peak_vram_mb\n")
        f.write(f"{peak_ram_mb:.2f},"+(f"{peak_vram_mb:.2f}" if peak_vram_mb else "")+"\n")

    print(f"✅ Training complete. Logs in: {run_dir}")

if __name__=="__main__":
    train_beta_dqn()
