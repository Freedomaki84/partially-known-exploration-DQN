# watch_betaDQN.py
import argparse
import time
import numpy as np
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

from beta_dqn_sb3_like import BetaDQN, BetaDQNConfig

# --- same CHW helper used in training ---
def to_channel_first_uint8(obs):
    if isinstance(obs, np.ndarray) and obs.ndim == 3:
        return np.transpose(obs, (2, 0, 1))
    raise ValueError("Unexpected obs shape; expected HWC ndarray.")

class CHWWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old = env.observation_space
        assert len(old.shape) == 3 and old.dtype == np.uint8
        H, W, C = old.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(C, H, W), dtype=np.uint8
        )
    def observation(self, obs):
        return to_channel_first_uint8(obs)

def make_env(env_id, seed=0, render_mode="human"):
    env = gym.make(env_id, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env)   # H×W×3
    env = ImgObsWrapper(env)             # H×W×C uint8
    env = CHWWrapper(env)                # C×H×W uint8
    env.reset(seed=seed)
    return env

def set_greedy(agent, greedy: bool):
    """Try to turn off exploration for greedy evaluation."""
    if not greedy:
        return
    # Common knobs your BetaDQN might expose
    if hasattr(agent, "set_epsilon"):
        try:
            agent.set_epsilon(0.0)
            return
        except Exception:
            pass
    if hasattr(agent, "epsilon"):
        try:
            agent.epsilon = 0.0
            return
        except Exception:
            pass
    # As a fallback, no change—agent.act will behave as implemented.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="Path to ./results/seed_xxx/models/beta_dqn.pt")
    ap.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-5x5-v0")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.02, help="Delay between steps (s)")
    ap.add_argument("--greedy", action="store_true",
                    help="Force epsilon=0 (greedy actions) if supported")
    args = ap.parse_args()

    # Load agent
    agent = BetaDQN.load(args.model)
    print(f"Loaded model: {args.model}")
    print(f"Env: {args.env_id}")
    print(f"Episodes: {args.episodes}")

    # Greedy toggle (sets epsilon=0 if the agent exposes it)
    set_greedy(agent, args.greedy)

    # Renderable env using SAME wrappers as training (including CHW)
    env = make_env(args.env_id, seed=123, render_mode="human")

    # Roll out
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        ep_ret = 0.0
        done = False

        while not done:
            # pick your eval epsilon
            eps = 0.0 if args.greedy else 0.05

            out = agent.act(obs, warmup_eps=eps)

            # handle both cases: int or (action, info)
            action = out[0] if isinstance(out, tuple) else out
            action = int(action)  # ensure plain int

            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated

            if args.sleep > 0:
                time.sleep(args.sleep)

        print(f"Episode {ep}: return = {ep_ret:.2f}")

    env.close()
    print("👋 Done playing.")

if __name__ == "__main__":
    main()



#python watch_betaDQN.py --model ./results/seed_000/models/beta_dqn.pt --greedy