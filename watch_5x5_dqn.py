# watch_dqn.py
from stable_baselines3 import DQN
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import time
import numpy as np

MODEL_PATH = "results/DoubleDQN/MiniGrid-DoorKey-5x5-v0/bf25k_seed_000/evaluation/best_model.zip"  # <- use the exact filename you saved

# 1) Create a renderable env (window pops up)
env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human")

# 2) Use the same wrappers as training (image-only, partial obs)
env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)

# 3) Load the trained model
model = DQN.load(MODEL_PATH)

# 4) Roll out a few episodes
num_episodes = 5
for ep in range(num_episodes):
    obs, info = env.reset()
    obs = np.transpose(obs, (2, 0, 1))
    done = False
    ep_reward = 0.0
    while not done:
        # model.predict expects a single observation; deterministic=True removes extra randomness
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.02)
        obs = np.transpose(obs, (2, 0, 1))
        ep_reward += reward
        done = terminated or truncated
    print(f"Episode {ep+1}: return = {ep_reward}")

env.close()
print("👋 Done playing.")
