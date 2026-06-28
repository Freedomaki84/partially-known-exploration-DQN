from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# MiniGrid wrappers to convert Dict obs -> image-only array
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

# 1) Make env
env = gym.make("MiniGrid-DoorKey-8x8-v0")

# 2) Keep only the partially observable RGB image (removes 'mission' text, etc.)
env = RGBImgPartialObsWrapper(env)  # agent-centered partial view as RGB
env = ImgObsWrapper(env)            # drop dict, keep obs["image"] only

# (SB3 will auto-wrap, but being explicit is fine)
env = DummyVecEnv([lambda: env])

# 3) Use a CNN policy for image input
model = DQN("CnnPolicy", env, verbose=1, buffer_size=50_000)

# 4) Train briefly to sanity-check
model.learn(total_timesteps=20_000)

# 5) Save
model.save("results/dqn_doorKey_cnn")
print("✅ Training finished and model saved!")