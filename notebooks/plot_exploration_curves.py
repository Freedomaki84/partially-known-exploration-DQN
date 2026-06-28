import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results\DQN\MiniGrid-DoorKey-5x5-v0\seed_000\progress.csv")

plt.figure(figsize=(8,4))
plt.plot(df["time/total_timesteps"], df["exploration/unique_obs"], label="Unique obs")
plt.plot(df["time/total_timesteps"], df["exploration/frontier_rate"], label="Frontier rate")
plt.xlabel("Training steps")
plt.ylabel("Exploration metrics")
plt.legend()
plt.title("DQN – 100K buffer (seed 000)")
plt.show()
