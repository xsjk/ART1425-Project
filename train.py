import hsup as _
import gymnasium as gym
from stable_baselines3 import PPO

from dataset.preprocess import train_data
from dataset.simulation import MLPModel

model_sec_back_t = MLPModel.load_from_checkpoint('lightning_logs/split.sec_back_t/checkpoints/epoch=816-step=90687.ckpt').eval()
model_indoor = MLPModel.load_from_checkpoint('lightning_logs/split.indoor/checkpoints/epoch=814-step=90465.ckpt').eval()

model_sec_back_t.requires_grad_(False)
model_indoor.requires_grad_(False)

env = gym.make(
    "hsup/HeatSupply",
    data=train_data,
    model_indoor=model_indoor,
    model_sec_back_t=model_sec_back_t,
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10, progress_bar=True)
