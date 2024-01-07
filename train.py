import torch
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

import hsup as _
import gymnasium as gym
from stable_baselines3 import DQN

import dataset.preprocess as preprocess
import dataset.simulation as simulation
import dataset.data as data

model_sec_back_t = simulation.MLPModel.load_from_checkpoint("lightning_logs/split.sec_back_t/checkpoints/epoch=816-step=90687.ckpt")
model_indoor = simulation.MLPModel.load_from_checkpoint("lightning_logs/split.indoor/checkpoints/epoch=814-step=90465.ckpt")

model_sec_back_t.eval()
model_indoor.eval()

env = gym.make(
    "hsup/HeatSupply",
    max_episode_steps=24 * 6,
    data=data.train_data,
    model_sec_back_t=model_sec_back_t,    
    model_indoor=model_indoor,
)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10, progress_bar=True)
