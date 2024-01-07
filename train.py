import torch
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

import hsup as _
import gymnasium as gym

import dataset.preprocess as preprocess
import dataset.simulation as simulation
import dataset.data as data

max_eposide_steps = 24
discretize = False


model_sec_back_t = simulation.MLPModel.load_from_checkpoint("lightning_logs/split.sec_back_t/checkpoints/epoch=816-step=90687.ckpt")
model_indoor = simulation.MLPModel.load_from_checkpoint("lightning_logs/split.indoor/checkpoints/epoch=814-step=90465.ckpt")

model_sec_back_t.eval()
model_indoor.eval()

env = gym.make(
    "hsup/HeatSupply",
    max_episode_steps=max_eposide_steps,
    data=data.train_data,
    model_sec_back_t=model_sec_back_t,    
    model_indoor=model_indoor,
    discretize=discretize,
)

import stable_baselines3
import torch
from stable_baselines3 import DQN, PPO, SAC

# model = DQN("MlpPolicy", env, 
#             verbose=1, 
#             tensorboard_log="./runs", 
#             seed=0,
#             policy_kwargs=dict(
#                 activation_fn=torch.nn.ReLU,
#                 net_arch=[64, 64, 64, 64],
#             ))

model = SAC("MlpPolicy", env, 
            verbose=1, 
            tensorboard_log="./runs", 
            seed=0)

total_timesteps = int(1e4)
log_interval = 1

model_class = model.__class__.__name__


model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=log_interval)
model.save(f"checkpoints/{model_class}-{total_timesteps}")
