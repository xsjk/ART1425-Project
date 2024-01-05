import hsup as _
import gymnasium as gym


from dataset.data import data
from dataset.simulation import model_indoor, model_sec_back_t

env = gym.make(
    "hsup/HeatSupply-v0",
    data=data,
    model_indoor=model_indoor,
    model_sec_back_t=model_sec_back_t,
)
