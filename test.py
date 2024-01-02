import gymnasium as gym
import os

env = gym.make('Pendulum-v1')
observation = env.reset()
for t in range(100):
    env.render()
    print(observation)
    action = env.action_space.sample()
    observation, reward, done, trunc, info = env.step(action)
    env.render()
    print(reward)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

# from torchrl.envs.libs.gym import GymEnv
# from tensordict import TensorDict, TensorDictBase
# import torch

# env = GymEnv('Pendulum-v1')

# observation = env.reset()
# for t in range(100):
#     env.render()
#     print(observation)
#     action = TensorDict({'action': torch.tensor([0.1])}, batch_size=[])
#     observation, reward, done, trunc, info = env.step(action)
#     print(reward)
#     if done:
#         print("Episode finished after {} timesteps".format(t+1))
#         break
