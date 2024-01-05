import random
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F

class BaseAgent(nn.Module):

    @property
    def hparams(self):
        try:
            return self.__hparams
        except:
            raise NotImplementedError
    
    @hparams.setter
    def hparams(self, hparams):
        self.__hparams = hparams

    def act(self, state) -> tuple['Action', 'LogProb']:
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx) -> dict[str, float]:
        raise NotImplementedError
    
    def setup(self):
        pass

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        agent = cls(checkpoint['hparams'])
        agent.load_state_dict(checkpoint['model_state_dict'])
        return agent


#######################################################################################

class ZeroAgent(BaseAgent):
    def act(self, state):
        return 0, None

########################################################################################

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: tuple[int]):
        super().__init__()
        self.layers = []
        dims = (state_dim,) + hidden_dim
        for i in range(len(dims) - 1):
            self.layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
            ]
        self.layers += [
            nn.Linear(hidden_dim[-1], action_dim),
            nn.Tanh()
        ]
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, state):
        return 2 * self.layers(state)

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: tuple[int]) -> None:
        super().__init__()
        self.layers = []
        dims = (state_dim + action_dim,) + hidden_dim
        for i in range(len(dims) - 1):
            self.layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
            ]
        self.layers.append(nn.Linear(hidden_dim[-1], 1))
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, state, action) -> Tensor:
        return self.layers(torch.cat([state, action], 1))


class ReplayBuffer(Dataset):
    def __init__(self, state_dim: int, action_dim: int, max_size: int=100000, device: str='cpu') -> None:
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        # self.buffers = [torch.empty((max_size, ) + dim, dtype=torch.float32, device=device) for dim in dims]
        self.state = torch.empty((max_size, state_dim), dtype=torch.float32, device=device)
        self.action = torch.empty((max_size, action_dim), dtype=torch.float32, device=device)
        self.next_state = torch.empty((max_size, state_dim), dtype=torch.float32, device=device)
        self.reward = torch.empty(max_size, dtype=torch.float32, device=device)
        self.done = torch.empty(max_size, dtype=torch.float32, device=device)

    def push(self, state, action, reward, next_state, done) -> None:
        self.state[self.ptr] = torch.tensor(state, device=self.device)
        self.action[self.ptr] = torch.tensor(action, device=self.device)
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = torch.tensor(next_state, device=self.device)
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (self.state[idx], self.action[idx], self.reward[idx], self.next_state[idx], self.done[idx])

    def to(self, device: str):
        r = object.__new__(ReplayBuffer)
        r.device = device
        r.max_size = self.max_size
        r.ptr = self.ptr
        r.size = self.size
        r.state = self.state.to(device)
        r.action = self.action.to(device)
        r.next_state = self.next_state.to(device)
        r.reward = self.reward.to(device)
        r.done = self.done.to(device)
        return r
    
    def __len__(self) -> int:
        return self.size


# DDPG Agent
class DDPGAgent(BaseAgent):
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            hidden_dim: tuple[int]=(256, 256), 
            memory_size: int=100000,
            lr_actor: float=1e-6,
            lr_critic: float=1e-4,
            gamma: float=0.99,
            soft_tau: float=1e-2,
            batch_size: int=1024,
            device: str='cpu'
        ) -> None:
        super().__init__()
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, memory_size)
        self.gamma = gamma
        self.soft_tau = soft_tau
        self.batch_size = batch_size
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.hparams = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_dim': hidden_dim,
            'memory_size': memory_size,
            'lr_actor': lr_actor,
            'lr_critic': lr_critic,
            'gamma': gamma,
            'soft_tau': soft_tau,
            'batch_size': batch_size
        }
        self.device = device
        self.loss = nn.MSELoss()
        
    def forward(self, state) -> Tensor:
        action = self.actor(state)
        return action, None
    
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self(state)

    def to(self, device) -> 'DDPGAgent':
        r = super().to(device)
        r.replay_buffer = r.replay_buffer.to(device)
        r.device = device
        return r

    def update(self) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # Update Critic
        q_value = self.critic(state, action).squeeze()
        next_action = self.actor_target(next_state)
        target_q_value = self.critic_target(next_state, next_action.detach()).squeeze()
        expected_q_value = reward + self.gamma * target_q_value * (1 - done)
        critic_loss = self.loss(q_value, expected_q_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        policy_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Soft Update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

        return {
            'Loss/Critic': critic_loss.item(), 
            'Loss/Policy':policy_loss.item(),
        }

    def training_step(self, batch, batch_idx):
        state, action, log_prob, next_state, reward, done, truncated, info = batch
        self.replay_buffer.push(state, action, reward, next_state, done)
        metrics = self.update()
        return metrics
    
########################################################################################

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mean = nn.Linear(32, action_dim)
        self.log_std = nn.Linear(32, action_dim)

    def forward(self, state):
        x = self.fc(state)
        mean = 2 * self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return mean, std


class ContinuousPolicyGradientAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.5, device='cpu'):
        super().__init__()
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.normal_dist = torch.distributions.Normal(loc=0, scale=1)
        self.device = device
        self.hparams = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'lr': lr,
            'gamma': gamma,
        }

    def forward(self, state):
        mean, std = self.policy(state)
        z = self.normal_dist.sample()
        
        y = mean + std * z 
        log_prob = self.normal_dist.log_prob(z) - torch.log(std)

        y = torch.tanh(y)
        log_prob -= torch.log(1 - y**2 + 1e-6)

        y = y * 2
        log_prob -= np.log(2)

        return y, log_prob
    
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self(state)
    
    def update(self, rewards, log_probs):
        log_probs = torch.cat(log_probs)
        n = len(rewards)
        discounts =  self.gamma ** np.arange(0, n)
        discounted_rewards = np.empty(n, dtype=np.float32)
        for t in range(n):
            discounted_rewards[t] = np.dot(rewards[t:], discounts[:n-t])

        policy_gradient = -torch.dot(log_probs, torch.tensor(discounted_rewards, device=self.device))
        
        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()

        return {
            'Loss/Policy': policy_gradient.item()
        }

    def setup(self):
        self.log_probs = []
        self.rewards = []
        pass

    def training_step(self, batch, batch_idx):
        state, action, log_prob, next_state, reward, done, truncated, info = batch
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        if done or truncated:
            metrics = self.update(self.rewards, self.log_probs)
            self.rewards.clear()
            self.log_probs.clear()
            return metrics


    def to(self, device) -> 'ContinuousPolicyGradientAgent':
        r = super().to(device)
        r.device = device
        return r


########################################################################################


class PIDAgent(BaseAgent):
    def __init__(self, kp=1, ki=0, kd=5, indoor_col_idx=0):
        super().__init__()
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.indoor_col_idx = indoor_col_idx
        self.hparams = {
            'kp': kp,
            'ki': ki,
            'kd': kd
        }

        self.I = 0
        self.E_ = None

    def act(self, state):
        prev_indoor = state[self.indoor_col_idx]
        P = 22 - prev_indoor
        self.I += P
        if self.E_ is None:
            self.E_ = P
        D = P - self.E_
        output = self.kp * P + self.ki * self.I + self.kd * D
        self.E_ = P
        output = min(output, 2)
        output = max(output, -2)
        return output, None
        
if __name__ == '__main__':
    from env.hsup.envs.v0 import make_env
    from train import Trainer

    env = make_env()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = ContinuousPolicyGradientAgent(state_dim, action_dim)
    trainer = Trainer(agent, env)
    trainer.fit()
