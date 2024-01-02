import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: tuple[int]):
        super(Actor, self).__init__()
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
        super(Critic, self).__init__()
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
        self.state = torch.empty((max_size, state_dim), dtype=torch.float32, device=device)
        self.action = torch.empty((max_size, action_dim), dtype=torch.float32, device=device)
        self.next_state = torch.empty((max_size, state_dim), dtype=torch.float32, device=device)
        self.reward = torch.empty(max_size, dtype=torch.float32, device=device)
        self.done = torch.empty(max_size, dtype=torch.float32, device=device)

    def push(self, state, action, reward, next_state, done) -> None:
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
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
class DDPGAgent(nn.Module):
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            hidden_dim: tuple[int]=(128,), 
            memory_size: int=10000,
            lr_actor: float=1e-6,
            lr_critic: float=1e-6,
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
        return action

    def to(self, device) -> 'DDPGAgent':
        r = super().to(device)
        r.replay_buffer = r.replay_buffer.to(device)
        r.device = device
        return r

    def update(self) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        # state = torch.stack(state)
        # action = torch.stack(action)
        # reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        # next_state = torch.stack(next_state)
        # done = torch.tensor(done, dtype=torch.float32, device=self.device)

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
            'critic_loss': critic_loss.item(), 
            'policy_loss':policy_loss.item()
        }

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, hparams_file):
        agent = cls(**yaml.load(open(hparams_file), yaml.Loader))
        agent.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
        return agent

