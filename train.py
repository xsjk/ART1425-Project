import numpy as np
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from rich.progress import track
from env import make_env, HeatSupplyEnvironment
from agent import DDPGAgent
import os
from itertools import count

class Trainer:
    def __init__(
            self, 
            agent: DDPGAgent, 
            env: HeatSupplyEnvironment, *, 
            device = "auto",
            log_every_n_steps: int = 1,
            save_interval: int = 10,
            max_epochs = 1000,
            min_epochs = 0,
            epsilon = 0.25,
            epsilon_decay = 0.95):
            
        self.agent = agent
        self.env = env
        self.save_interval = save_interval
        self.log_every_n_steps = log_every_n_steps
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        if device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.agent = self.agent.to(self.device)
        self.env.device = self.device

    def fit(self):
        
        writer = SummaryWriter()

        with open(f'{writer.get_logdir()}/hparams.yaml', 'w') as f:
            yaml.dump(self.agent.hparams, f)

        os.mkdir(f'{writer.get_logdir()}/checkpoints')

        episodes = 1000
        for episode in track(range(episodes)):
            state, info = self.env.reset()
            episode_reward = 0
            critic_losses = []
            policy_losses = []
            for step in count():
                if np.random.rand() < self.epsilon:
                    action = self.agent(state).detach()
                else:
                    action = torch.tensor(self.env.action_space.sample(), dtype=torch.float32, device=self.device)
                next_state, reward, done, truncated, info = self.env.step(action.item())
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                metrics = self.agent.update()  # sample from replay buffer and update the network
                state = next_state
                episode_reward += reward
                if metrics:
                    critic_losses.append(metrics['critic_loss'])
                    policy_losses.append(metrics['policy_loss'])
                if done or truncated:
                    break
            
            self.epsilon *= self.epsilon_decay

            if critic_losses and policy_losses:
                avg_critic_loss = np.mean(critic_losses)
                avg_policy_loss = np.mean(policy_losses)

                if episode % self.log_every_n_steps == 0:
                    writer.add_scalar('Loss/Critic', avg_critic_loss, episode)
                    writer.add_scalar('Loss/Policy', avg_policy_loss, episode)
                    writer.add_scalar('Reward/Episode', episode_reward, episode)
                    print(f'Episode: {episode}\t Reward: {episode_reward:.2f}\t Average Critic Loss: {avg_critic_loss:.2f}\t Average Policy Loss: {avg_policy_loss:.2f}')

                if episode % self.save_interval == 0:
                    torch.save({
                        'epoch': episode,
                        'model_state_dict': self.agent.state_dict(),
                        'metrics': {
                            'Loss/Critic': avg_critic_loss,
                            'Loss/Policy': avg_policy_loss,
                            'Reward/Episode': episode_reward
                        }
                    }, f'{writer.get_logdir()}/checkpoints/{episode}-{episode_reward}.pth')
                    pass

    def test(self, exit_on_done=True):
        state, info = self.env.reset()
        episode_reward = 0
        rewards = []
        for step in count():
            action = self.agent(state)
            state, reward, done, truncated, info = self.env.step(action.item())
            rewards.append(rewards)
            episode_reward += reward
            if truncated:
                break
            if done and exit_on_done:
                break
        X = self.env.X.dropna()
        X[['indoor', 'outdoor', 'sec_supp_t', 'sec_back_t']].plot(figsize=(15, 5), grid=True)


if __name__ == '__main__':
    env = make_env()
    agent = DDPGAgent(
        state_dim=env.observation_space.shape[0], 
        action_dim=env.action_space.shape[0], 
    )
    trainer = Trainer(agent, env)
    trainer.fit()
