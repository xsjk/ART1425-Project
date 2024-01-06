import numpy as np
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from rich.progress import track
import os
from itertools import count
from collections import defaultdict
from argparse import ArgumentParser

class Trainer:
    def __init__(
            self, 
            agent: 'BaseAgent', 
            env: 'HeatSupplyEnvironment', *, 
            device = "auto",
            log_every_n_steps: int = 1,
            save_interval: int = 10,
            max_epochs = 1000,
            min_epochs = 0,
            epsilon = 0.25,
            epsilon_decay = 0.95,
            epsilon_min = 0.01
        ):
            
        self.agent = agent
        self.env = env
        self.save_interval = save_interval
        self.log_every_n_steps = log_every_n_steps
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        if device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.agent = self.agent.to(self.device)

    def fit(self):
        
        writer = SummaryWriter()

        with open(f'{writer.get_logdir()}/hparams.yaml', 'w') as f:
            yaml.dump(self.agent.hparams, f)

        os.mkdir(f'{writer.get_logdir()}/checkpoints')

        self.agent.setup()

        for episode in track(range(self.max_epochs)):
            state, info = self.env.reset()
            episode_reward = 0
            metric_lists = defaultdict(lambda:[])
            for step in count():
                if np.random.rand() < self.epsilon:
                    action, log_prob = self.agent.act(state)
                    action = action.detach().cpu().numpy()
                else:
                    action = self.env.action_space.sample()
                    log_prob = torch.tensor([np.log(1/4)], device=self.device, requires_grad=True, dtype=torch.float32)

                next_state, reward, done, truncated, info = self.env.step(action)
                metrics = self.agent.training_step((state, action, log_prob, next_state, reward, done, truncated, info), step)
                state = next_state
                episode_reward += reward
                if metrics:
                    for k, v in metrics.items():
                        metric_lists[k].append(v)
                if done or truncated:
                    break
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if metric_lists:
                metrics = {k: np.mean(v) for k, v in metric_lists.items()}
                metrics['Reward/Episode'] = episode_reward

                if episode % self.log_every_n_steps == 0:
                    for k, v in metrics.items():
                        writer.add_scalar(k, v, episode)
                    print(f'Episode: {episode}\t Reward: {episode_reward:.2f}')

                if episode % self.save_interval == 0:
                    torch.save({
                        'hparams': self.agent.hparams,
                        'epoch': episode,
                        'model_state_dict': self.agent.state_dict(),
                        'metrics': metrics
                    }, f'{writer.get_logdir()}/checkpoints/{episode:03d}-{episode_reward:.2f}.pth')
                    pass

    def test(self, exit_on_done=True):
        state, info = self.env.reset()
        episode_reward = 0
        rewards = []
        for step in count():
            action, _ = self.agent.act(state)
            state, reward, done, truncated, info = self.env.step(action)
            rewards.append(reward)
            episode_reward += reward
            if truncated:
                break
            if done and exit_on_done:
                break
        self.env.plot()
        return rewards


def config_parser(
    parser: ArgumentParser = ArgumentParser(),
    targets: list[str] = None,
) -> ArgumentParser:

    parser.add_argument("agent", type=str, default="DDPGAgent", choices=targets, help="the type of agent to train")
    subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand', required=True)

    new_parser = subparsers.add_parser('new', help='Train a new model')
    resume_parser = subparsers.add_parser('resume', help='Resume training a model')

    resume_parser = resume_parser.add_argument_group("Resume")
    resume_parser.add_argument("checkpoint_path", type=str, default=None)
    
    for p in (new_parser, resume_parser):
        p.add_argument("--min-epochs", type=int, default=100, help="Minimum number of epochs")
        p.add_argument("--max-epochs", type=int, default=-1, help="Maximum number of epochs")

    return parser


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':

    from agent import BaseAgent, DDPGAgent, ContinuousPolicyGradientAgent
    from env import make_env, HeatSupplyEnvironment

    fix_seed(42)

    agent_map = {
        'DDPGAgent': DDPGAgent,
        'ContinuousPolicyGradientAgent': ContinuousPolicyGradientAgent
    }
    parser = config_parser(targets=[
        'DDPGAgent',
        'ContinuousPolicyGradientAgent'
    ])
    args = parser.parse_args()
    agent_type = agent_map[args.agent]
    env = make_env()

    match args.subcommand:
        case 'new':
            agent = agent_type(
                state_dim=env.observation_space.shape[0], 
                action_dim=env.action_space.shape[0], 
            )
            trainer = Trainer(agent, env)
            trainer.fit()
            pass
        case 'resume':
            print('resume')
            agent = agent_type.load_from_checkpoint(args.checkpoint_path)
            trainer = Trainer(agent, env)
            trainer.fit()
            pass