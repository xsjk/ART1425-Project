import random
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Optional
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import torch

from dataset.preprocess import X_indoor_columns, X_sec_back_t_columns

H = pd.Timedelta('10min')

class HeatSupplyEnv(Env):

    State = pd.Series
    Action = float      # the increase or decrease of `sec_supp_t` per hour
    Reward = float

    def __init__(
            self, 
            data: pd.DataFrame, 
            model_indoor: nn.Module, 
            model_sec_back_t: nn.Module,
            start_time: Optional[pd.Timestamp] = None,
            discretize: bool = False,
        ) -> None:
            self.data = data.copy()
            if start_time is None:
                self.start_time = None
            else:
                self.start_time = pd.Timestamp(start_time).round('10min')  # Round start_time to the nearest hour
            self.end_time = self.data.index[-1].floor('10min')
            self.model_indoor = model_indoor
            self.model_sec_back_t = model_sec_back_t


            self.X_indoor_cols: list[str] = X_indoor_columns
            self.X_sec_back_t_cols: list[str] = X_sec_back_t_columns
            self.X_cols = list(set(self.X_indoor_cols) | set(self.X_sec_back_t_cols) | {'indoor'})
            self.observation_cols = list((set(self.X_indoor_cols) | set(self.X_sec_back_t_cols)) - {'sec_back_t', 'sec_supp_t'})
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(self.observation_cols),))
            if discretize:
                self.action_space = Discrete(5, start=-2)
            else:
                self.action_space = Box(low=-2, high=2, shape=(1,))
            self.reset()

    def reset(self, seed=None, *args, **kwargs) -> tuple[np.ndarray, dict]:
        self.X = self.data.resample('10min').mean().dropna()
        if self.start_time is None:
            self.T = random.choice(self.X.index[:-2])
        else:
            self.T = self.start_time
            
        self.end_time = self.data.index[-1].floor("10min")
        
        self.X = self.X.loc[self.T:, self.X_cols]
        self.X.loc[self.T + 0 * H:, ['indoor', 'sec_back_t', 'sec_supp_t']] = np.nan
        self.X.loc[self.T + 1 * H:, ['indoor_10min', 'sec_back_t_10min', 'sec_supp_t_10min']] = np.nan
        self.X.loc[self.T + 2 * H:, ['indoor_20min', 'sec_back_t_20min', 'sec_supp_t_20min']] = np.nan
        self.X.loc[self.T + 3 * H:, ['indoor_30min', 'sec_back_t_30min', 'sec_supp_t_30min']] = np.nan
        self.S: pd.Series = self.X.loc[self.T, self.observation_cols]
        return self.S.values.astype(np.float32), dict()

    def step(self, A: Action) -> tuple[np.ndarray, Reward, bool, bool, dict]:
        # Apply the action to the environment
        # Return the next state, reward, done (whether the episode is finished), and additional info
        self.S = next_state = self._get_next_state(A)
        self.T += H
        reward = self._get_reward(next_state)
        done = not max(0, next_state['outdoor_10min']) < next_state['sec_supp_t_10min'] < 100
        truncated = self.T == self.end_time
        assert self.S.name == self.T
        return next_state.values.astype(np.float32), reward, done, truncated, {}

    def render(self) -> None:
        print(self.X.loc[self.T, self.observation_cols])

    def close(self) -> None:
        print('Close')

    def plot(self) -> None:
        self.X.dropna()[['indoor', 'outdoor', 'sec_supp_t', 'sec_back_t']].plot(figsize=(15, 5), grid=True)

    #############################################################

    def _get_next_state(self, A: Action) -> State:
        
        if abs(A) > 2:
            print(f'Action must be between -2 and 2, but got {A}')
        T: pd.Timestamp = self.T
        T_ = T + H
        T__ = T_ + H
        T___ = T__ + H
        T____ = T___ + H

        self.X.loc[T, 'sec_supp_t'] = \
        self.X.loc[T_,'sec_supp_t_10min'] = \
        self.X.loc[T__,'sec_supp_t_20min'] = \
        self.X.loc[T___,'sec_supp_t_30min'] = \
            self.X.loc[T, 'sec_supp_t_10min'] + A
        
        x = self.X.loc[T, self.X_sec_back_t_cols].to_frame().T
        with torch.no_grad():
            y = self.model_sec_back_t(
                torch.tensor(
                    x.values, 
                    dtype=torch.float32, 
                    device=self.model_sec_back_t.device)
                )[0].cpu().numpy()

        self.X.loc[T, 'sec_back_t'] = \
        self.X.loc[T_, 'sec_back_t_10min'] = \
        self.X.loc[T__, 'sec_back_t_20min'] = \
        self.X.loc[T___, 'sec_back_t_30min'] = y
        
        x = self.X.loc[T, self.X_indoor_cols].to_frame().T
        with torch.no_grad():
            y = self.model_indoor(
                torch.tensor(
                    x.values, 
                    dtype=torch.float32,
                    device=self.model_indoor.device)
                )[0].cpu().numpy()

        self.X.loc[T, 'indoor'] = \
        self.X.loc[T_, 'indoor_10min'] = \
        self.X.loc[T__, 'indoor_20min'] = \
        self.X.loc[T___, 'indoor_30min'] = y

        return self.X.loc[T_, self.observation_cols]

    def _get_reward(self, S: State) -> Reward:
        return 4 - (S['indoor_10min'] - 22) ** 2 - max((S['sec_supp_t_10min'] - 40) ** 2 / 100 - 1, 0)


