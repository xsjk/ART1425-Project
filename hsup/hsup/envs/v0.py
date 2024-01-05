import random
import numpy as np
import pandas as pd
from typing import Optional
import gymnasium as gym
from gymnasium.spaces import Box
from sklearn.linear_model import LinearRegression

H = pd.Timedelta("1H")


class HeatSupplyEnvV0(gym.Env):
    State = pd.Series
    Action = float  # the increase or decrease of `sec_supp_t` per hour
    Reward = float

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        data: pd.DataFrame,
        model_indoor: LinearRegression,
        model_sec_back_t: LinearRegression,
        start_time: Optional[pd.Timestamp] = None,
    ) -> None:
        self.data = data.copy()
        if start_time is None:
            self.start_time = None
        else:
            self.start_time = pd.Timestamp(start_time).round(
                "H"
            )  # Round start_time to the nearest hour
        self.render_mode = "human"

        self.model_indoor = model_indoor
        self.model_sec_back_t = model_sec_back_t

        self.X_indoor_cols: list[str] = self.model_indoor.feature_names_in_
        self.X_sec_back_t_cols: list[str] = self.model_sec_back_t.feature_names_in_
        self.X_cols = list(
            set(self.X_indoor_cols) | set(self.X_sec_back_t_cols) | {"indoor"}
        )
        self.observation_cols = list(
            (set(self.X_indoor_cols) | set(self.X_sec_back_t_cols))
            - {"sec_back_t", "sec_supp_t"}
        )

        self.action_space = Box(low=-2, high=2, shape=(1,))
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(len(self.observation_cols),)
        )

    def reset(self, seed=None, *args, **kwargs) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.X = self.data.resample("H").mean().dropna()
        if self.start_time is None:
            self.T = random.choice(self.X.index[:-2])
        else:
            self.T = self.start_time

        self.end_time = min(self.T + 24 * H, self.data.index[-1].floor("H"))

        self.X = self.X.loc[self.T :, self.X_cols]
        self.X.loc[self.T + 0 * H :, ["indoor", "sec_back_t", "sec_supp_t"]] = np.nan
        self.X.loc[
            self.T + 1 * H :, ["indoor_60min", "sec_back_t_60min", "sec_supp_t_60min"]
        ] = np.nan
        self.X.loc[
            self.T + 2 * H :,
            ["indoor_120min", "sec_back_t_120min", "sec_supp_t_120min"],
        ] = np.nan
        self.X.loc[
            self.T + 3 * H :,
            ["indoor_180min", "sec_back_t_180min", "sec_supp_t_180min"],
        ] = np.nan

        self.S: pd.Series = self.X.loc[self.T, self.observation_cols]
        return self.S.values.astype(np.float32), {}

    def step(self, A: Action) -> tuple[np.ndarray, Reward, bool, bool, dict]:
        # Apply the action to the environment
        # Return the next state, reward, done (whether the episode is finished), and additional info
        self.S = next_state = self._get_next_state(A)
        self.T += H

        reward = self._get_reward(next_state)
        truncate = self.T >= self.end_time
        done = (
            not max(0, next_state["outdoor_60min"])
            < next_state["sec_supp_t_60min"]
            < 100
        ) or truncate

        assert self.S.name == self.T

        return next_state.values.astype(np.float32), reward, done, truncate, {}

    def render(self) -> None:
        # print(self.X.loc[self.T, self.observation_cols])
        self.X.dropna()[["indoor", "outdoor", "sec_supp_t", "sec_back_t"]].plot(
            figsize=(15, 5), grid=True
        )

    def close(self) -> None:
        print("Close")

    def _get_next_state(self, A: Action) -> State:
        if abs(A) > 2:
            print(f"Action must be between -2 and 2, but got {A}")
        T: pd.Timestamp = self.T
        T_ = T + H
        T__ = T_ + H
        T___ = T__ + H
        T____ = T___ + H

        self.X.loc[T, "sec_supp_t"] = self.X.loc[T_, "sec_supp_t_60min"] = self.X.loc[
            T__, "sec_supp_t_120min"
        ] = self.X.loc[T___, "sec_supp_t_180min"] = (
            self.X.loc[T, "sec_supp_t_60min"] + A
        )

        x = self.X.loc[T, self.X_sec_back_t_cols].to_frame().T
        y = self.model_sec_back_t.predict(x)[0]

        self.X.loc[T, "sec_back_t"] = self.X.loc[T_, "sec_back_t_60min"] = self.X.loc[
            T__, "sec_back_t_120min"
        ] = self.X.loc[T___, "sec_back_t_180min"] = (
            self.X.loc[T, "sec_back_t_60min"] + y
        )

        x = self.X.loc[T, self.X_indoor_cols].to_frame().T
        y = self.model_indoor.predict(x)[0]

        self.X.loc[T, "indoor"] = self.X.loc[T_, "indoor_60min"] = self.X.loc[
            T__, "indoor_120min"
        ] = self.X.loc[T___, "indoor_180min"] = (self.X.loc[T, "indoor_60min"] + y)

        return self.X.loc[T_, self.observation_cols]

    def _get_reward(self, S: State) -> Reward:
        return (
            4
            - (S["indoor_60min"] - 22) ** 2
            - max((S["sec_supp_t_60min"] - 40) ** 2 / 100 - 1, 0)
        )
