from .data import train_data as data
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
import warnings

def train_val_split(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
    # train val split
    train_idx = (X.index <= '2021-12-20') | \
                (X.index >= '2022-1-1') & (y.index <= '2022-1-20') | \
                (X.index >= '2022-2-1') & (y.index <= '2022-2-20')
    val_idx =  ~train_idx

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    nona = ~X_train.isna().any(axis=1) & ~y_train.isna()
    X_train = X_train[nona]
    y_train = y_train[nona]

    nona = ~X_val.isna().any(axis=1) & ~y_val.isna()
    X_val = X_val[nona]
    y_val = y_val[nona]

    return X_train, y_train, X_val, y_val


for col in [
    'sec_supp_t',
    'sec_back_t',
    'indoor',
    'outdoor',
]:
    data[f'{col}_flow'] = data[col] * data['sec_flow']

with warnings.catch_warnings(action='ignore'):
    for col in [
        'sec_supp_t',
        'sec_supp_t_flow',
        'sec_back_t',
        "sec_heat",
        "sec_flow",
        'indoor',
        'indoor_flow',
        'outdoor',
        'outdoor_flow',
        "irradiance",
    ]:
        for i in range(1, 1+24):
            data[f'{col}_{i}0min'] = data[col].shift(i)


X_indoor_columns = [
    "sec_supp_t",
    "sec_supp_t_10min",
    "sec_supp_t_20min",
    "sec_supp_t_30min",
    "sec_back_t",
    "sec_back_t_10min",
    "sec_back_t_20min",
    "irradiance",
    "irradiance_10min",
    "indoor_10min",
    "outdoor",
]

X_sec_back_t_columns = [
    "sec_supp_t",
    "sec_supp_t_10min",
    "sec_supp_t_20min",
    "sec_back_t_10min",
    "indoor_10min",
    "indoor_20min",
    "outdoor",
    "outdoor_10min",
]

X_indoor = data[X_indoor_columns]
y_indoor = data['indoor']

nona = ~X_indoor.isna().any(axis=1) & ~y_indoor.isna()
X_indoor = X_indoor[nona]
y_indoor = y_indoor[nona]

X_sec_back_t = data[X_sec_back_t_columns]
y_sec_back_t = data['sec_back_t']

nona = ~X_sec_back_t.isna().any(axis=1) & ~y_sec_back_t.isna()
X_sec_back_t = X_sec_back_t[nona]
y_sec_back_t = y_sec_back_t[nona]


X_indoor, y_indoor, \
X_sec_back_t, y_sec_back_t, \
X_indoor_train, y_indoor_train, \
X_indoor_val, y_indoor_val, \
X_sec_back_t_train, y_sec_back_t_train, \
X_sec_back_t_val, y_sec_back_t_val = \
    map(lambda x: torch.tensor(x.values, dtype=torch.float32), 
        (X_indoor, y_indoor, X_sec_back_t, y_sec_back_t) + \
        train_val_split(X_indoor, y_indoor) + \
        train_val_split(X_sec_back_t, y_sec_back_t)
    )

indoor_train_dataset = TensorDataset(X_indoor_train, y_indoor_train)
indoor_val_dataset = TensorDataset(X_indoor_val, y_indoor_val)
sec_back_t_train_dataset = TensorDataset(X_sec_back_t_train, y_sec_back_t_train)    
sec_back_t_val_dataset = TensorDataset(X_sec_back_t_val, y_sec_back_t_val)

indoor_train_dataloader = DataLoader(indoor_train_dataset, batch_size=64, shuffle=True)
indoor_val_dataloader = DataLoader(indoor_val_dataset, batch_size=64, shuffle=False)
sec_back_t_train_dataloader = DataLoader(sec_back_t_train_dataset, batch_size=64, shuffle=True)
sec_back_t_val_dataloader = DataLoader(sec_back_t_val_dataset, batch_size=64, shuffle=False)

