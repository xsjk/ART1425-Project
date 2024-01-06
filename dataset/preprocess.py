from .data import train_data as data
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import warnings

def drop_na(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_na = X.isna().any(axis=1)
    y_na = y.isna().any(axis=1) if len(y.shape) > 1 else y.isna()

    nona = ~(x_na | y_na)
    return X[nona], y[nona]

def train_val_split(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
    # train val split
    train_idx = (X.index <= '2021-12-20') | \
                (X.index >= '2022-1-1') & (y.index <= '2022-1-20') | \
                (X.index >= '2022-2-1') & (y.index <= '2022-2-20')
    val_idx =  ~train_idx

    X_train, y_train = drop_na(X[train_idx], y[train_idx])
    X_val, y_val = drop_na(X[val_idx], y[val_idx])

    return X_train, y_train, X_val, y_val

def to_tensor(*args):
    return map(lambda x: torch.tensor(x.values, dtype=torch.float32), args)

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
    "sec_back_t",
    "sec_back_t_10min",
    "indoor_10min", 

    "sec_supp_t",
    "sec_supp_t_10min",       
    "irradiance",
    "irradiance_10min",
    "outdoor",
    "outdoor_10min",    
]

X_sec_back_t_columns = [
    "sec_back_t_10min",
    "indoor_10min",

    "sec_supp_t",
    "sec_supp_t_10min",    
    "irradiance",
    "irradiance_10min",    
    "outdoor",
    "outdoor_10min",
]

X_indoor, y_indoor = drop_na(data[X_indoor_columns], data['indoor'])
X_sec_back_t, y_sec_back_t = drop_na(data[X_sec_back_t_columns], data['sec_back_t'])

X_indoor_train, y_indoor_train, X_indoor_val, y_indoor_val = to_tensor(*train_val_split(X_indoor, y_indoor))
X_sec_back_t_train, y_sec_back_t_train, X_sec_back_t_val, y_sec_back_t_val = to_tensor(*train_val_split(X_sec_back_t, y_sec_back_t))

X_indoor, y_indoor = to_tensor(X_indoor, y_indoor)
X_sec_back_t, y_sec_back_t = to_tensor(X_sec_back_t, y_sec_back_t)

indoor_train_dataset = TensorDataset(X_indoor_train, y_indoor_train)
indoor_val_dataset = TensorDataset(X_indoor_val, y_indoor_val)
sec_back_t_train_dataset = TensorDataset(X_sec_back_t_train, y_sec_back_t_train)    
sec_back_t_val_dataset = TensorDataset(X_sec_back_t_val, y_sec_back_t_val)

indoor_train_dataloader = DataLoader(indoor_train_dataset, batch_size=64, shuffle=True)
indoor_val_dataloader = DataLoader(indoor_val_dataset, batch_size=64, shuffle=False)
sec_back_t_train_dataloader = DataLoader(sec_back_t_train_dataset, batch_size=64, shuffle=True)
sec_back_t_val_dataloader = DataLoader(sec_back_t_val_dataset, batch_size=64, shuffle=False)




X_branch_columns = [
    "sec_back_t_10min",
    "indoor_10min",

    "sec_supp_t",
    "sec_supp_t_10min",    
    "irradiance",
    "irradiance_10min",    
    "outdoor",
    "outdoor_10min",
]

X_branch, y_branch = drop_na(data[X_branch_columns], data[['sec_back_t', 'indoor']])

X_branch_train, y_branch_train, X_branch_val, y_branch_val = to_tensor(*train_val_split(X_branch, y_branch))

X_branch, y_branch = to_tensor(X_branch, y_branch)

branch_train_dataset = TensorDataset(X_branch_train, y_branch_train)
branch_val_dataset = TensorDataset(X_branch_val, y_branch_val)

branch_train_dataloader = DataLoader(branch_train_dataset, batch_size=64, shuffle=True)
branch_val_dataloader = DataLoader(branch_val_dataset, batch_size=64, shuffle=False)
