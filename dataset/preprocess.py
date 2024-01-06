from .data import train_data as data
import warnings

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
    "sec_supp_t_60min",
    "sec_supp_t_120min",
    "sec_supp_t_180min",
    # "sec_supp_t_flow_60min",
    # "sec_heat_60min",
    # "sec_flow",
    # "sec_flow_60_min",
    "sec_back_t",
    "sec_back_t_60min",
    "sec_back_t_120min",
    # "indoor_flow_60min",
    "irradiance",
    "irradiance_60min",
    "irradiance_120min",
    # "indoor_10min",
    "indoor_60min",
    # "indoor_70min",
    # "indoor_80min",
    # "indoor_90min",
    # "indoor_100min",
    # "indoor_110min",
    "indoor_120min",
    # "diff_t_60min",
    "outdoor",
    "outdoor_60min",
    "outdoor_120min",
]

X_sec_back_t_columns = [
    "sec_supp_t",
    "sec_supp_t_60min",
    "sec_supp_t_120min",
    "sec_supp_t_180min",
    # "sec_supp_t_flow_60min",
    # "sec_heat_60min",
    # "sec_flow",
    # "sec_flow_60_min",
    # "sec_back_t",
    "sec_back_t_60min",
    "sec_back_t_120min",
    "sec_back_t_180min",
    # "indoor_flow_60min",
    "irradiance",
    "irradiance_60min",
    "irradiance_120min",
    # "irradiance_180min",
    # "indoor_10min",
    "indoor_60min",
    "indoor_120min",
    "indoor_180min",
    # "diff_t_60min",
    "outdoor",
    "outdoor_60min",
    "outdoor_120min",
]


X_indoor = data[X_indoor_columns]
X_sec_back_t = data[X_sec_back_t_columns]
y_indoor = data['indoor']
y_sec_back_t = data['sec_back_t']

X_indoor = X_indoor.resample("1H").mean()
X_sec_back_t = X_sec_back_t.resample("1H").mean()
y_indoor = y_indoor.resample("1H").mean()
y_sec_back_t = y_sec_back_t.resample("1H").mean()
