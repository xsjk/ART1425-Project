import pandas as pd

data = (
    pd.merge(
        pd.read_excel("dataset/train.xlsx", sheet_name="Sheet1", index_col=0, parse_dates=True),
        pd.read_excel("dataset/train.xlsx", sheet_name="Sheet2", index_col=0, parse_dates=True),
        left_index=True,
        right_index=True,
        how="outer",
    )
    .interpolate("quadratic")
    .dropna()
)
data["irradiance"].clip(lower=0, inplace=True)
data["sec_heat"] = (data["sec_supp_t"] - data["sec_back_t"]) * data["sec_flow"]

