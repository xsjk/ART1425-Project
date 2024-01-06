import pandas as pd

def load(filename: str):
    data = (
        pd.merge(
            pd.read_excel(filename, sheet_name="Sheet1", index_col=0, parse_dates=True),
            pd.read_excel(filename, sheet_name="Sheet2", index_col=0, parse_dates=True),
            left_index=True,
            right_index=True,
            how="outer",
        )
        .interpolate("quadratic")
    )
    data["irradiance"].clip(lower=0, inplace=True)
    data["sec_heat"] = (data["sec_supp_t"] - data["sec_back_t"]) * data["sec_flow"]
    return data.iloc[1:-5]

train_data = load(__file__[:-7]+'train.xlsx')
test_data = load(__file__[:-7]+'test.xlsx')
