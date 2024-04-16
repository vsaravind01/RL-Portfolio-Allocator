from copy import deepcopy

import pandas as pd
from pyfolio import timeseries


def data_split(df, start, end, target_date_col="date"):
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)


def extract_weights(drl_actions_list):
    a2c_weight_df = {"date": [], "weights": []}
    for i in range(len(drl_actions_list)):
        date = drl_actions_list.index[i]
        tic_list = list(drl_actions_list.columns)
        weights_list = drl_actions_list.reset_index()[list(drl_actions_list.columns)].iloc[i].values
        weight_dict = {"tic": [], "weight": []}
        for j in range(len(tic_list)):
            weight_dict["tic"] += [tic_list[j]]
            weight_dict["weight"] += [weights_list[j]]

        a2c_weight_df["date"] += [date]
        a2c_weight_df["weights"] += [pd.DataFrame(weight_dict)]

    a2c_weights = pd.DataFrame(a2c_weight_df)
    return a2c_weights


def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def backtest_stats(account_value, value_col_name="account_value"):
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats_all)
    return perf_stats_all
