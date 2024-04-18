from copy import deepcopy

import pandas as pd
from pyfolio import timeseries


def data_split(df, start, end, target_date_col="date"):
    """
    Split the given DataFrame based on the specified start and end dates.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be split.
    start : str or datetime-like
        The start date for the split (inclusive).
    end : str or datetime-like
        The end date for the split (exclusive).
    target_date_col : str, optional
        The name of the column containing the dates in the DataFrame.
        Default is "date".

    Returns
    -------
    pandas.DataFrame
        The split DataFrame.

    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def convert_daily_return_to_pyfolio_ts(df):
    """
    Convert a DataFrame with daily return data to a pandas Series with a datetime index.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing daily return data.

    Returns
    -------
    pandas.Series
        Series with a datetime index representing the daily returns.
    """
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)


def extract_weights(drl_actions_list):
    """
    Extracts weights from a list of DRL actions.

    Parameters
    ----------
    drl_actions_list : pandas.DataFrame
        A DataFrame containing DRL actions.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the extracted weights.

    Notes
    -----
    This function takes a list of DRL actions and extracts the weights for each action.
    The input DataFrame should have dates as the index and tickers as the columns.
    The output DataFrame will have two columns: 'date' and 'weights'.
    The 'date' column contains the dates from the input DataFrame, and the 'weights' column
    contains a DataFrame for each date, where each row represents a ticker and its corresponding weight.

    Examples
    --------
    >>> drl_actions = pd.DataFrame({'AAPL': [0.2, 0.3, 0.5], 'GOOGL': [0.1, 0.4, 0.5]}, index=['2021-01-01', '2021-01-02', '2021-01-03'])
    >>> extract_weights(drl_actions)
                date                                            weights
    0    2021-01-01       tic  weight\n0  AAPL    0.2\n1  GOOGL    0.1
    1    2021-01-02       tic  weight\n0  AAPL    0.3\n1  GOOGL    0.4
    2    2021-01-03       tic  weight\n0  AAPL    0.5\n1  GOOGL    0.5
    """
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
    """
    Calculate the daily return of a DataFrame based on a specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    value_col_name : str, optional
        The name of the column representing the account value. Default is "account_value".

    Returns
    -------
    pandas.Series
        A Series containing the daily returns.
    """
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def backtest_stats(account_value, value_col_name="account_value"):
    """
    Calculate performance statistics for a backtest.

    Parameters
    ----------
    account_value : pandas.DataFrame
        DataFrame containing the account value over time.
    value_col_name : str, optional
        Name of the column in `account_value` DataFrame that represents the account value.
        Default is "account_value".

    Returns
    -------
    pandas.DataFrame
        DataFrame containing performance statistics for the backtest.

    """
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats_all)
    return perf_stats_all
