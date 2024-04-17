import logging
import os
from os import path

import pandas as pd
import plotly.graph_objs as go
from finrl import config
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from pyfolio import timeseries

from portfolio_manager.data_loader import Yahoo_Downloader
from portfolio_manager.drl_agent import DRLAgent
from portfolio_manager.env import StockPortfolioEnv
from portfolio_manager.settings import MODEL_PARAMS_MAP, MODEL_TRAINED_MAP, TICKERS
from portfolio_manager.utils import convert_daily_return_to_pyfolio_ts, data_split

logger = logging.getLogger(__name__)

if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

def get_data(tickers, start_date, end_date):
    logger.info(f"Getting data for tickers: {tickers}")
    if not all([ticker in TICKERS for ticker in tickers]):
        raise ValueError("All tickers must be in the DOW_30_TICKER list")
    df = Yahoo_Downloader(
        ticker_list=tickers, start_date=start_date, end_date=end_date
    ).fetch_data()
    fe = FeatureEngineer(
        use_technical_indicator=True, use_turbulence=False, user_defined_feature=False
    )
    df = fe.preprocess_data(df)

    # add covariance matrix as states
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []

    # look back is one year
    lookback = 252
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback : i, :]
        price_lookback = data_lookback.pivot_table(index="date", columns="tic", values="close")
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        covs = return_lookback.cov().values
        cov_list.append(covs)

    df_cov = pd.DataFrame(
        {"date": df.date.unique()[lookback:], "cov_list": cov_list, "return_list": return_list}
    )
    df = df.merge(df_cov, on="date")
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    df = data_split(df, start_date, end_date)
    return df


def generate_environment(
    df,
    initial_amount,
    transaction_cost_pct=0,
    tech_indicator_list=["macd", "rsi_30", "cci_30", "dx_30"],
):
    stock_dimension = len(df.tic.unique())
    state_space = stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": initial_amount,
        "transaction_cost_pct": transaction_cost_pct,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-1,
    }
    e_gym = StockPortfolioEnv(df=df, **env_kwargs)
    return e_gym


def get_agent(env):
    return DRLAgent(env)


def get_model(model_name, agent, pretrained=True):
    model_params = MODEL_PARAMS_MAP[model_name]
    model = agent.get_model(model_name, model_kwargs=model_params)
    if pretrained:
        model.load(MODEL_TRAINED_MAP[model_name])
    return model


def get_prediction(models, env):
    result = {}
    for model in models:
        df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model, environment=env)
        result[model.__class__.__name__] = {
            "daily_return": df_daily_return,
            "actions": df_actions,
        }
    return result


def get_stats(predictions):
    stats = {}
    for prediction in predictions:
        daily_return = predictions[prediction]["daily_return"]
        pyfolio_ts = convert_daily_return_to_pyfolio_ts(daily_return)
        stats[prediction] = timeseries.perf_stats(
            returns=pyfolio_ts,
            factor_returns=pyfolio_ts,
            positions=None,
            transactions=None,
        )
    return stats


def get_profit(stats, env):
    profit = {}
    for prediction in stats:
        profit[prediction] = env.initial_amount * stats[prediction]["Cumulative returns"] / 100
    return profit


def get_splits(predictions, amount):
    splits = {}
    for prediction in predictions:
        action_df = predictions[prediction]["actions"]
        splits[prediction] = action_df.multiply(amount)
    return splits


def prediction_plot(predictions):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    traces = []
    for i, prediction in enumerate(predictions):
        df_daily_return = predictions[prediction]["daily_return"]
        df_cumprod = (df_daily_return.daily_return + 1).cumprod() - 1
        time_ind = pd.Series(df_daily_return.date)
        trace_portfolio = go.Scatter(
            x=time_ind, y=df_cumprod, mode="lines", name=prediction, line=dict(color=colors[i])
        )
        traces.append(trace_portfolio)
    print(traces)
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(family="sans-serif", size=15, color="black"),
            bgcolor="White",
            bordercolor="white",
            borderwidth=2,
        ),
    )
    fig.update_layout(
        title={
            "text": "Cumulative Return Time Series",
            "y": 0.85,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(1,1,0,0)",
        plot_bgcolor="rgba(1, 1, 0, 0)",
        xaxis_title="Date",
        yaxis=dict(titlefont=dict(size=30), title="Cumulative Return"),
        font=dict(
            size=40,
        ),
    )
    fig.update_layout(font_size=20)
    fig.update_traces(line=dict(width=2))

    fig.update_xaxes(
        showline=True,
        linecolor="black",
        showgrid=True,
        gridwidth=1,
        gridcolor="LightSteelBlue",
        mirror=True,
    )
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        showgrid=True,
        gridwidth=1,
        gridcolor="LightSteelBlue",
        mirror=True,
    )
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="LightSteelBlue")
    return fig
