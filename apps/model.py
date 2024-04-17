import datetime

import coloredlogs
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib import colors

from portfolio_manager.engine import (
    generate_environment,
    get_agent,
    get_data,
    get_model,
    get_prediction,
    get_profit,
    get_splits,
    get_stats,
    prediction_plot,
)
from portfolio_manager.settings import MODEL_TRAINED_MAP, TICKERS

coloredlogs.install(level="DEBUG")


def app():

    @st.cache_data
    def load_data(tickers, start_date, end_date):
        return get_data(tickers, start_date, end_date)

    side_bar = st.sidebar
    main_page = st

    config_set = False

    with side_bar.form("model_settings_form"):
        st.subheader("Model Settings")
        tickers = st.multiselect("Select tickers", TICKERS, default=["AAPL", "MSFT", "IBM"])
        date_range = st.date_input(
            "Select date range", (datetime.date(2009, 1, 1), datetime.date(2023, 12, 31))
        )
        models = st.multiselect("Select models", list(MODEL_TRAINED_MAP.keys()), default=["ppo"])
        initail_amount = st.number_input(
            "Initial amount", value=1000000, step=10000, min_value=10000
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            config_set = True

    if config_set:
        start_date = date_range[0].strftime("%Y-%m-%d")  # type: ignore
        end_date = date_range[1].strftime("%Y-%m-%d")  # type: ignore
        data = load_data(tickers, start_date, end_date)
        environment = generate_environment(data, initail_amount)
        agent = get_agent(environment)
        model_objects = [get_model(model, agent) for model in models]
        predictions = get_prediction(model_objects, environment)
        splits = get_splits(predictions, initail_amount)
        stats = get_stats(predictions)
        profilts = get_profit(stats, environment)

        plot = prediction_plot(predictions)

        main_page.subheader("Cumulative Returns Plot")
        main_page.plotly_chart(plot, use_container_width=True)

        main_page.subheader("Profits")
        # set color for lowest and highest profit as green and red
        min_profit = min(profilts.values())
        max_profit = max(profilts.values())
        colors = [
            "#31c571" if profit == max_profit else "#da4856" if profit == min_profit else "#5996fc"
            for profit in profilts.values()
        ]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(profilts.keys()),
                    y=list(profilts.values()),
                    marker_color=colors,
                    text=list(f"${profit:,.2f}" for profit in profilts.values()),
                )
            ]
        )
        fig.update_traces(textfont_size=18)
        main_page.plotly_chart(fig, use_container_width=True)

        main_page.subheader("Statistics")
        stats_df = pd.DataFrame(stats)
        filter_columns = [
            "Annual return",
            "Cumulative returns",
            "Max drawdown",
            "Sharpe ratio",
            "Annual volatility",
        ]
        main_page.dataframe(stats_df.loc[filter_columns].T, use_container_width=True)

        main_page.subheader("Allocations (actions)")
        for split in splits:
            main_page.write(f"Model: {split}")
            main_page.dataframe(splits[split], use_container_width=True)
    else:
        main_page.markdown(
            "<h3 style='height: 30vh; display: flex; justify-content: center; text-align: center; color: grey;'>Please set model settings</h3>",
            unsafe_allow_html=True,
        )

    config_set = False
