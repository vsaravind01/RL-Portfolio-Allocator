from os import path

TICKERS = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
    "DOW",
]

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.001,
    "batch_size": 128,
}

A2C_PARAMS = {"n_steps": 10, "ent_coef": 0.005, "learning_rate": 0.0004}

DDPG_PARAMS = {
    "buffer_size": 10_000,
    "learning_rate": 0.0005,
    "batch_size": 64,
}

SAC_PARAMS = {
    "buffer_size": 10_000,
    "learning_rate": 0.0005,
    "batch_size": 64,
}

MODEL_PARAMS_MAP = {
    "ppo": PPO_PARAMS,
    "a2c": A2C_PARAMS,
    "ddpg": DDPG_PARAMS,
    "sac": SAC_PARAMS,
}


MODEL_TRAINED_MAP = {
    "ppo": path.join(path.dirname(__file__), "trained_models/ppo_dow30_2010-01-01_2023-12-31"),
    "a2c": path.join(path.dirname(__file__), "trained_models/a2c_dow30_2010-01-01_2023-12-31"),
    "ddpg": path.join(path.dirname(__file__), "trained_models/ddpg_dow30_2010-01-01_2023-12-31"),
    "sac": path.join(path.dirname(__file__), "trained_models/sac_dow30_2010-01-01_2023-12-31"),
}
