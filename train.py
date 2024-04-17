import sys
from os import path

from portfolio_manager.engine import (
    generate_environment,
    get_agent,
    get_data,
    get_model,
)
from portfolio_manager.settings import MODEL_PARAMS_MAP, TICKERS


def train(model_name, start_date, end_date, initial_amount=1000000):
    data = get_data(TICKERS, start_date, end_date)
    if model_name not in MODEL_PARAMS_MAP:
        raise ValueError(
            f"Model name {model_name} not supported. Supported models: {MODEL_PARAMS_MAP.keys()}"
        )

    environment = generate_environment(data, initial_amount)
    env_train, _ = environment.get_sb_env()
    agent = get_agent(env_train)
    model = get_model(model_name, agent, pretrained=False)
    agent.train_model(model=model, tb_log_name=model_name, total_timesteps=40000)

    model.save(
        path.join(
            path.dirname(__file__),
            f"portfolio_manager/trained_models/{model_name}_dow30_{start_date}_{end_date}",
        )
    )


if __name__ == "__main__":
    model_name = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    if len(sys.argv) > 4:
        initial_amount = int(sys.argv[4])
    else:
        initial_amount = 1000000
    train(model_name, start_date, end_date, initial_amount)
