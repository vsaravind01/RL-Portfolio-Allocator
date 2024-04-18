import logging

from finrl import config
from finrl.agents.stablebaselines3.models import TensorboardCallback
from stable_baselines3 import A2C, DDPG, PPO, SAC

logger = logging.getLogger(__name__)

MODELS = {"a2c": A2C, "ddpg": DDPG, "sac": SAC, "ppo": PPO}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
        tensorboard_log=None,
    ):
        """Get the model with the corresponding algorithm.

        Parameters
        ----------
        model_name: str
            Name of the model to use
        policy: str
            The policy model to use (MlpPolicy, CnnPolicy, ...)
        policy_kwargs: dict
            Additional arguments to be passed to the policy on creation
        model_kwargs: dict
            Additional arguments to be passed to the model on creation
        verbose: int
            The verbosity level: 0 none, 1 training information, 2 tensorflow debug
        seed: int
            Seed for the pseudo-random generators
        tensorboard_log: str
            the log location for tensorboard

        Returns
        -------
        model
            the respective model
        """
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        print(model_kwargs)
        logger.info(f"model: {model_name} | {model_kwargs}")
        return MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    @staticmethod
    def train_model(model, tb_log_name, total_timesteps=5000):
        """Train the model

        Parameters
        ----------
        model: model
            the model to be trained
        tb_log_name: str
            the log name for tensorboard
        total_timesteps: int
            the total timesteps to train the model (default is 5000)

        Returns
        -------
        model
            the trained model
        """
        logger.info(f"Start training model - {model}")
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(verbose=1),
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        """make a prediction and get results

        Parameters
        ----------
        model: model
            the trained model
        environment: gym environment class
            the testing environment
        deterministic: bool
            whether the prediction should be deterministic or not

        Returns
        -------
        account_memory: list
            the account value for each day
        actions_memory: list
            the action for each day
        """
        test_env, test_obs = environment.get_sb_env()
        account_memory = None  # This help avoid unnecessary list creation
        actions_memory = None  # optimize memory consumption

        test_env.reset()
        max_steps = len(environment.df.index.unique()) - 1

        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, dones, info = test_env.step(action)

            if i == max_steps - 1:
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            # add current state to state memory
            # state_memory=test_env.env_method(method_name="save_state_memory")

            if dones[0]:
                print("hit end!")
                break
        assert account_memory is not None, "account_memory is None"
        assert actions_memory is not None, "actions_memory is None"
        return account_memory[0], actions_memory[0]

    @staticmethod
    def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic=True):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except BaseException as error:
            raise ValueError(f"Failed to load agent. Error: {str(error)}") from error

        # test on the testing env
        state = environment.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [environment.initial_total_asset]
        done = False
        while not done:
            action = model.predict(state, deterministic=deterministic)[0]
            state, reward, done, _ = environment.step(action)

            total_asset = (
                environment.amount
                + (environment.price_ary[environment.day] * environment.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.initial_total_asset
            episode_returns.append(episode_return)

        print("episode_return", episode_return)
        print("Test Finished!")
        return episode_total_assets
