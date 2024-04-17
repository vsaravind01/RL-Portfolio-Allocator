import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


class StockPortfolioEnv(gym.Env):
    """A portfolio allocation environment for OpenAI gym

    Attributes
    ----------
    df : DataFrame
        input data
    stock_dim : int
        number of unique stocks
    hmax : int
        maximum number of shares to trade
    initial_amount : int
        start money
    transaction_cost_pct : float
        transaction cost percentage per trade
    reward_scaling : float
        scaling factor for reward, good for training
    state_space : int
        the dimension of input features
    action_space : int
        equals stock dimension
    tech_indicator_list : list
        a list of technical indicator names
    turbulence_threshold : int, optional
        a threshold to control risk aversion
    lookback : int, optional
        number of historical data points to consider
    day : int, optional
        an increment number to control date

    Methods
    -------
    step(actions)
        At each step, the agent will return actions, then calculate the reward, and return the next observation.
    reset()
        Reset the environment
    render(mode="human")
        Use render to return other functions
    save_asset_memory()
        Return account value at each time step
    save_action_memory()
        Return actions/positions at each time step
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        day=0,
    ):
        """
        Initialize the StockPortfolioEnv.

        Parameters
        ----------
        df : DataFrame
            Input data
        stock_dim : int
            Number of unique stocks
        hmax : int
            Maximum number of shares to trade
        initial_amount : int
            Start money
        transaction_cost_pct : float
            Transaction cost percentage per trade
        reward_scaling : float
            Scaling factor for reward, good for training
        state_space : int
            The dimension of input features
        action_space : int
            Equals stock dimension
        tech_indicator_list : list
            A list of technical indicator names
        turbulence_threshold : int, optional
            A threshold to control risk aversion
        lookback : int, optional
            Number of historical data points to consider
        day : int, optional
            An increment number to control date
        """
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space + len(self.tech_indicator_list), self.state_space),
        )

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        """
        At each step, the agent will return actions, then calculate the reward, and return the next observation.

        Parameters
        ----------
        actions : array-like
            The actions to be taken by the agent

        Returns
        -------
        state : array-like
            The next observation/state
        reward : float
            The reward for the current step
        terminal : bool
            Whether the episode is terminated or not
        info : dict
            Additional information
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            # Plot cumulative reward and save the figure
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]
            plt.plot(df.daily_return.cumsum(), "r")
            plt.savefig("results/cumulative_reward.png")
            plt.close()

            # Plot rewards and save the figure
            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig("results/rewards.png")
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                print("Sharpe: ", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            weights = self.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data["cov_list"].values[0]
            self.state = np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            )
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights
            )
            log_portfolio_return = np.log(
                sum((self.data.close.values / last_day_memory.close.values) * weights)
            )
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        """
        Reset the environment.

        Returns
        -------
        state : array-like
            The initial state/observation
        """
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # load states
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state

    def render(self, mode="human"):
        """
        Use render to return other functions.

        Parameters
        ----------
        mode : str, optional
            The rendering mode

        Returns
        -------
        state : array-like
            The current state/observation
        """
        return self.state

    def softmax_normalization(self, actions):
        """
        Apply softmax normalization to the actions.

        Parameters
        ----------
        actions : array-like
            The actions to be normalized

        Returns
        -------
        softmax_output : array-like
            The normalized actions
        """
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        """
        Return account value at each time step.

        Returns
        -------
        df_account_value : DataFrame
            Account value at each time step
        """
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({"date": date_list, "daily_return": portfolio_return})
        return df_account_value

    def save_action_memory(self):
        """
        Return actions/positions at each time step.

        Returns
        -------
        df_actions : DataFrame
            Actions/positions at each time step
        """
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.set_index(df_date["date"], inplace=True)
        return df_actions

    def _seed(self, seed=None):
        """
        Set the seed for the random number generator.

        Parameters
        ----------
        seed : int, optional
            The seed value

        Returns
        -------
        list
            The seed value
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        """
        Get the Stable Baselines environment.

        Returns
        -------
        e : DummyVecEnv
            The Stable Baselines environment
        obs : array-like
            The initial observation/state
        """
        e = DummyVecEnv([lambda: self])  # type: ignore
        obs = e.reset()
        return e, obs
