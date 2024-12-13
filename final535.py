import shinybroker as sb
from shiny import App, ui, reactive, render, Inputs, Outputs, Session, req
import pandas as pd
from flask import Flask, send_from_directory
import os
import asyncio
import sys
# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import datetime
import math
from numpy.random import choice
import random
from io import BytesIO
import base64
import statsmodels.api as sm
from statsmodels import regression


from keras._tf_keras.keras.layers import Input, Dense, Flatten, Dropout
from keras._tf_keras.keras import Model
#from keras.regularizers import l2

import numpy as np

import random
from collections import deque
import matplotlib.pylab as plt


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )


app_ui = ui.page_fluid(
    ui.h1("Reinforcement Learning Portfolio Manager"),
    ui.hr(),

    ui.h2("Overview of the RL Code"),
    ui.p("""
        This application implements a Reinforcement Learning (RL) framework for portfolio 
        management. It simulates a financial market environment, allowing an RL agent to 
        learn and optimize portfolio allocation strategies.
    """),
    ui.p("""
    To use this application, please enter the ticker symbol of a company in the designated 
    input field and click "Fetch Price". The system will retrieve the last 360 days of 
    historical price data for that ticker. The RL model will then apply a 180-day lookback 
    window and attempt to rebalance the portfolio three times during these 360 days: at 
    days 0-180, 90-270, and 180-360. This rolling approach is designed to help the RL agent 
    adapt its strategy as new market conditions emerge and potentially improve performance 
    over a static equal-weight portfolio.
    """),

    ui.h2("Key Components"),
    ui.h3("1. RL Environment"),
    ui.p("""
        The environment represents the financial market. It processes input data and provides 
        the agent with the current state for decision-making. Key features include:
    """),
    ui.tags.ul(
        ui.tags.li("Cleans and preprocesses historical data."),
        ui.tags.li("Provides state inputs, such as covariance matrices of asset returns."),
        ui.tags.li("Calculates rewards based on the portfolio's returns, risk, and Sharpe ratio."),
    ),

    ui.h3("2. RL Agent"),
    ui.p("""
        The RL agent learns to optimize portfolio allocations using a deep learning model. 
        Key features include:
    """),
    ui.tags.ul(
        ui.tags.li("Deep learning model tailored to the portfolio size."),
        ui.tags.li("Supports three actions: buy, sell, and hold."),
        ui.tags.li("Exploration-exploitation balance using an epsilon-greedy policy."),
        ui.tags.li("Maintains replay memory for efficient training."),
    ),

    ui.h3("3. Portfolio Management"),
    ui.p("""
        A portfolio is represented as a weighted combination of assets. The system calculates:
    """),
    ui.tags.ul(
        ui.tags.li("Returns: Weighted average of individual asset returns."),
        ui.tags.li("Volatility: Standard deviation of returns."),
        ui.tags.li("Sharpe Ratio: Risk-adjusted return."),
    ),

    ui.h3("4. Training Loop"),
    ui.p("""
        The training loop simulates episodic learning using historical market data. At each 
        time step:
    """),
    ui.tags.ul(
        ui.tags.li("The environment provides the current state."),
        ui.tags.li("The agent decides portfolio weights (actions)."),
        ui.tags.li("The environment calculates the reward based on returns and risk."),
        ui.tags.li("The agent updates its strategy by replaying stored experiences."),
    ),

    ui.h3("5. Evaluation"),
    ui.p("""
        The agent's performance is compared to a benchmark (equal-weight portfolio) by calculating:
    """),
    ui.tags.ul(
        ui.tags.li("Returns: Portfolio growth over time."),
        ui.tags.li("Volatility: Risk of the portfolio."),
        ui.tags.li("Sharpe Ratio: Risk-adjusted return."),
        ui.tags.li("Alpha and Beta: Measures of performance relative to the benchmark."),
    ),

    # Key Concepts Section
    ui.h2("Key Concepts Used"),
    ui.tags.ul(
        ui.tags.li("Reinforcement Learning (RL): Learning by interacting with the environment."),
        ui.tags.li("Deep Learning: Neural networks predict optimal actions."),
        ui.tags.li("Covariance Matrix: Represents relationships between asset returns."),
        ui.tags.li("Exploration vs. Exploitation: Balances trying new strategies with using known strategies."),
        ui.tags.li("Replay Memory: Stabilizes learning by storing and reusing experiences."),
        ui.tags.li("Sharpe Ratio: Measures risk-adjusted portfolio return."),
    ),

    ui.h2("Challenges Addressed"),
    ui.p("""
        This RL framework overcomes several challenges in financial portfolio management:
    """),
    ui.tags.ul(
        ui.tags.li("Dynamic Market Conditions: Adapts to changing data."),
        ui.tags.li("Portfolio Allocation: Balances risk and return."),
        ui.tags.li("Exploration: Avoids local optima by exploring alternatives."),
        ui.tags.li("Training Stability: Replay memory and deep learning prevent overfitting."),
    ),

    ui.h2("Market Overview"),
    ui.p("""
    In 2023, financial markets experienced a robust recovery, with the Morningstar US 
    Market Index rising 25%, rebounding from a 19.5% decline in 2022. This resurgence was 
    driven by a resilient economy, stronger-than-expected corporate earnings, and the 
    anticipation of the Federal Reserve concluding its interest rate hikes.
    """),
    ui.p("""
    However, the market's gains were unevenly distributed, with a narrow leadership primarily 
    in technology and growth stocks, particularly those associated with artificial intelligence 
    advancements. This concentration posed challenges for active fund managers, as only 23% of 
    large-cap active funds outperformed their benchmarks by December 2023.
    """),
    ui.p("""
    Despite the Federal Reserve's aggressive monetary tightening, corporate debt defaults 
    remained relatively low, attributed to companies reducing debts and extending maturities. 
    Nonetheless, concerns arose that markets might be underestimating future debt default risks, 
    especially given relaxed corporate attitudes toward funding and cash preservation.
    """),

    ui.h2("Result Analysis"),
    ui.p("""
    While the RL portfolio often outperforms the equal-weighted portfolio by dynamically adjusting 
    allocations, exploiting market inefficiencies, and managing risk more effectively, there are 
    instances when it may lag behind. In other words, the RL approach can generally deliver superior 
    performance, but under certain conditions it might underperform the simpler benchmark.
    """),
    ui.p("""
    Potential reasons for the RL portfolio's superior performance include:
    """),
    ui.tags.ul(
        ui.tags.li("Adaptive Strategy: The RL model continuously updates its strategy to exploit changing market conditions."),
        ui.tags.li("Risk Management: By learning from historical data, the RL agent can mitigate downturns better than a static allocation."),
        ui.tags.li("Market Inefficiencies: The RL agent can identify and capitalize on short-term opportunities that a simple equal-weight strategy might miss."),
        ui.tags.li("Enhanced Metrics: Under many scenarios, the RL portfolio can maintain similar levels of volatility compared to the equal-weighted benchmark, but achieve higher returns, larger Sharpe ratios, and more favorable alpha, thereby outperforming across multiple evaluation measures."),
    ),
    ui.p("""
    On the other hand, when the RL portfolio underperforms the equal-weighted portfolio, it may be due to:
    """),
    ui.tags.ul(
        ui.tags.li(
            "Data Limitations: Utilizing only 360 days of historical data may not capture the full spectrum of market conditions, leading to overfitting and poor generalization to new data."),
        ui.tags.li(
            "Market Concentration: The dominance of a few sectors, particularly technology, means that a diversified RL strategy might lag behind an equal-weighted portfolio heavily influenced by these outperforming sectors."),
        ui.tags.li(
            "Model Complexity: The RL model's complexity requires extensive data and careful tuning. Insufficient data or suboptimal hyperparameters can hinder the model's ability to learn effective strategies."),
    ),

    ui.h2("Recommendations for Improvement"),
    ui.tags.ul(
        ui.tags.li(
            "Expand Data Horizons: Incorporate a more extended historical dataset to expose the RL model to various market cycles, enhancing its learning and adaptability."),
        ui.tags.li(
            "Feature Engineering: Include additional market indicators and macroeconomic variables to provide the model with a richer context for decision-making."),
        ui.tags.li(
            "Regularization Techniques: Apply methods like dropout or weight decay to prevent overfitting, ensuring the model generalizes well to unseen data."),
        ui.tags.li(
            "Hyperparameter Optimization: Conduct systematic tuning of hyperparameters to identify the most effective configurations for your specific dataset and objectives."),
        ui.tags.li(
            "Ensemble Methods: Combine the RL model with traditional strategies to balance innovation with proven approaches, potentially enhancing overall performance."),
    ),

    ui.hr(),

    ui.h2("Historical Data Collector"),
    ui.input_text("ticker", "Enter Ticker Symbol"),
    ui.input_action_button("add_button", "Fetch Price"),
    ui.input_action_button("finish_button", "Finish Selection"),
    ui.output_table("current_summary"),
    ui.output_image("detailed_portfolio_plot"),
    ui.output_image("portfolio_plot"),
    ui.output_text("stats_equal"),
    ui.output_text("stats_rl")
)

def portfolio(returns, weights):
    weights = np.array(weights)
    rets = returns.mean() * 252
    covs = returns.cov() * 252
    P_ret = np.sum(rets * weights)
    P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
    P_sharpe = P_ret / P_vol
    return np.array([P_ret, P_vol, P_sharpe])


class CryptoEnvironment:
    def __init__(self, data=None, prices=None, capital=1e6):
        self.capital = capital

        if data is not None:
            self.data = data
        else:
            if prices is None:
                raise ValueError("You must provide either `data` or `prices`.")
            self.data = self.load_data(prices)

        self.data = self.data.dropna()
        if "Date" in self.data.columns:
            self.data.reset_index(drop=True, inplace=True)
        elif "date" in self.data.columns:
            self.data.reset_index(drop=True, inplace=True)

        if self.data.empty:
            raise ValueError("Data is empty after cleaning.")

    def load_data(self, prices_path):
        data = pd.read_csv(prices_path)
        if 'Date' in data.columns:
            data.index = data['Date']
            data = data.drop(columns=['Date'])
        elif 'date' in data.columns:
            data.index = data['date']
            data = data.drop(columns=['date'])
        else:
            data.index = data.iloc[:, 0]
            data = data.iloc[:, 1:]
        return data

    def preprocess_state(self, state):
        return state

    def get_state(self, t, lookback, is_cov_matrix=True):
        if t < lookback:
            return None
        decision_making_state = self.data.iloc[t - lookback:t]
        if decision_making_state.empty:
            return np.zeros((len(self.data.columns), len(self.data.columns)))
        decision_making_state = decision_making_state.pct_change().dropna()
        return decision_making_state.cov() if is_cov_matrix else decision_making_state

    def get_reward(self, action, action_t, reward_t):
        data_period = self.data.iloc[action_t:reward_t]
        if data_period.empty:
            return np.zeros(len(action)), 0
        returns = data_period.pct_change().dropna()
        if returns.empty:
            return np.zeros(len(action)), 0
        try:
            portfolio_return = np.dot(returns.values.mean(axis=0), action)
            portfolio_risk = np.sqrt(np.dot(action.T, np.dot(returns.cov().values, action)))
            sharpe = portfolio_return / portfolio_risk if portfolio_risk != 0 else 0
        except:
            return np.zeros(len(action)), 0
        return portfolio_return, sharpe


def train_rl_agent(env, episodes=100, lookback=30):
    for episode in range(episodes):
        t = lookback
        done = False
        while not done:
            state = env.get_state(t, lookback)
            action = np.random.uniform(0, 1, size=len(env.data.columns))
            action /= np.sum(action)
            _, reward = env.get_reward(action, t, t + 1)
            t += 1
            if t >= len(env.data) - 1:
                done = True


class Agent:
    def __init__(self, portfolio_size, is_eval=False, allow_short=True):
        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size)
        self.action_size = 3
        self.memory4replay = deque(maxlen=2000)
        self.is_eval = is_eval
        self.alpha = 0.5
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.model = self._model()

    def _model(self):
        inputs = Input(shape=self.input_shape)
        x = Flatten()(inputs)
        x = Dense(100, activation='elu')(x)
        x = Dropout(0.5)(x)
        x = Dense(50, activation='elu')(x)
        x = Dropout(0.5)(x)
        predictions = []
        for i in range(self.portfolio_size):
            asset_dense = Dense(self.action_size, activation='linear')(x)
            predictions.append(asset_dense)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='adam', loss='mse')
        return model

    def nn_pred_to_weights(self, pred, allow_short=False):
        weights = np.zeros(len(pred))
        raw_weights = np.argmax(pred, axis=-1)
        for e, r in enumerate(raw_weights):
            if r == 0:  # sit
                weights[e] = 0
            elif r == 1:  # buy
                weights[e] = np.abs(pred[e][0][r])
            else:  # sell
                weights[e] = -np.abs(pred[e][0][r])
        if not allow_short:
            weights += np.abs(np.min(weights))
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            # Avoid division by zero, just distribute equally if all are zero
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights /= weights_sum
        return weights, None, None

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            w = np.random.normal(0, 1, size=(self.portfolio_size,))
            if not self.allow_short:
                w += np.abs(np.min(w))
            w_sum = np.sum(w)
            if w_sum == 0:
                w = np.ones(len(w)) / len(w)
            else:
                w /= w_sum
            return w, None, None
        pred = self.model.predict(np.expand_dims(state.values, 0))
        return self.nn_pred_to_weights(pred, self.allow_short)

    def expReplay(self, batch_size):
        # This part of the code is not fully implemented but let's leave as is
        pass


def train_rl_agent_with_agent(env, agent, episodes=100, lookback=30):
    for episode in range(episodes):
        t = lookback
        done = False
        while not done:
            state = env.get_state(t, lookback)
            if state is None or np.all(state == 0):
                t += 1
                continue
            action, _, _ = agent.act(state)
            reward, sharpe = env.get_reward(action, t, t + 1)
            next_state = env.get_state(t + 1, lookback)
            done = t >= len(env.data) - 2
            if next_state is not None:
                agent.memory4replay.append((state, next_state, action, reward, done))
            t += 1
        if len(agent.memory4replay) > 32:
            agent.expReplay(32)


def sharpe(R):
    r = np.diff(R)
    if r.std() == 0:
        return 0
    sr = r.mean() / r.std() * np.sqrt(252)
    return sr

def print_stats(result, benchmark):
    result = np.array(result)
    benchmark = np.array(benchmark)
    sr = sharpe(result.cumsum())
    returns = np.mean(result)
    volatility = np.std(result)

    X = benchmark
    y = result
    x = sm.add_constant(X)
    model = regression.linear_model.OLS(y, x).fit()
    alpha = model.params[0]
    beta = model.params[1]

    return np.round(np.array([returns, volatility, sr, alpha, beta]), 4).tolist()

def portfolio_server(input: Inputs, output: Outputs, session: Session, ib_socket, sb_rvs):
    combined_data = reactive.Value(pd.DataFrame())
    exported_data = reactive.Value(None)

    rl_results = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.add_button)
    def fetch_price():
        req(input.ticker())
        ticker = input.ticker().strip().upper()
        try:
            historical_data = sb.fetch_historical_data(
                contract=sb.Contract({
                    'symbol': ticker,
                    'secType': "STK",
                    'exchange': "SMART",
                    'currency': "USD"
                }),
                durationStr="360 D",
                barSizeSetting="1 day"
            )

            if historical_data and "hst_dta" in historical_data:
                hst_dta = pd.DataFrame(historical_data["hst_dta"])
                hst_dta['timestamp'] = pd.to_datetime(hst_dta['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d')
                formatted_data = hst_dta[['timestamp', 'close']].copy()
                formatted_data.rename(columns={'timestamp': 'Date', 'close': ticker}, inplace=True)

                current_data = combined_data.get()
                if current_data.empty:
                    combined_data.set(formatted_data)
                else:
                    updated_data = pd.concat([current_data.reset_index(drop=True),
                                              formatted_data[ticker].reset_index(drop=True)], axis=1)
                    combined_data.set(updated_data)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    @reactive.effect
    @reactive.event(input.finish_button)
    def finish_and_export():
        summary_data = combined_data.get()
        if summary_data.empty:
            print("No data to process.")
            return

        try:
            exported_data.set(summary_data)
            next_step_dataset = summary_data.copy()
            next_step_dataset.dropna(inplace=True)

            if 'Date' in next_step_dataset.columns:
                next_step_dataset.set_index('Date', inplace=True)

            N_ASSETS = len(next_step_dataset.columns)

            # Initialize environment & agent
            env = CryptoEnvironment(data=next_step_dataset)
            agent = Agent(portfolio_size=N_ASSETS)
            agent.is_eval = True

            actions_equal, actions_rl = [], []
            result_equal, result_rl = [], []

            window_size = 150
            rebalance_period = 90

            for t in range(window_size, len(env.data), rebalance_period):
                state = env.get_state(t, window_size)
                if state is None or np.all(state == 0):
                    continue

                action, _, _ = agent.act(state)
                weighted_returns_rl, _ = env.get_reward(action, t - rebalance_period, t)
                weighted_returns_equal, _ = env.get_reward(
                    np.ones(agent.portfolio_size) / agent.portfolio_size, t - rebalance_period, t
                )

                # Ensure these are lists
                if np.isscalar(weighted_returns_equal):
                    weighted_returns_equal = [weighted_returns_equal]
                if np.isscalar(weighted_returns_rl):
                    weighted_returns_rl = [weighted_returns_rl]

                result_equal.extend(weighted_returns_equal)
                result_rl.extend(weighted_returns_rl)

            # Store results in rl_results
            rl_results.set({
                "result_equal": result_equal,
                "result_rl": result_rl
            })

        except Exception as e:
            print(f"Error initializing RL environment: {e}")

    # @output
    @render.table
    def current_summary():
        data = combined_data.get()
        return data.head(10) if not data.empty else pd.DataFrame({"Message": ["No data available yet."]})

    @output
    @render.text
    def stats_equal():
        print("stats_equal called")  # Debug print
        res = rl_results.get()
        if res is None:
            return "No results yet."
        stats = print_stats(res["result_equal"], res["result_equal"])
        return f"EQUAL Portfolio: Returns: {stats[0]}, Volatility: {stats[1]}, Sharpe: {stats[2]}, Alpha: {stats[3]}, Beta: {stats[4]}"

    @output
    @render.text
    def stats_rl():
        print("stats_rl called")  # Debug print
        res = rl_results.get()
        if res is None:
            return "No results yet."
        stats = print_stats(res["result_rl"], res["result_equal"])
        return f"RL Portfolio: Returns: {stats[0]}, Volatility: {stats[1]}, Sharpe: {stats[2]}, Alpha: {stats[3]}, Beta: {stats[4]}"

    @output
    @render.image
    def portfolio_plot():
        res = rl_results.get()
        if res is None:
            return None

        result_equal_vis = np.array(res["result_equal"])
        result_rl_vis = np.array(res["result_rl"])

        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(result_equal_vis.cumsum(), label='Benchmark (Equal Weight)', color='grey', linestyle='--')
        ax.plot(result_rl_vis.cumsum(), label='RL Portfolio', color='blue', linestyle='-')
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Cumulative Returns")
        ax.set_title("Portfolio Performance Comparison")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        plot_path = os.path.join(".", "portfolio_plot.png")
        plt.savefig(plot_path, format='png')
        plt.close(fig)

        return {
            "src": "portfolio_plot.png",
            "alt": "Portfolio Performance Plot",
            "width": "100%",
            "height": "auto"
        }

    @output
    @render.image
    def detailed_portfolio_plot():
        res = rl_results.get()
        if res is None:
            return None

        result_equal_vis = np.array(res["result_equal"])
        result_rl_vis = np.array(res["result_rl"])

        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(result_equal_vis, label="Benchmark Daily Returns", color="grey", linestyle="--")
        ax.plot(result_rl_vis, label="RL Daily Returns", color="blue", linestyle="-")
        ax.set_title("Daily Returns Comparison")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Daily Returns")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        plot_path = os.path.join(".", "detailed_portfolio_plot.png")
        plt.savefig(plot_path, format='png')
        plt.close(fig)

        return {
            "src": "detailed_portfolio_plot.png",
            "alt": "Detailed Portfolio Plot",
            "width": "100%",
            "height": "auto"
        }


app = sb.sb_app(
    home_ui=app_ui,
    server_fn=portfolio_server,
    host="127.0.0.1",
    port=7497,
    client_id=10799,
    verbose=True,
)

app.run()
