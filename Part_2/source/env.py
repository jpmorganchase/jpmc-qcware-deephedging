import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp


def compute_black_scholes_deltas(
    seq_prices: jnp.ndarray,
    *,
    num_days: int = 10,
    num_trading_days: int = 30,
    mu: float = 0.0,
    sigma: float = 0.2,
    strike: float = 1.0,
) -> jnp.ndarray:
    """Computes the Black-Scholes delta for a given sequence of prices.
    Args:
        seq_prices: An array containing the price sequence for each path.
        num_days: The number of days in the trading period.
        num_trading_days: The total number of trading days (month/year).
        mu: The drift parameter in the Black-Scholes model.
        sigma: The volatility parameter in the Black-Scholes model.
        strike: The strike price of the asset.

    Returns:
        An array of deltas.
    """
    seq_prices = seq_prices[:, :-1][..., None]
    strike_price = seq_prices[0, 0] * strike
    T = jnp.arange(1, num_days + 1) / num_trading_days
    T = jnp.repeat(jnp.flip(T[None, :]), seq_prices.shape[0], 0)
    d1 = jnp.divide(
        jnp.log(seq_prices[..., 0] / strike_price) + (mu + 0.5 * sigma**2) * T,
        sigma * jnp.sqrt(T),
    )
    seq_deltas = jsp.stats.norm.cdf(d1, 0.0, 1.0)
    return seq_deltas


def compute_prices(
    seq_jumps: jnp.ndarray,
    *,
    num_trading_days: int = 30,
    mu: float = 0.0,
    sigma: float = 0.2,
    initial_price: float = 100.0,
) -> jnp.ndarray:
    """Computes the stock prices at each day given the jump sequences and other parameters.
    Args:
        seq_jumps: An array representing the jump sequence.
        num_trading_days: An integer representing the number of trading days per year.
        mu: A float representing the average rate of return.
        sigma: A float representing the standard deviation of the rate of return.
        initial_price: The initial price of the asset.

    Returns:
      An array representing the stock prices for each path and day.
    """
    bernoulli_prob = 0.5
    seq_jumps = seq_jumps - bernoulli_prob  # mean 0
    seq_jumps /= np.sqrt(bernoulli_prob * (1 - bernoulli_prob))  # std 1
    num_paths, num_days = seq_jumps.shape
    # Calculate the Brownian motion for each path and day
    brownian = jnp.cumsum(seq_jumps, axis=1)
    brownian /= np.sqrt(num_trading_days)
    t = jnp.arange(1, 1 + num_days) / num_trading_days
    # Calculate the log prices using the Black-Scholes formula
    log_prices = (mu - sigma**2 / 2) * t + sigma * brownian
    seq_prices = jnp.exp(log_prices)
    # Add the initial price
    seq_prices = jnp.concatenate([jnp.ones((num_paths, 1)), seq_prices], axis=1)
    seq_prices *= initial_price
    return seq_prices


def compute_rewards(
    seq_prices: jnp.ndarray,
    seq_deltas: jnp.ndarray,
    *,
    strike: float = 1.0,
    cost_eps: float = 0.0,
) -> jnp.ndarray:
    """
    Computes the rewards given a sequence of prices and deltas.
    Args:
        seq_prices: An array containing the price sequence for each path.
        seq_deltas: An array containing the delta sequence for each path.
        strike: The strike price of the asset.
        cost_eps: The transaction cost.

    Returns:
        seq_rewards : An array containing the rewards for each path.
    """
    # Compute the actions by taking the difference of the delta sequence
    # between each time step and the starting and ending position
    seq_actions = [
        seq_deltas[:, [0]],
        seq_deltas[:, 1:] - seq_deltas[:, :-1],
        -seq_deltas[:, [-1]],
    ]
    seq_actions = jnp.concatenate(seq_actions, axis=1)
    # Compute the payoff at the end of the path using the maximum of
    # the difference between the final price and the strike price
    payoff = -jnp.maximum(seq_prices[:, -1] - strike * seq_prices[:, 0], 0.0)
    costs = -(jnp.abs(seq_actions) * cost_eps + seq_actions) * seq_prices
    # Compute the rewards as the sum of the payoff and transaction costs at the final time step
    seq_rewards = costs.at[:, -1].add(payoff)
    return seq_rewards


def compute_bounds(
    num_days: int = 10,
    num_trading_days: int = 30,
    mu: float = 0.0,
    sigma: float = 0.2,
    initial_price: float = 100.0,
    strike: float = 1.0,
    cost_eps: float = 0.0,
) -> jnp.ndarray:
    """Computes the bounds of R(t) for a given set of parameters.
    Args:
        num_days: The number of days in the trading period.
        num_trading_days: The total number of trading days (month/year).
        mu: The drift parameter in the Black-Scholes model.
        sigma: The volatility parameter in the Black-Scholes model.
        initial_price: The initial price of the asset.
        strike: The strike price of the asset.
        cost_eps: The transaction cost.

    Returns:
        An array representing the upper and lower bounds of R(t).
    """
    jumps_max = jnp.ones((num_days))
    jumps_min = jnp.zeros((num_days))
    seq_jumps = jnp.stack([jumps_min, jumps_max], axis=0)
    prices_min, prices_max = compute_prices(
        seq_jumps,
        num_trading_days=num_trading_days,
        mu=mu,
        sigma=sigma,
        initial_price=initial_price,
    )
    payoffs_min = -jnp.maximum(prices_max - strike * initial_price, 0)
    values_max = (2 * (prices_max - strike * initial_price))[::-1][:-1]
    values_min = (2 * (prices_min - strike * initial_price) + payoffs_min)[::-1][:-1]
    Rt_range = jnp.stack((values_min, values_max), axis=0)
    return Rt_range


def compute_returns(
    seq_rewards: np.ndarray,
) -> np.ndarray:
    """Computes the sequence of cumulative returns.
    Args:
        seq_rewards: An array ontaining the rewards for each path on each day.

    Returns:
        An array containing the cumulative future returns for each path on each day.
    """
    seq_returns = jnp.cumsum(seq_rewards[:, ::-1], axis=1)[:, ::-1]
    return seq_returns


def compute_utility(
    seq_rewards: jnp.ndarray,
    *,
    utility_lambda: float = 1.0,
) -> jnp.ndarray:
    """Computes the utility of the sequence of rewards using exponential utility.
    Args:
        seq_rewards: An array containing the rewards for each path and day.
        utility_lambda: The risk aversion parameter for the exponential utility function.

    Returns:
        An array containing the utility of each path.
    """
    returns = seq_rewards.sum(axis=1)
    utility = (
        -1 / utility_lambda * jnp.log(jnp.mean(jnp.exp(-utility_lambda * returns)))
    )
    return utility
