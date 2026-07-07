import numpy as np
import pandas as pd

from pypfopt import EfficientFrontier
from pypfopt import risk_models


def calculate_historical_returns(prices):
    """
    Calculate daily percentage returns.
    """
    return prices.pct_change().dropna()


def calculate_expected_returns(
    forecast_prices,
    historical_returns,
):
    """
    Expected returns:
    TSLA -> forecast
    SPY/BND -> historical annualized mean
    """

    tsla_return = (
        forecast_prices.iloc[-1] /
        forecast_prices.iloc[0]
        - 1
    )

    spy_return = historical_returns["SPY"].mean() * 252
    bnd_return = historical_returns["BND"].mean() * 252

    expected_returns = pd.Series({
        "TSLA": tsla_return,
        "SPY": spy_return,
        "BND": bnd_return
    })

    return expected_returns


def covariance_matrix(returns):
    """
    Annualized covariance matrix.
    """

    return returns.cov() * 252


def optimize_max_sharpe(expected_returns, covariance):
    ef = EfficientFrontier(expected_returns, covariance)

    weights = ef.max_sharpe()

    cleaned = ef.clean_weights()

    performance = ef.portfolio_performance(verbose=False)

    return cleaned, performance


def optimize_min_volatility(expected_returns, covariance):
    ef = EfficientFrontier(expected_returns, covariance)

    weights = ef.min_volatility()

    cleaned = ef.clean_weights()

    performance = ef.portfolio_performance(verbose=False)

    return cleaned, performance
