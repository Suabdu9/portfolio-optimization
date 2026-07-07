import numpy as np


def cumulative_returns(returns):

    return (1 + returns).cumprod()


def portfolio_returns(returns, weights):

    return (returns * weights).sum(axis=1)


def annual_return(r):

    return (1+r.mean())**252 - 1


def sharpe_ratio(r):

    return np.sqrt(252) * r.mean() / r.std()


def maximum_drawdown(cumulative):

    rolling = cumulative.cummax()

    drawdown = cumulative / rolling - 1

    return drawdown.min()
