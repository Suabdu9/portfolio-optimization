from pathlib import Path

import numpy as np
import pandas as pd


def create_last_sequence(prices_scaled, window_size=60):
    """
    Returns the last window of scaled prices to initialize
    multi-step forecasting.
    """

    if len(prices_scaled) < window_size:
        raise ValueError(
            f"Need at least {window_size} observations."
        )

    return prices_scaled[-window_size:].reshape(
        1,
        window_size,
        1
    )


def iterative_forecast(
    model,
    last_sequence,
    scaler,
    days=252,
):
    """
    Forecast future prices using recursive prediction.

    Parameters
    ----------
    model : keras model

    last_sequence : ndarray
        Shape = (1, window_size, 1)

    scaler : sklearn scaler

    days : int
        Number of trading days.

    Returns
    -------
    ndarray
        Forecasted prices.
    """

    current_sequence = last_sequence.copy()

    forecasts = []

    for _ in range(days):

        next_price_scaled = model.predict(
            current_sequence,
            verbose=0
        )[0][0]

        forecasts.append(next_price_scaled)

        current_sequence = np.append(
            current_sequence[:, 1:, :],
            [[[next_price_scaled]]],
            axis=1
        )

    forecasts = np.array(forecasts).reshape(-1, 1)

    forecasts = scaler.inverse_transform(forecasts)

    return forecasts.flatten()


def build_forecast_dataframe(
    last_date,
    forecasts,
):
    """
    Create a forecast dataframe.
    """

    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(forecasts)
    )

    forecast_df = pd.DataFrame({

        "Date": future_dates,

        "Forecast": forecasts

    })

    return forecast_df


def add_confidence_interval(
    forecast_df,
    residual_std,
):
    """
    Approximate 95% confidence interval.
    """

    horizon = np.arange(
        1,
        len(forecast_df) + 1
    )

    interval = 1.96 * residual_std * np.sqrt(horizon)

    forecast_df["Lower_CI"] = (
        forecast_df["Forecast"] - interval
    )

    forecast_df["Upper_CI"] = (
        forecast_df["Forecast"] + interval
    )

    return forecast_df


def save_forecast(
    forecast_df,
    filename="tsla_future_forecast.csv",
):

    output = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
        / filename
    )

    forecast_df.to_csv(
        output,
        index=False
    )

    print(f"Forecast saved to:\n{output}")
