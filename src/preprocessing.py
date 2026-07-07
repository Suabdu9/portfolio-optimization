import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale_prices(df):

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        df[["Adj Close"]]
    )

    return scaler, scaled


def create_sequences(data, window=60):

    X = []
    y = []

    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])

    return np.array(X), np.array(y)
