# Time Series Forecasting for Portfolio Management Optimization

## Overview

This project was completed as part of the 10 Academy Week 11 challenge.

The objective is to develop time series forecasting models and use their predictions to optimize an investment portfolio for **Guide Me in Finance (GMF) Investments**.

The project analyzes three financial assets:

- **TSLA** – Tesla Inc.
- **SPY** – SPDR S&P 500 ETF
- **BND** – Vanguard Total Bond Market ETF

Historical market data is obtained using the Yahoo Finance API (`yfinance`) covering the period:

**January 1, 2015 – June 30, 2026**

---

## Business Objective

GMF Investments is a personalized portfolio management firm that uses data-driven investment strategies to maximize client returns while managing portfolio risk.

This project develops forecasting models to support portfolio allocation decisions by:

- Forecasting Tesla stock prices
- Comparing statistical and deep learning forecasting methods
- Constructing an optimal investment portfolio
- Backtesting the proposed strategy against a benchmark portfolio

---

## Project Structure

```
portfolio-optimization/
│
├── .github/
│   └── workflows/
│       └── unittests.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_data_preprocessing_eda.ipynb
│   ├── 02_time_series_forecasting.ipynb
│   ├── 03_future_forecasting.ipynb
│   ├── 04_portfolio_optimization.ipynb
│   └── 05_backtesting.ipynb
│
├── src/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python 3.12
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- pmdarima
- TensorFlow / Keras
- PyPortfolioOpt
- yfinance

---

## Installation

Clone the repository

```bash
git clone <repository-url>
cd portfolio-optimization
```

Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the notebooks in the following order:

1. Data Preprocessing & EDA
2. Time Series Forecasting
3. Future Forecasting
4. Portfolio Optimization
5. Strategy Backtesting

---

## Forecasting Models

### ARIMA

- Automatic parameter selection using Auto-ARIMA
- Statistical baseline model

### LSTM

- Deep learning sequence model
- 60-day input window
- Two LSTM layers with Dropout

The best-performing model is used for future forecasting.

---

## Portfolio Optimization

Modern Portfolio Theory (MPT) is applied to determine the optimal allocation among:

- Tesla (TSLA)
- Vanguard Bond ETF (BND)
- SPY ETF

The Efficient Frontier is generated and the Maximum Sharpe Ratio portfolio is selected.

---

## Backtesting

The optimized portfolio is evaluated against a benchmark portfolio:

- 60% SPY
- 40% BND

Performance metrics include:

- Total Return
- Annualized Return
- Sharpe Ratio
- Maximum Drawdown

---

## Author

Sumeya Abdulsemed