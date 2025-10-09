import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from datetime import datetime

# Assets and dates
assets = ['SPY', 'QQQ', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'DBC', 'VNQ', 'SHY']
start_date = '2015-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Download adjusted close prices only (flatten columns)
data = yf.download(assets, start=start_date, end=end_date, interval='1mo', progress=False, auto_adjust=True)

# Select only the 'Close' prices columns to avoid MultiIndex issues
data = data['Close']

# Drop rows where all prices are NaN
data.dropna(how='all', inplace=True)

print(f"Data columns: {data.columns.tolist()}")  # Should print just the 10 asset tickers

# Calculate returns
returns = data.pct_change().dropna()

def mean_variance_optimization(returns_window):
    mean_returns = returns_window.mean()
    cov_matrix = returns_window.cov()
    n = len(mean_returns)
    cov_matrix += np.eye(n) * 1e-6  # Small regularization for numerical stability
    P = cvxopt.matrix(cov_matrix.values)
    q = cvxopt.matrix(np.zeros(n))
    G = cvxopt.matrix(-np.eye(n))
    h = cvxopt.matrix(np.zeros(n))
    A = cvxopt.matrix(np.ones((1, n)))
    b = cvxopt.matrix(1.0)
    sol = cvxopt.solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    return np.array(sol['x']).flatten()

def momentum_weights(returns_window):
    momentum = returns_window.mean()
    positive = momentum.clip(lower=0)
    if positive.sum() == 0:
        return np.ones_like(momentum) / len(momentum)
    return positive / positive.sum()

lookback = 6
weights_mv, weights_mom = [], []
dates = []

for i in range(lookback, len(returns)):
    window = returns.iloc[i - lookback:i]
    dates.append(returns.index[i])
    weights_mv.append(mean_variance_optimization(window))
    weights_mom.append(momentum_weights(window))

weights_mv = pd.DataFrame(weights_mv, index=dates, columns=assets)
weights_mom = pd.DataFrame(weights_mom, index=dates, columns=assets)

strategy_returns = returns.loc[weights_mv.index]

def compute_portfolio_returns(weights, returns):
    # Use weights from previous period (shift 1)
    return (weights.shift(1) * returns).sum(axis=1)

mv_returns = compute_portfolio_returns(weights_mv, strategy_returns)
mom_returns = compute_portfolio_returns(weights_mom, strategy_returns)
ew_returns = strategy_returns.mean(axis=1)

mv_cum = (1 + mv_returns).cumprod()
mom_cum = (1 + mom_returns).cumprod()
ew_cum = (1 + ew_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(mv_cum, label='Mean-Variance')
plt.plot(mom_cum, label='Momentum')
plt.plot(ew_cum, label='Equal Weight', linestyle='--')
plt.title('Cumulative Return Comparison (2015–Now)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cumulative_returns_comparison.png")
plt.show()

def plot_allocations(latest_weights, title):
    plt.figure(figsize=(10, 4))
    latest_weights.plot(kind='bar')
    plt.title(title)
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

plot_allocations(weights_mv.iloc[-1], 'Latest Mean-Variance Allocation')
plot_allocations(weights_mom.iloc[-1], 'Latest Momentum Allocation')

def annualized_return(r):
    return (1 + r).prod() ** (12 / len(r)) - 1

def sharpe_ratio(r):
    return r.mean() / r.std() * np.sqrt(12)

def max_drawdown(cum):
    return ((cum / cum.cummax()) - 1).min()

print("Performance Metrics:\n")
print(f"Mean-Variance Annualized Return: {annualized_return(mv_returns):.2%}")
print(f"Momentum Annualized Return: {annualized_return(mom_returns):.2%}")
print(f"Equal Weight Annualized Return: {annualized_return(ew_returns):.2%}\n")

print(f"Mean-Variance Sharpe Ratio: {sharpe_ratio(mv_returns):.2f}")
print(f"Momentum Sharpe Ratio: {sharpe_ratio(mom_returns):.2f}")
print(f"Equal Weight Sharpe Ratio: {sharpe_ratio(ew_returns):.2f}\n")

print(f"Mean-Variance Max Drawdown: {max_drawdown(mv_cum):.2%}")
print(f"Momentum Max Drawdown: {max_drawdown(mom_cum):.2%}")
print(f"Equal Weight Max Drawdown: {max_drawdown(ew_cum):.2%}")
