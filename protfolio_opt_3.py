import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import os
import matplotlib.pyplot as plt
os.chdir('/Users/macbookpro/Documents/data')

mid_price = pd.read_csv("mid_price.csv", index_col=0, parse_dates=True)
#portfolio_performance = pd.DataFrame(index=mid_price.index[100:], columns=['Return'])
tickers = list(mid_price.columns)
current_value = 1.0
current_portfolio = [1.0 / 3 / mid_price[ticker].iloc[0] for ticker in mid_price.columns]

portfolio_performance =[]

for i in range(10, len(mid_price.index)):
   
    df = mid_price[(mid_price.index > mid_price.index[i-10]) & (mid_price.index <= mid_price.index[i])]
    if df.shape[0] < 1:
        continue
    mu = expected_returns.mean_historical_return(df[tickers])
    mu = 10000 * mu
    S = risk_models.sample_cov(df[tickers])
    S = 10000 ** 2 * S
    # print(mu)
    
    # print(mu)
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    weights = ef.max_quadratic_utility()  # ef.max_sharpe()

    print(weights)
    
    current_value = current_portfolio[0] * df["USDCAD"].iloc[-1] + current_portfolio[1] * df["EURUSD"].iloc[-1] + current_portfolio[2] * df["GBPUSD"].iloc[-1]
    current_portfolio = [current_value * weights["USDCAD"] / df["USDCAD"].iloc[-1], 
                         current_value * weights["EURUSD"] / df["EURUSD"].iloc[-1], 
                         current_value * weights["GBPUSD"] / df["GBPUSD"].iloc[-1]]
    portfolio_performance.append(current_value)
    if mu["EURUSD"] == 0:
        print("finished")

# Print the portfolio performance
#print(portfolio_performance)
#portfolio_performance.to_csv("portfolio_performance.csv")
plt.plot(portfolio_performance)
plt.savefig('efficient_frontier_plot.png')  # Save the figure
