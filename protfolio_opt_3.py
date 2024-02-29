import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import os
import matplotlib.pyplot as plt
from glob import glob

forecast_file=glob('/Users/macbookpro/Documents/data/aws_data/20s/*.csv')

dates=list(set([i.split('_')[1].split('.')[0] for i in forecast_file]))
tickers=list(set([i.split('_')[0] for i in forecast_file]))
forecast_file={i:[j for j in forecast_file if i in j] for i in tickers}
os.chdir('/Users/macbookpro/Documents/data/')
file_names=glob('*.zip')

data_file=[i for i in file_names if i.split('_')[2] in dates and i.split('_')[-1].split('.')[0] in tickers]
data_file={i:[j for j in data_file if i in j] for i in tickers}

data={i:pd.DataFrame() for i in tickers}

for i in data_file.keys():
    for j in data_file[i][:3]:
        data[i]=pd.concat([data[i], pd.read_csv(j, index_col=0, parse_dates=True)], axis=1)

forecast_data={i:pd.DataFrame() for i in tickers}
for i in forecast_file:
    for j in tickers:
        if j in i:
            forecast_data[j]=pd.concat([forecast_data[j], pd.read_csv(i, index_col=0, parse_dates=True)], axis=1)

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
