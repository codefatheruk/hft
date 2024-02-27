import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import os

os.chdir('/Users/macbookpro/Documents/data')

mid_price = pd.read_csv("mid_price.csv")
#mid_price = mid_price.set_index("time")            
#mid_price = mid_price.resample("1T").last()
#mid_price = mid_price.ffill()


# Define the tickers and the backtest period
tickers = ["USDCAD", "EURUSD", "GBPUSD"]
start_date = mid_price.iloc[0]["Date"]
end_date = mid_price.iloc[-1]["Date"]

# Define the rebalancing frequency (e.g., '1M' for monthly)c
rebalance_freq = '5min'

# Generate a date range for rebalancing
date_range = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
portfolio_performance = pd.DataFrame(index=date_range, columns=['Return'])

# Now try with a nonconvex objective from  Kolm et al (2014)
def deviation_risk_parity(w, cov_matrix):
    diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
    return (diff**2).sum().sum()

for i in range(len(date_range)-1):
    if(date_range[i].weekday() > 4):
        continue
    # Define the period for the current rebalance
    period_start = date_range[i].strftime('%Y-%m-%d %H:%M:%S')
    period_end = date_range[i+1].strftime('%Y-%m-%d %H:%M:%S')
    
    #print(period_start, period_end)

    df = mid_price[(mid_price['Date'] > period_start) & (mid_price['Date'] <= period_end)]
    if(df.shape[0] < 1):
        continue
    mu = expected_returns.mean_historical_return(df[tickers])
    S = risk_models.sample_cov(df[tickers])
    #print(mu)
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    weights = ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
    ef.portfolio_performance(verbose=True)
    portfolio_performance.loc[period_end, 'Return'] = np.sum(list(weights.values()))
    if(mu["EURUSD"] == 0):
        print("finished")

# Print the portfolio performance
print(portfolio_performance)
portfolio_performance.to_csv("portfolio_performance.csv")