import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns


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


S = risk_models.sample_cov(mid_price[tickers])
S = 10000*S
S = 10000*S
print(S)


current_value = 1.0
current_portfolio = [1.0/3/mid_price["USDCAD"].iloc[0], 
                     1.0/3/mid_price["EURUSD"].iloc[0], 
                     1.0/3/mid_price["GBPUSD"].iloc[0]]


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
    mu = 10000*mu
    #print(mu)
    
    #print(mu)
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    weights = ef.max_quadratic_utility() #ef.max_sharpe()
    #weights = ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
    print(weights)
    #print(ef._risk_free_rate)
    #ef.portfolio_performance(verbose=True,risk_free_rate=0.0000000002)
    current_value = current_portfolio[0]*df["USDCAD"].iloc[-1]+current_portfolio[1]*df["EURUSD"].iloc[-1]+current_portfolio[2]*df["GBPUSD"].iloc[-1]
    current_portfolio = [current_value*weights["USDCAD"]/df["USDCAD"].iloc[-1], 
                     current_value*weights["EURUSD"]/df["EURUSD"].iloc[-1], 
                     current_value*weights["GBPUSD"]/df["GBPUSD"].iloc[-1]]
    portfolio_performance.loc[period_end, 'Return'] = current_value
    if(mu["EURUSD"] == 0):
        print("finished")

# Print the portfolio performance
print(portfolio_performance)
portfolio_performance.to_csv("portfolio_performance.csv")