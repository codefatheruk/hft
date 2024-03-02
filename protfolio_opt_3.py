import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import os
import matplotlib.pyplot as plt
from glob import glob

df=pd.read_csv("C:/Users/rouho/Documents/GitHub/aws/60s_prediction.csv",index_col=0,parse_dates=True)
tickers=list(set(i.split('_')[0] for i in df.columns))
names=list(set(i.split('_')[1] for i in df.columns))
main_df={i:pd.DataFrame() for i in names}
for i in names:
    c_name=[j for j in df.columns if i in j]
    temp_df=df[c_name]
    main_df[i]=pd.concat([main_df[i],temp_df],axis=1)
    main_df[i].columns=[j.split('_')[0] for j in main_df[i].columns]
    del temp_df

expected_price=(main_df['pos']*0.0001)-(main_df['neg']*0.0001)+main_df['price']


current_portfolio = [1.0 / 3 / main_df['price'][i].iloc[0] for i in tickers]

portfolio_performance =[]

for i in range(50, len(main_df['price'].index)):

    df = expected_price.iloc[i-50:i]
    df2 = main_df['price'].iloc[i-50:i]
    if df.shape[0] < 1:
        continue
    mu = pd.Series( main_df['ExcpectReturn'].iloc[i].values,index=[0,1,2])
    S = risk_models.sample_cov(df[tickers])
    S = 10000 ** 2 * S
    # print(mu)

    # print(mu)
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    weights =ef.max_quadratic_utility()  
    #weights =ef.max_sharpe(risk_free_rate=-10)  

    print(weights)

    current_value = current_portfolio[0] * df2[tickers[0]].iloc[-1] + current_portfolio[1] * df2[tickers[1]].iloc[-1] + current_portfolio[2] * df2[tickers[2]].iloc[-1]
    current_portfolio = [current_value * weights[0] / df2[tickers[0]].iloc[-1],
                         current_value * weights[1] / df2[tickers[1]].iloc[-1],
                         current_value * weights[2] / df2[tickers[2]].iloc[-1]]
    portfolio_performance.append(current_value)

plt.plot(portfolio_performance)
plt.show()
plt.savefig('efficient_frontier_plot2.png')  # Save the figure


