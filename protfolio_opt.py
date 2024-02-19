import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import os
from glob import glob
import itertools

os.chdir('C:/Users/rouho/Desktop/hedge fund/ligh_gbm_functions/paid_data/')
file_names=glob('*.zip')
currenies=list(set([i.split('_')[-1].split('.')[0] for i in file_names]))

file_currency={i:[] for i in currenies}
for i in file_names:
    for j in file_currency.keys():
        if j in i:
            file_currency[j].append(i)

column_names = [['Date']] \
                  + [[f'BP{i}' , f'BV{i}'] for i in range(1, 6)] \
                  + ['M'] \
                  + [[f'AP{i}' , f'AV{i}'] for i in range(1, 6)] \
                  + [['']]
column_names=list(itertools.chain.from_iterable(column_names))

main_df={i:pd.DataFrame() for i in currenies}
for i in main_df.keys():
    df=pd.DataFrame()
    for j in file_currency[i][32:35]:
        temp_df=pd.read_csv(j,index_col=0,parse_dates=True,names=column_names)
        temp_df=temp_df.resample('100ms').mean()
        df=pd.concat([df,temp_df])
    df.drop([''],axis=1,inplace=True)
    df.fillna(method='ffill',inplace=True)
    df.fillna(method='bfill',inplace=True)
    df['price']=(df['BP1']+df['AP1'])/2
    df['volume']=(df['BV1']+df['AV1'])/2
    df['spread']=df['AP1']-df['BP1']
    df['high']=df['price'].rolling(20).max()
    df['low']=df['price'].rolling(20).min()
    df['open']=df['price'].rolling(20).apply(lambda x: x.iloc[0])
    df['close']=df['price'].rolling(20).apply(lambda x: x.iloc[-1])
    df.drop(['price'],axis=1,inplace=True)
    main_df[i]=df

main_df.keys()
main_df['USDJPY'].dropna()
main_df['GBPUSD'].dropna()
main_df['CHFJPY'].dropna().shape[0]

lenth={i:main_df[i].dropna().shape[0] for i in main_df.keys()}
min_length_df = min(lenth, key=lenth.get)
