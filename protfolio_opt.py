import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import os
from glob import glob
import itertools
import joblib  # For saving models
import scipy.optimize as sco

os.getcwd()
os.chdir('/Users/macbookpro/Documents/data')
file_names = glob('*.zip')
file_names.sort()
currenies = list(set([i.split('_')[-1].split('.')[0] for i in file_names]))

file_currency = {i: [] for i in currenies}
for i in file_names:
    for j in file_currency.keys():
        if j in i:
            file_currency[j].append(i)

dates={i:[] for i in currenies}
for i in file_names:
    for j in file_currency.keys():
        if j in i:
            dates[j].append(i.split('_')[2])

common_dates=set(dates[currenies[0]]) & set(dates[currenies[1]]) & set(dates[currenies[2]])

file_currency_filter={i:[] for i in currenies}
for i in file_names:
    for j in file_currency.keys():
        if j in i:
            if i.split('_')[2] in common_dates:
                file_currency_filter[j].append(i)
file_currency=file_currency_filter

column_names = [['Date']] \
               + [[f'BP{i}', f'BV{i}'] for i in range(1, 6)] \
               + ['M'] \
               + [[f'AP{i}', f'AV{i}'] for i in range(1, 6)] \
               + [['']]
column_names = list(itertools.chain.from_iterable(column_names))

main_df = {i: pd.DataFrame() for i in currenies}
for i in main_df.keys():
    df = pd.DataFrame()
    print(file_currency[i][:30])
    for j in file_currency[i][:30]:
        temp_df = pd.read_csv(j, index_col=0, parse_dates=True, names=column_names)
        temp_df.drop(['M', ''], axis=1, inplace=True)
        temp_df = temp_df.resample('60s').mean()
        df = pd.concat([df, temp_df])
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df['price'] = (df['BP1'] + df['AP1']) / 2
    df['volume'] = (df['BV1'] + df['AV1']) / 2
    df['spread'] = df['AP1'] - df['BP1']
    df['high'] = df['price'].rolling(20).max()
    df['low'] = df['price'].rolling(20).min()
    df['open'] = df['price'].rolling(20).apply(lambda x: x.iloc[0])
    df['close'] = df['price'].rolling(20).apply(lambda x: x.iloc[-1])
    main_df[i] = df


lenth={i:main_df[i].dropna().shape[0] for i in main_df.keys()}

min_length_df = min(lenth, key=lenth.get)
index=main_df[min_length_df].index

c_name=main_df[min_length_df].columns

for i in main_df.keys():
    main_df[i].columns=[j+i for j in main_df[i].columns]

df=pd.concat([main_df[i] for i in main_df.keys()],axis=1)
df.dropna(inplace=True)

main_df={i:pd.DataFrame() for i in currenies}

for i in main_df.keys():
    names=[j for j in df.columns if i in j]
    main_df[i]=df[names]
    main_df[i].columns=c_name

mid_price = pd.DataFrame()

for i in currenies:
    mid_price[i] = main_df[i].price.apply(lambda x: round(x, 6))

mid_price.to_csv('mid_price.csv',index=True)

window_size=1
# Generate log return for each dataframe in main_df, categorize it, and split datasets into train and test
train_test_data = {}
model_predictions={}
for currency, df in main_df.items():
    df['log_return'] = np.log(df['price'] / df['price'].shift(window_size))
    df['log_return_cat'] = df['log_return'].apply(lambda x: 1 if x > 0 else (2 if x < 0 else 0))
    df['log_return_cat'] = df['log_return_cat'].fillna(0)
    df.dropna(inplace=True)
    X = df.drop(['log_return', 'log_return_cat'], axis=1)
    y = df['log_return_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=False)
    train_test_data[currency] = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    # Train LightGBM model
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval)

    # Save the model
    joblib.dump(gbm, f'{currency}_lgbm_model.pkl')

    # Make predictions
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    model_predictions[currency] = pd.DataFrame(y_pred)


tick_move=0.0001
protfolio={'expect_price':{i:[] for i in currenies},'volatility':{i:[] for i in currenies}}
for i in currenies:
    expect_price=[]
    volatility=[]
    for j in range(len(train_test_data[i]['X_test'])):
        price_t=train_test_data[i]['X_test'].price.values[j]
        prob=model_predictions[i].values[j]
        expect_price_t=price_t + (prob[1]*tick_move) - (prob[2]*tick_move)
        volatility_t=(prob[0]*(price_t)**2) + (prob[1]*(price_t+tick_move)**2) + (prob[2]*(price_t-tick_move)**2) - (expect_price_t)**2
        expect_price.append(expect_price_t)
        volatility.append(volatility_t)
    protfolio['expect_price'][i]=expect_price
    protfolio['volatility'][i]=volatility

expect_price=pd.DataFrame(protfolio['expect_price'])
volatility=pd.DataFrame(protfolio['volatility'])

histcal_price=pd.DataFrame([pd.concat([train_test_data[i]['X_train'].price.tail(30),train_test_data[i]['X_test'].price] ,axis=0) for i in currenies]).T
histcal_price.columns=currenies
histcal_bid=pd.DataFrame([pd.concat([train_test_data[i]['X_train'].BP1.tail(30),train_test_data[i]['X_test'].BP1] ,axis=0) for i in currenies]).T
histcal_bid.columns=currenies
histcal_ask=pd.DataFrame([pd.concat([train_test_data[i]['X_train'].AP1.tail(30),train_test_data[i]['X_test'].AP1] ,axis=0) for i in currenies]).T
histcal_ask.columns=currenies

expect_price.index=train_test_data[currenies[0]]['X_test'].index
volatility.index=train_test_data[currenies[0]]['X_test'].index
expect_price=pd.concat([histcal_price.head(30),expect_price],axis=0)
volatility=pd.concat([histcal_price.head(30),volatility],axis=0)


expected_return=expect_price.shift(-1)-histcal_price
expected_return.dropna(inplace=True)
expected_mean_return=expected_return.rolling(31).mean()
expected_mean_return.dropna(inplace=True)

expected_cov_return=expected_return.rolling(31).cov()
expected_cov_return.dropna(inplace=True)

expected_return=expected_return.loc[expected_mean_return.index]



