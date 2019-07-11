from matplotlib import pyplot as plt
import os
import pandas as pd
import re
import numpy as np
from joblib import dump, load
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


dirs = [(x[0] , x[2]) for x in os.walk('data') if len(x[0].split("/")) > 2 ]
dirs = [os.path.join(x[0], i) for x in dirs for i in x[1]]
set(dirs).difference(set([i for i in dirs if re.match('data/\d{4}/\d+/\d+',i)]))
dirs = [i for i in dirs if re.match('data/\d{4}/\d+/\d+',i)]
data = pd.concat([pd.read_csv(i, na_values="?") for i in dirs])


data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
data['Datetime_1'] = data.groupby(['Date'])['Datetime'].shift(1)
data['lag_time'] = (data['Datetime'] - data['Datetime_1']).dt.seconds
data['lag_time'].describe()




data = data.set_index('Datetime')
data = data.drop(["Date", "Time", "Datetime_1", "lag_time"], axis = 1)
data = data.apply(pd.to_numeric)

import statsmodels.api as sm
#

pct_vars = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
pct_vars_change = [i + "_pct_change" for i in pct_vars]
other = [i for i in data.columns if i not in pct_vars]
num_cols = data.columns

df_lag = data[pct_vars]
df_lag[pct_vars_change] = data[pct_vars].pct_change()
df_lag[other] = data[other]

df_lag_cols = df_lag.columns
LAG = 3
for i in range(1, LAG):
    for col in df_lag_cols:
        df_lag[col + '__' + str(i)] = df_lag[col].shift(i)

df_lag.replace([np.inf, -np.inf], np.nan, inplace = True)


ii = IterativeImputer()
data_imputed = ii.fit_transform(df_lag)
from joblib import dump, load
dump(ii, 'iterative_imputer_2_lags.joblib')


# df_lag.columns
imputed_2_lags = pd.DataFrame(data_imputed, columns = df_lag.columns, index = df_lag.index)
imputed_2_lags = imputed_2_lags.loc[:,num_cols]
imputed_2_lags.to_csv("imputed_2_lags.csv")
data.to_csv("data_original.csv")



# ii = IterativeImputer()


ii = load('iterative_imputer_10_lags.joblib')
data_imputed = ii.fit_transform(df_lag)

# df_lag.columns
imputed_2_lags = pd.DataFrame(data_imputed, columns = df_lag.columns, index = df_lag.index)
imputed_2_lags = imputed_2_lags.loc[:,num_cols]
imputed_2_lags.to_csv("imputed_2_lags.csv")
data.to_csv("data_original.csv")
