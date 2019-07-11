import pandas as pd
import numpy as np

data_10 = pd.read_csv('imputed_10_lags.csv')
data_10['Datetime'] = pd.to_datetime(data_10.loc[:,'Datetime'], infer_datetime_format=True)
data_10 = data_10.set_index('Datetime')

data_2 = pd.read_csv('imputed_2_lags.csv')
data_2['Datetime'] = pd.to_datetime(data_2.loc[:,'Datetime'], infer_datetime_format=True)
data_2 = data_2.set_index('Datetime')


data_original = pd.read_csv('data_original.csv')
data_original['Datetime'] = pd.to_datetime(data_original.loc[:,'Datetime'], infer_datetime_format=True)
data_original = data_original.set_index('Datetime')


data_10.describe().apply(lambda x: round(x, 4)).to_csv("dash_app/data/eda/describe_10.csv")
data_2.describe().apply(lambda x: round(x, 4)).to_csv("dash_app/data/eda/describe_2.csv")
data_original.describe().apply(lambda x: round(x, 4)).to_csv("dash_app/data/eda/describe_original.csv")
