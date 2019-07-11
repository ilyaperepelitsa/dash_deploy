# from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
# import statsmodels.api as sm
# from statsmodels.tsa.seasonal import seasonal_decompose
#
# original_data = pd.read_csv('data_original.csv')
# imputed_2_data = pd.read_csv('imputed_2_lags.csv')
#
# original_data['Datetime'] = pd.to_datetime(original_data['Datetime'], infer_datetime_format=True)
# original_data = original_data.set_index('Datetime')
#
# imputed_2_data['Datetime'] = pd.to_datetime(imputed_2_data['Datetime'], infer_datetime_format=True)
# imputed_2_data = imputed_2_data.set_index('Datetime')
#
# imputed_2_data
# pew =  (imputed_2_data['Global_active_power'] * 1000 / 60) - imputed_2_data.loc[:,imputed_2_data.columns[4:]].sum(axis = 1)
# pew.head()
#
# (imputed_2_data['Global_active_power'] * 1000 / 60) + (imputed_2_data['Global_reactive_power'] * 1000 / 60)
#
# imputed_2_data
#
#
#
# plt.figure(figsize=(15,6))
# orig, = plt.plot(original_data['Global_active_power'].resample('W').mean(), label='missing', color = "red")
# imputed_2, = plt.plot(imputed_2_data['Global_active_power'].resample('W').mean(), label='imputed', color = 'green')
# imputed_10, = plt.plot(imputed_10_data['Global_active_power'].resample('W').mean(), label='imputed_10', color = 'blue')
# plt.legend(handles=[orig, imputed_2, imputed_10])
# plt.show()
#
#
#
#
# plt.figure(figsize=(15,6))
# orig, = plt.plot(original_data['Global_reactive_power'].resample('W').mean(), label='missing', color = "red")
# imputed_2, = plt.plot(imputed_2_data['Global_reactive_power'].resample('W').mean(), label='imputed', color = 'green')
# imputed_10, = plt.plot(imputed_10_data['Global_reactive_power'].resample('W').mean(), label='imputed_10', color = 'blue')
# plt.legend(handles=[orig, imputed_2, imputed_10])
# plt.show()
#
#
# plt.figure(figsize=(15,6))
# orig, = plt.plot(original_data['Voltage'].resample('W').mean(), label='missing', color = "red")
# imputed_2, = plt.plot(imputed_2_data['Voltage'].resample('W').mean(), label='imputed', color = 'green')
# plt.legend(handles=[orig, imputed_2])
# plt.show()
#
#
#
# plt.figure(figsize=(15,6))
# orig, = plt.plot(original_data['Global_intensity'].resample('W').mean(), label='missing', color = "red")
# imputed_2, = plt.plot(imputed_2_data['Global_intensity'].resample('W').mean(), label='imputed', color = 'green')
# plt.legend(handles=[orig, imputed_2])
# plt.show()
#
#
#
# plt.figure(figsize=(15,6))
# orig, = plt.plot(original_data['Sub_metering_1'].resample('W').mean(), label='missing', color = "red")
# imputed_2, = plt.plot(imputed_2_data['Sub_metering_1'].resample('W').mean(), label='imputed', color = 'green')
# plt.legend(handles=[orig, imputed_2])
# plt.show()
#
#
#
# plt.figure(figsize=(15,6))
# orig, = plt.plot(original_data['Sub_metering_2'].resample('W').mean(), label='missing', color = "red")
# imputed_2, = plt.plot(imputed_2_data['Sub_metering_2'].resample('W').mean(), label='imputed', color = 'green')
# plt.legend(handles=[orig, imputed_2])
# plt.show()
#
#
# plt.figure(figsize=(15,6))
# orig, = plt.plot(original_data['Sub_metering_3'].resample('W').mean(), label='missing', color = "red")
# imputed_2, = plt.plot(imputed_2_data['Sub_metering_3'].resample('W').mean(), label='imputed', color = 'green')
# plt.legend(handles=[orig, imputed_2])
# plt.show()
#
#
# decomposition = sm.tsa.seasonal_decompose(imputed_2_data['Global_active_power'], model='additive')
#
# fig = decomposition.plot()
# plt.show()
#
# imputed_2_data.index.freq = pd.infer_freq(imputed_2_data.index)
#
#
#
# pd.date_range(imputed_2_data.index.min(), imputed_2_data.index.max(), freq='60s').shape == imputed_2_data.index.shape
#
# imputed_2_data = imputed_2_data.set_index(pd.date_range(imputed_2_data.index.min(), imputed_2_data.index.max(), freq='60s'))
# imputed_2_data
#
#
#
# plt.figure(figsize=(15,6))
# orig, = plt.plot(original_data['Sub_metering_3'].resample('M').mean(), label='missing', color = "red")
# imputed_2, = plt.plot(imputed_2_data['Sub_metering_3'].resample('W').mean(), label='imputed', color = 'green')
# plt.legend(handles=[orig, imputed_2])
# plt.show()
#
# imputed_2_data.index
#
#
#
#
# #
# data_original = pd.read_csv('data_original.csv')
# data_original['Datetime'] = pd.to_datetime(data_original.loc[:,'Datetime'], infer_datetime_format=True)
# data_original = data_original.set_index('Datetime')
#
# data_original.sample(100000).to_csv('data_original_sampled.csv')
# # data_original
# # imputed_10_lags
#
#
# imputed_10_lags = pd.read_csv('imputed_10_lags.csv')
# imputed_10_lags['Datetime'] = pd.to_datetime(imputed_10_lags.loc[:,'Datetime'], infer_datetime_format=True)
# imputed_10_lags = imputed_10_lags.set_index('Datetime')
#
# imputed_10_lags.sample(100000).to_csv('imputed_10_lags_sampled.csv')
#
#
#
# imputed_10_lags = pd.read_csv('imputed_10_lags.csv')
# imputed_10_lags['Datetime'] = pd.to_datetime(imputed_10_lags.loc[:,'Datetime'], infer_datetime_format=True)
# imputed_10_lags = imputed_10_lags.set_index('Datetime')
#
# imputed_10_lags.sample(100000).to_csv('imputed_10_lags_sampled.csv')
#
#
#
# imputed_2_lags_outliers_aware = pd.read_csv('imputed_2_lags_outliers_aware.csv')
# imputed_2_lags_outliers_aware['Datetime'] = pd.to_datetime(imputed_2_lags_outliers_aware.loc[:,'Datetime'], infer_datetime_format=True)
# imputed_2_lags_outliers_aware = imputed_2_lags_outliers_aware.set_index('Datetime')
#
# imputed_2_lags_outliers_aware.sample(100000).to_csv('imputed_2_lags_outliers_aware_sampled.csv')


prediction = pd.read_csv("final_model/final_prediction.csv", header = None)[1]
data = pd.read_csv("dash_app/data/model/predictions/0_model_est.csv", header = 0)
ind = data["Datetime"].values[-1]
ind = pd.date_range(start=pd.Timestamp(ind, freq = "D"), periods=8)[1:]

prediction.sum()
pd.DataFrame({'y_hat' : prediction.values}, index = ind).to_csv("final_model/final_prediction_df.csv")
# dash_app/data/model/predictions/0_model_est.csv
#
# import pkg_resources
# pkg_resources.get_distribution("pandas").version
# pkg_resources.get_distribution("numpy").version
# pkg_resources.get_distribution("dash").version
# pkg_resources.get_distribution("dash_table").version
# pkg_resources.get_distribution("plotly").version
# pkg_resources.get_distribution("numpy").version
#
# import os
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# import dash_table
#
# import plotly.graph_objs as go
