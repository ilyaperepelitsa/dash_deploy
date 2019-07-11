from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

original_data = pd.read_csv('data_original.csv')
imputed_2_data = pd.read_csv('imputed_2_lags.csv')
imputed_10_data = pd.read_csv('imputed_10_lags.csv')
imputed_2_lags_outliers_aware = pd.read_csv('imputed_2_lags_outliers_aware.csv')

original_data['Datetime'] = pd.to_datetime(original_data['Datetime'], infer_datetime_format=True)
original_data = original_data.set_index('Datetime')

imputed_2_data['Datetime'] = pd.to_datetime(imputed_2_data['Datetime'], infer_datetime_format=True)
imputed_2_data = imputed_2_data.set_index('Datetime')

imputed_10_data['Datetime'] = pd.to_datetime(imputed_10_data['Datetime'], infer_datetime_format=True)
imputed_10_data = imputed_10_data.set_index('Datetime')

imputed_2_lags_outliers_aware['Datetime'] = pd.to_datetime(imputed_2_lags_outliers_aware['Datetime'], infer_datetime_format=True)
imputed_2_lags_outliers_aware = imputed_2_lags_outliers_aware.set_index('Datetime')

(imputed_10_data[original_data.isnull().any(axis = 1)] -
        imputed_2_data[original_data.isnull().any(axis = 1)]).to_csv("dash_app/data/nan_difference.csv")
# original_data.columns

# imputed_2_data['Global_active_power'] /= 60
# imputed_2_data['Global_reactive_power'] /= 60
# imputed_2_data.head()

# Index(['Global_active_power', 'Global_reactive_power', 'Voltage',
#        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
#        'Sub_metering_3'],
#       dtype='object')
def transform_resample(data, frequency = 'd'):
    transformed_data = pd.DataFrame(data.resample(frequency).\
                        apply(lambda x: pd.Series([(x['Global_active_power']/60).sum(),
                                    (x['Global_reactive_power']/60).sum(),
                                    x['Voltage'].mean(),
                                    x['Global_intensity'].mean(),
                                    (x['Sub_metering_1']/1000).sum(),
                                    (x['Sub_metering_2']/1000).sum(),
                                    (x['Sub_metering_3']/1000).sum()])))

    transformed_data = transformed_data.\
                                rename(columns={i: x for i,x in
                                                zip(transformed_data.columns,
                                                data.columns)})
    return transformed_data

transformed_1_monthly = transform_resample(original_data, 'M')
transformed_2_monthly = transform_resample(imputed_2_data, 'M')
transformed_10_monthly = transform_resample(imputed_10_data, 'M')

transformed_1_monthly.to_csv("dash_app/data/resampled/original_resampled_monthly.csv")
transformed_2_monthly.to_csv("dash_app/data/resampled/imput_2_resampled_monthly.csv")
transformed_10_monthly.to_csv("dash_app/data/resampled/imput_10_resampled_monthly.csv")



transformed_1_daily = transform_resample(original_data, 'd')
transformed_2_daily = transform_resample(imputed_2_data, 'd')
transformed_10_daily = transform_resample(imputed_10_data, 'd')
final_daily = transform_resample(imputed_2_lags_outliers_aware, 'd')
# imputed_2_lags_outliers_aware


transformed_1_daily.to_csv("dash_app/data/resampled/original_resampled_daily.csv")
transformed_2_daily.to_csv("dash_app/data/resampled/imput_2_resampled_daily.csv")
transformed_10_daily.to_csv("dash_app/data/resampled/imput_10_resampled_daily.csv")
final_daily.to_csv("dash_app/data/resampled/final_daily.csv")


transformed_1_weekly = transform_resample(original_data, 'w')
transformed_2_weekly = transform_resample(imputed_2_data, 'w')
transformed_10_weekly = transform_resample(imputed_10_data, 'w')

transformed_1_weekly.to_csv("dash_app/data/resampled/original_resampled_weekly.csv")
transformed_2_weekly.to_csv("dash_app/data/resampled/imput_2_resampled_weekly.csv")
transformed_10_weekly.to_csv("dash_app/data/resampled/imput_10_resampled_weekly.csv")



first_day = imputed_10_data.copy()
first_day['day'] = first_day.index.normalize()
first_day['Global_active_power'] /= 60
first_day.loc[:,['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']] /= 1000
first_day.loc[first_day['day'] == first_day['day'].unique()[0],:].\
                drop("day", axis = 1).to_csv("dash_app/data/first_day.csv")

first_ten_days = imputed_10_data.copy()
first_ten_days['day'] = first_ten_days.index.normalize()
first_ten_days['Global_active_power'] /= 60
first_ten_days.loc[:,['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']] /= 1000
first_ten_days.loc[first_ten_days['day'].isin(first_ten_days['day'].unique()[0:9]),:].\
                to_csv("dash_app/data/first_ten_days.csv")

sorted(first_ten_days['day'].unique())

# Index(['Global_active_power', 'Global_reactive_power', 'Voltage',
#        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
#        'Sub_metering_3', 'day'],
#       dtype='object')
