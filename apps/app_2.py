import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table

from app import app, page_style
from dash.dependencies import Input, Output
import plotly.graph_objs as go
# table_1_index = pd.read_csv('data/table_1_index.csv')
table_missing_vals = pd.read_csv('data/missing_vals.csv')

original_resampled_daily = pd.read_csv('data/resampled/original_resampled_daily.csv')
imput_2_resampled_daily = pd.read_csv('data/resampled/imput_2_resampled_daily.csv')
imput_10_resampled_daily = pd.read_csv('data/resampled/imput_10_resampled_daily.csv')
nan_difference = pd.read_csv('data/nan_difference.csv')

original_resampled_daily['Datetime'] = pd.to_datetime(original_resampled_daily.loc[:,'Datetime'], infer_datetime_format=True)
original_resampled_daily = original_resampled_daily.set_index('Datetime')
imput_2_resampled_daily['Datetime'] = pd.to_datetime(imput_2_resampled_daily.loc[:,'Datetime'], infer_datetime_format=True)
imput_2_resampled_daily = imput_2_resampled_daily.set_index('Datetime')
imput_10_resampled_daily['Datetime'] = pd.to_datetime(imput_10_resampled_daily.loc[:,'Datetime'], infer_datetime_format=True)
imput_10_resampled_daily = imput_10_resampled_daily.set_index('Datetime')
nan_difference['Datetime'] = pd.to_datetime(nan_difference.loc[:,'Datetime'], infer_datetime_format=True)
nan_difference = nan_difference.set_index('Datetime')

nan_difference = nan_difference.describe().\
                    apply(lambda x: round(x, 4)).\
                    loc[["mean", "50%","min", "max"],:].\
                    reset_index()


original_resampled_monthly = pd.read_csv('data/resampled/original_resampled_monthly.csv')
imput_2_resampled_monthly = pd.read_csv('data/resampled/imput_2_resampled_monthly.csv')
imput_10_resampled_monthly = pd.read_csv('data/resampled/imput_10_resampled_monthly.csv')

original_resampled_monthly['Datetime'] = pd.to_datetime(original_resampled_monthly.loc[:,'Datetime'], infer_datetime_format=True)
original_resampled_monthly = original_resampled_monthly.set_index('Datetime')
imput_2_resampled_monthly['Datetime'] = pd.to_datetime(imput_2_resampled_monthly.loc[:,'Datetime'], infer_datetime_format=True)
imput_2_resampled_monthly = imput_2_resampled_monthly.set_index('Datetime')
imput_10_resampled_monthly['Datetime'] = pd.to_datetime(imput_10_resampled_monthly.loc[:,'Datetime'], infer_datetime_format=True)
imput_10_resampled_monthly = imput_10_resampled_monthly.set_index('Datetime')



original_resampled_weekly = pd.read_csv('data/resampled/original_resampled_weekly.csv')
imput_2_resampled_weekly = pd.read_csv('data/resampled/imput_2_resampled_weekly.csv')
imput_10_resampled_weekly = pd.read_csv('data/resampled/imput_10_resampled_weekly.csv')

original_resampled_weekly['Datetime'] = pd.to_datetime(original_resampled_weekly.loc[:,'Datetime'], infer_datetime_format=True)
original_resampled_weekly = original_resampled_weekly.set_index('Datetime')
imput_2_resampled_weekly['Datetime'] = pd.to_datetime(imput_2_resampled_weekly.loc[:,'Datetime'], infer_datetime_format=True)
imput_2_resampled_weekly = imput_2_resampled_weekly.set_index('Datetime')
imput_10_resampled_weekly['Datetime'] = pd.to_datetime(imput_10_resampled_weekly.loc[:,'Datetime'], infer_datetime_format=True)
imput_10_resampled_weekly = imput_10_resampled_weekly.set_index('Datetime')



from itertools import product
dfs = [{"data_name" : 'original_data', "display_name" : 'Original Data',
                        "sampling_freq" : 'd', "data" : original_resampled_daily},

        {"data_name" : 'imputed_2_lags',"display_name" : 'Imputed 2 lags Data',
                        "sampling_freq" : 'd', "data" : imput_2_resampled_daily},

        {"data_name" : 'imputed_9_lags',"display_name" : 'Imputed 9 lags Data',
                        "sampling_freq" : 'd', "data" : imput_10_resampled_daily},

        {"data_name" : 'original_data',"display_name" : 'Original Data',
                        "sampling_freq" : 'm', "data" : original_resampled_monthly},

        {"data_name" : 'imputed_2_lags',"display_name" : 'Imputed 2 lags Data',
                        "sampling_freq" : 'm', "data" : imput_2_resampled_monthly},

        {"data_name" : 'imputed_9_lags',"display_name" : 'Imputed 9 lags Data',
                        "sampling_freq" : 'm', "data" : imput_10_resampled_monthly},

        {"data_name" : 'original_data',"display_name" : 'Original Data',
                        "sampling_freq" : 'w', "data" : original_resampled_weekly},

        {"data_name" : 'imputed_2_lags',"display_name" : 'Imputed 2 lags Data',
                        "sampling_freq" : 'w', "data" : imput_2_resampled_weekly},

        {"data_name" : 'imputed_9_lags',"display_name" : 'Imputed 9 lags Data',
                        "sampling_freq" : 'w', "data" : imput_10_resampled_weekly}]

text = sorted(list(set([(i["data_name"], i["display_name"]) for i in dfs])), key = lambda x: x[0], reverse = False)
text_dict = {i[0] : i[1] for i in text}

options = [{"data_name" : i[0], "display_name" : i[1]} for i in text]

dropdown_options = [{"label" : i, "value" : x} for (i, x) in
                            zip(list(map(lambda x: x.replace("_", " ").title(),
                                            original_resampled_daily.columns)),
                                        list(original_resampled_daily.columns))]


def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    , style={
        'background' : '#dbdbdb',
    })

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# colors = {
#     'background': '#ffefef',
#     'text': '#424856'
# }

layout = html.Div(style=page_style, children=[
    html.Br(),
    # dcc.Link('Go to App 2', href='/apps/app2'),
    html.H1(
        children='Dataset Preparation',
        style={
            'textAlign': 'center',
            # 'color': colors['text'],

            # 'position': 'absolute'
            # 'margin-left': 'auto',
            # 'margin-right': 'auto',
        }
    ),


    html.H4(
        children='Step 2: Missing Values.',
        style={
            'textAlign': 'left',
            # 'color': colors['text']
        }
    ),

    dcc.Markdown("""
        There is no particular pattern to missing data. Some days either the meter
        wasn't working or data was dropped intentionally
        """),

    dash_table.DataTable(
            id='table_app_2_1',
            columns=[{"name": i, "id": i} for i in table_missing_vals.columns],
            data=table_missing_vals.to_dict("rows"),
        ),

    # generate_table(table_missing_vals),
    dcc.Markdown("""
        If we're wrong in our assumptions we'll revisit the process after we explore the data.

        We'll proceed with NA's based on the following:
        - We're dealing with time series of 3 years - we should assume to have trends and cycles
            - Imputing with 0 won't work since at least some features won't be 0 (voltage)
            - Mean/Median/Mode imputing won't work because we will be ignoring  the trend and cycle values
        - Imputing with 'last seen value' works in some cases but not when it gives us a vector of constant
        value that has ∆X=0 across hours. Again, it ignores the daily/weekly/monthly cycles and seasonality
        - Perfect solution would be to train a series of recursive models that:
            - Iterate over the span that has no interruptions up to the point of first `np.nan`
            - ARIMA forecast the mising span
            - Proceed to the next span
            - Recursion
        - Alternative perfect - train a deep learning network that will fill the gaps in a similar manner
        (LSTM models are likely candidate)

        The method that we will use (until we have enough time to try the ones described as perfect and alternative_perfect)
        is `IterativeImputer` - we will create:
        - percent change of variables with consistent numeric shifts:
        ```py
        ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
        ```
        - cycle through all variables including pct_change variables
        - up to N create `lag(i...N)` variables
        - fill infinite values with `np.nan`
        - use `IterativeImputer` to fill all missing values
        - select only original columns that are now filled

        ```py
        pct_vars = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
        pct_vars_change = [i + "_pct_change" for i in pct_vars]
        other = [i for i in data.columns if i not in pct_vars]
        num_cols = data.columns

        df_lag = data[pct_vars]
        df_lag[pct_vars_change] = data[pct_vars].pct_change()
        df_lag[other] = data[other]

        LAG = 10 # We do two variants - one with 2 lags and one with 9
        for i in range(1, LAG):
            for col in lag_cols:
                df_lag[col + '__' + str(i)] = df_lag[col].shift(i)

        df_lag.replace([np.inf, -np.inf], np.nan, inplace = True)


        ```

        Here's a subset of variables created for this step:

        ```py
        ['Global_active_power',
         'Global_reactive_power',
         'Voltage',
         'Global_intensity',
         'Global_active_power_pct_change',
         'Global_reactive_power_pct_change',
         'Voltage_pct_change',
         'Global_intensity_pct_change',
         'Sub_metering_1',
         'Sub_metering_2',
         'Sub_metering_3',
         'Global_active_power__1',
         'Global_reactive_power__1',
         'Voltage__1',
         'Global_intensity__1',
         'Global_active_power_pct_change__1',
         'Global_reactive_power_pct_change__1',
         'Voltage_pct_change__1',
         'Global_intensity_pct_change__1',
         'Sub_metering_1__1',
         'Sub_metering_2__1',
         'Sub_metering_3__1'
         ...]
         ```

         Then as described above we use Imputer to fill NA's and save the dataset
         ```py
         ii = IterativeImputer()
         data_imputed = ii.fit_transform(df_lag)

         imputed_2_lags = pd.DataFrame(data_imputed, columns = df_lag.columns, index = df_lag.index)
         imputed_2_lags = imputed_2_lags.loc[:,num_cols]
         imputed_2_lags.to_csv("imputed_10_lags.csv")
         ```

         We will explain the transformations in the next section but for now all
         we need is to visualize the imputed values alongside the missing ones
         in the original dataset.


        """),

        html.Div(
            [html.Div(
                [html.H4("Imputed missing data")], style={'textAlign': "center"}),
                   html.Div(
                        [dcc.RadioItems(id="selected_frequency", value="d",
                          options=[{"label": "Daily Aggregation", "value": "d"},
                                    {"label": "Weekly Aggregation", "value": "w"},
                                    {"label": "Monthly Aggregation", "value": "m"}],
                                    labelStyle={'display': 'inline'}),
                           dcc.Dropdown(id="selected_variable", multi=False,
                                        value=dropdown_options[0]["value"],
                             options=dropdown_options)],
                                className="row",
                                style={"display": "inline",
                                        "width": "30%",
                                        "margin-left": "auto",
                                        "margin-right": "auto"}),
                       html.Div(
                    [dcc.Graph(id="my-graph")])], className="container"),
        dcc.Markdown("""
            We calculate the difference between imputed values for (Lag 9 Model - Lag 2 Model).

            Upon close examination we see that there isn't much difference between the
            2 and 9 lag imputed values.

            That being the Mean and Median - if we look at the
            min and max difference between two estimates we will see that difference can
            be extreme especially for the scale of these variables.

            """),

        dash_table.DataTable(
                id='table_app_2_2',
                columns=[{"name": i, "id": i} for i in nan_difference.columns],
                data=nan_difference.to_dict("rows"),
            ),
        html.Br(),
        ])

# list(map(lambda x: x.lower(), original_resampled_daily.columns))
#
#
#
# list(map(lambda x: x.replace("_", " "), original_resampled_daily.columns))
# list(original_resampled_daily.columns)


@app.callback(Output('my-graph', 'figure'),
    [Input('selected_frequency', 'value'), Input('selected_variable', 'value')])
def update_figure(frequency, variable):
    text = text_dict
    variable_string = [i['label'] for i in dropdown_options if i["value"] == variable][0]
    freq_string = {"m" : "Monthly", "w" : "Weekly", "d" : "Daily"}[frequency]
    aggregation_string = [("Total",('Global_active_power', 'Global_reactive_power',
                                        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3')),
                            ("Average", ('Voltage', 'Global_intensity'))]
    aggregation_string = [i[0] for i in aggregation_string if variable in i[1]][0]
    measurement_string = [i[0] for i in (("Kilowatt", ('Global_active_power', 'Global_reactive_power',
                                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3')),
                 ("Volt", ('Voltage')),
                 ("Amper", ('Global_intensity'))) if variable in i[1]][0]
    trace = []
    for dataframe in [i for i in dfs if i['sampling_freq'] == frequency]:
        trace.append(go.Scatter(x=dataframe['data'].index,
                                y=dataframe['data'][variable],
                                name=dataframe['display_name'],
                                mode='lines',
                                marker={'size': 3, "opacity": 0.7,
                                        "line": {'width': 0.2}},
                                line=dict(width=1)))
    return {"data": trace,
            "layout": go.Layout(title="{aggregation_string} {freq_string} {variable_string}".format(**{
                                        "freq_string" : freq_string,
                                        "variable_string" : variable_string,
                                        "aggregation_string" : aggregation_string}),
                                colorway=['#1b9e77', '#d95f02', '#7570b3'],
                                yaxis={"title": "{measurement_string}".format(**{
                                    "measurement_string": measurement_string}), "showgrid": False},
                                 xaxis={"title": "Date", "showgrid": False},
                                 paper_bgcolor='#ffefef',
                                plot_bgcolor='#e6e1e1')}



#  {'label': 'Voltage', 'value': 'Voltage'},
#  {'label': 'Global Intensity', 'value': 'Global_intensity'},
# Volt, Amper
