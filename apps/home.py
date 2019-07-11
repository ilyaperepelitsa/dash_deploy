import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table

from app import app, page_style
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from apps.app_5 import return_ts, return_line_trace
import os
import itertools


dirs = [x[2] for x in os.walk('data/model/predictions')]
dirs = list(itertools.chain(*dirs))
dirs = [(int(i.split("_")[0]), 'data/model/predictions/' + i) for i in dirs]
dirs.sort(key = lambda x: x[0])
dirs = dirs[0:10]

final_predictions_df = pd.read_csv('../final_model/final_prediction_df.csv')
line_cols = ['#e41a1c','#377eb8','#4daf4a',
            '#984ea3','#ff7f00','#ffff33',
            '#a65628','#f781bf','#999999']

heat_options = [
                # {'value' : 'cols_drop', 'label' : 'Columns dropped'},
                {'value' : 'dropout', 'label' : 'LSTM dropout'},
                {'value' : 'lag_size', 'label' : 'Lag size'},
                {'value' : 'n_drop', 'label' : 'Number of variables dropped'}]
scores_final = pd.read_csv('data/model/scores_final.csv')
scores_final = scores_final.drop(scores_final.columns[0], axis=1)

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

model_scores = pd.read_csv('data/model/model_scores.csv', index_col = None)
model_scores = model_scores.drop(model_scores.columns[0], axis=1)


traces_models_training = [return_line_trace(model_scores, 0, train_or_test = 'train',color = '#088c08'),
       return_line_trace(model_scores, 0, train_or_test = 'test', color = '#055205',width = 2)]

traces_models_training.extend([return_line_trace(model_scores, i,
                            train_or_test = 'test',
                            color = '#292929', width = 0.5) for i in range(1, 10)])
figure_models_training = {"data": traces_models_training,
          "layout": go.Layout(
                      yaxis={"title": "MSE", "showgrid": False, 'automargin' : True,},
                      title='Train and Test MSE - final model compared to ten best',
                      xaxis={"showgrid": False, 'automargin' : True,},
                      paper_bgcolor='#ffefef',
                      plot_bgcolor='#e6e1e1',
                      )}



traces_models_training_worst = [return_line_trace(model_scores, 0, train_or_test = 'train',color = '#088c08'),
       return_line_trace(model_scores, 0, train_or_test = 'test', color = '#055205',width = 2)]

traces_models_training_worst.extend([return_line_trace(model_scores, i,
                            train_or_test = 'test',
                            color = '#292929', width = 0.5) for i in
                            range(model_scores.id.max() - 10, model_scores.id.max())])
figure_models_training_worst = {"data": traces_models_training_worst,
          "layout": go.Layout(
                      yaxis={"title": "MSE", "showgrid": False, 'automargin' : True,},
                      title='Train and Test MSE - final model compared to ten best',
                      xaxis={"showgrid": False, 'automargin' : True,},
                      paper_bgcolor='#ffefef',
                      plot_bgcolor='#e6e1e1',
                      )}



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



line_cols = ['#e41a1c','#377eb8','#4daf4a',
            '#984ea3','#ff7f00','#ffff33',
            '#a65628','#f781bf','#999999']


first_day = pd.read_csv('data/first_day.csv')
first_day['Datetime'] = pd.to_datetime(first_day.loc[:,'Datetime'], infer_datetime_format=True)
first_day = first_day.set_index('Datetime')

first_ten_days = pd.read_csv('data/first_ten_days.csv')
first_ten_days['Datetime'] = pd.to_datetime(first_ten_days.loc[:,'Datetime'], infer_datetime_format=True)
first_ten_days = first_ten_days.set_index('Datetime')


data_original = pd.read_csv('../data_original_sampled.csv')
data_original['Datetime'] = pd.to_datetime(data_original.loc[:,'Datetime'], infer_datetime_format=True)
data_original = data_original.set_index('Datetime')


imputed_10_lags = pd.read_csv('../imputed_10_lags_sampled.csv')
imputed_10_lags['Datetime'] = pd.to_datetime(imputed_10_lags.loc[:,'Datetime'], infer_datetime_format=True)
imputed_10_lags = imputed_10_lags.set_index('Datetime')


imputed_2_lags_outliers_aware = pd.read_csv('../imputed_2_lags_outliers_aware_sampled.csv')
imputed_2_lags_outliers_aware['Datetime'] = pd.to_datetime(imputed_2_lags_outliers_aware.loc[:,'Datetime'], infer_datetime_format=True)
imputed_2_lags_outliers_aware = imputed_2_lags_outliers_aware.set_index('Datetime')

# describe_original = pd.read_csv('data/eda/describe_original.csv')
# describe_2 = pd.read_csv('data/eda/describe_2.csv')
# describe_10 = pd.read_csv('data/eda/describe_10.csv')
describe_final = pd.read_csv('data/eda/describe_final.csv')

describe_original_outliers = pd.read_csv('data/eda/describe_original_outliers.csv')
describe_original_no_outliers = pd.read_csv('data/eda/describe_original_non_outliers.csv')

power_vars_dict = {'Global_active_power' : "Global active power",
            'Sub_metering_1' : "Kitchen",
            'Sub_metering_2' : "Laundry",
            'Sub_metering_3' : "Water heater and AC"}
power_vars = [{"label" : x, 'value' : i} for i, x in power_vars_dict.items()]

full_vars_measure = {'Global_active_power' : "Kilowatt / hour",
            'Global_reactive_power' : "Kilowatt / hour",
            'Voltage' : 'Volt',
            'Global_intensity' : 'Ampere',
            'Sub_metering_1' : "Watt / minute",
            'Sub_metering_2' : "Watt / minute",
            'Sub_metering_3' : "Watt / minute"}

full_vars_dict = {'Global_active_power' : "Global active power",
            'Global_reactive_power' : "Global reactive power",
            'Voltage' : 'Voltage',
            'Global_intensity' : 'Global intensity',
            'Sub_metering_1' : "Kitchen",
            'Sub_metering_2' : "Laundry",
            'Sub_metering_3' : "Water heater and AC"}
full_vars = [{"label" : x, 'value' : i} for i, x in full_vars_dict.items()]



imput_10_resampled_weekly = pd.read_csv('data/resampled/imput_10_resampled_weekly.csv')
imput_10_resampled_weekly['Datetime'] = pd.to_datetime(imput_10_resampled_weekly.loc[:,'Datetime'], infer_datetime_format=True)
imput_10_resampled_weekly = imput_10_resampled_weekly.set_index('Datetime')
imput_10_resampled_weekly["Rest"] = imput_10_resampled_weekly['Global_active_power'] - \
                            imput_10_resampled_weekly.loc[:,['Sub_metering_1',
                                                    'Sub_metering_2',
                                                    'Sub_metering_3']].sum(axis = 1)


init_hist_traces =[
            dict(
            x=imput_10_resampled_weekly.index,
            y=imput_10_resampled_weekly[y],
            name=name,
            hoverinfo='x+y',
            mode='lines',
            line=dict(width=1,
                      color=color),
            stackgroup='one'
        ) for y, name, color in zip(['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', "Rest"],
                                ['Kitchen', 'Laundry', 'Water heater and AC', "Rest"],
                                ['#1b9e77', '#d95f02', '#7570b3', 'black'])]
layout = dict(font=dict(family='Aleo'),
                    title={
                    'text':'Total weekly power consumption in kitchen, laundry, by water heater and AC'},
                        yaxis={"title": "Kilowatt per week",'textAlign': "center"},
                                        xaxis={"showgrid": False},
                                        paper_bgcolor='#ffefef',
                                        plot_bgcolor='#e6e1e1')


#
#
final_daily = pd.read_csv('data/resampled/final_daily.csv')
final_daily['Datetime'] = pd.to_datetime(final_daily.loc[:,'Datetime'], infer_datetime_format=True)
final_daily = final_daily.set_index('Datetime')
final_daily["Rest"] = final_daily['Global_active_power'] - \
                            final_daily.loc[:,['Sub_metering_1',
                                                    'Sub_metering_2',
                                                    'Sub_metering_3']].sum(axis = 1)
# # print(imput_10_resampled_daily.columns)
final_daily_traces =[
            dict(
            x=final_daily.index,
            y=final_daily[y],
            name=name,
            hoverinfo='x+y',
            mode='lines',
            line=dict(width=0.5,
                      color=color),
            stackgroup='one'
        ) for y, name, color in zip(['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', "Rest"],
                                ['Kitchen', 'Laundry', 'Water heater and AC', "Rest"],
                                ['#1b9e77', '#d95f02', '#7570b3', 'black'])]

final_layout = dict(font=dict(family='Aleo'),
                    title={
                    'text':'Total daily power consumption in kitchen, laundry, by water heater and AC'},
                        yaxis={"title": "Kilowatt per day",'textAlign': "center"},
                                        xaxis={"showgrid": False},
                                        paper_bgcolor='#ffefef',
                                        plot_bgcolor='#e6e1e1')




first_day_lines =[
            go.Scatter(
            x=first_day.index,
            y=first_day[y],
            name=name,
            mode='lines',
            line=dict(width=1,
                      color=color),
        ) for y, name, color in zip(['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', "Global_active_power"],
                                ['Kitchen', 'Laundry', 'Water heater and AC', "Global power consumption"],
                                ['#1b9e77', '#d95f02', '#7570b3', 'black'])]
layout_lines = dict(font=dict(family='Aleo'), yaxis={"title": "Kilowatt per minute"},
                title={'text':'Global power consumption, consumption in kitchen, laundry, by water heater and AC'},
                                            xaxis={"showgrid": False,
                                                    'automargin' : True,},
                                            paper_bgcolor='#ffefef',
                                            plot_bgcolor='#e6e1e1')

# from itertools import product


layout = html.Div(style=page_style, children=[
    html.Br(),
    # dcc.Link('Go to App 2', href='/apps/app2'),
    html.H1(
        children='Using LSTM to predict power consumption',
        style={
            'textAlign': 'center',

        }
    ),

    html.H4(
        children='Ilya Perepelitsa, July 11th, 2019',
        style={
            'textAlign': 'right',
            # 'color': colors['text']
        }
    ),

    html.H4(
        children='Final prediction - 7 days of unseen data from October 27th to 3rd of December',
        style={
            'textAlign': 'left',
            # 'color': colors['text']
        }
    ),

    dash_table.DataTable(
            id='final_vector',
            columns=[{"name": i, "id": i} for i in final_predictions_df.columns],
            data=final_predictions_df.to_dict("rows"),

        ),

        dcc.Markdown("""
            Total for the next 7 days = 182.407229 Kilowatt
            """),
    html.H4(
        children='Step 1-2: Loading and imputing missing values.',
        style={
            'textAlign': 'left',
            # 'color': colors['text']
        }
    ),

    dcc.Markdown("""
        We started our analysis with loading data and imputing
        missing values.

        The data was missing completely for certain observations so we had to resort to Iterative Imputer
        in order to simulate how data behaves in relation to other variables and their lags.

        We used 2 and 9 lags ultimately noticing that there were some outliers/anomalies in the original
        data. After they were detected with IsolationForest model we set outliers to `np.nan` and repeated
        the IterativeImputer to the whole set.

        *The visualization below presents one of the earlier steps of imputations without replacing anomalies*
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
                    [dcc.Graph(id="my-graph_home")])], className="container"),
        html.Br(),

        html.H4(
            children='Step 3-5: Transformations and Variable examination',
            style={
                'textAlign': 'left',
                # 'color': colors['text']
            }
        ),

        dcc.Markdown("""
            Then we proceeded to examination of weekly data, introduced data transformations
            that allowed us to determine the power consumption not captured by submeters
            and identified seasonality in data as well as some other speculations like
            a potential that residents go on vacation every August and potential
            AC installation after the first summer in our observations.
            """),

        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="hist-graph", clear_on_unhover=True,
                                figure = {"data": init_hist_traces, "layout": layout}),]),

            ],
            className="container"),


        dcc.Markdown("""
            We looked into one day of power consumption by source and made some speculations
            regarding the schedule of residents, guessed that maybe there is more than one
            residents and they have slightly different work schedules.

            We also made some guesses about how appliances consume power.
            """),

        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="line-graph", clear_on_unhover=True,
                                figure = {"data": first_day_lines, "layout": layout_lines}),],
                                style={"width": "100%"}),
            ],
            className="container"),

        dcc.Markdown("""
            Then we proceeded to examine multiple days of power consumption.
            """),

        html.Div(
            [
            html.Div(dcc.Dropdown(id="selected_variable-home", multi=False,
                         value=power_vars[0]["value"],
                         options=power_vars)),
            html.Div([
                    dcc.Slider(id='my-slider-home', min=2, max=len(first_ten_days.day.unique()), step=1, value=2),
                    html.Div(id='slider-output-container-home')]),
            html.Div(
                    [dcc.Graph(id="line_graph_multiple-home")]),
            ],
            className="container"),

        dcc.Markdown("""
            We identified issues in outliers and missing values methodology during variable
            distribution plotting and examination of `df.describe`. We also made
            a decision to plot a 100k sample of final data in order to preserve
            data loading time.
            """),

        html.Div(
            [
            html.Div(dcc.Dropdown(id="selected_variable_hist_final-home", multi=False,
                                    value=full_vars[0]["value"],
                         options=full_vars),
                            className="row",
                            style={"display": "inline",
                                    "width": "30%",
                                    "margin-left": "auto",
                                    "margin-right": "auto"}),
            html.Div(
                    [dcc.Graph(id="hist-graph_4-home", clear_on_unhover=True)]),
            ],
            className="container"),

        dcc.Markdown("""
            Finally we resampled original data to transform it to daily aggregates
            both for computational reasons and in order to reshape data to fit our
            objective which is stated as finding total consumption for the next 7 days
            even though minute-sampled data could provide more accurate estimates,
            it's more convenient to deal with data that fits the objective.
            """),




        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="hist-final", clear_on_unhover=True,
                                figure = {"data": final_daily_traces, "layout": final_layout}),]),

            ],
            className="container"),

        html.H4(
            children='Step 6: Creating LSTM model',
            style={
                'textAlign': 'left',
            }
        ),

        dcc.Markdown("""
            We ran hundreds of model training operations and finally trained a grid of 108
            LSTM models in order to test multiple hyperparameters:
            - Number of dropped variables
            - Number of periods to look back in order to forecast values from current day through the week
            - Portion of LSTM cells to drop in order to avoid over-fitting

            Even though the following model looks quite simple, we designed
            data transformations very rigorously, giving one model enough attention
            given that we couldn't dedicate significant computing power and time in order
            to test other models.

            ```py

            model = Sequential()
            model.add(LSTM(150, input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dropout(dropout))
            model.add(Dense(7))
            model.compile(loss='mean_squared_error', optimizer='adam')
            history = model.fit(train_X, train_y,
                                epochs=200, batch_size=200,
                                validation_data=(test_X, test_y), verbose=0, shuffle=False)
            ```

            We also utilized a `MinMaxScaler` on this step rather than on earlier exploratory
            step in order to study data better.
            """),

            html.H4(
                children='Step 7: Model evaluation.',
                style={
                    'textAlign': 'left',
                    # 'color': colors['text']
                }
            ),

            dcc.Markdown("""
                We examined how different hyperparameters of our 108 models
                affect the MSE and identified some patterns for combinations of
                such parameters using the following heat map.
                """),

            html.Div([
                    html.Div([
                         dcc.Dropdown(id="heat_var_y_home", multi=False,
                                      value=heat_options[0]["value"],
                           options=heat_options),
                    html.H6("Select Y axis")],
                                    className="column",
                                 style={"display": "inline-block'",
                                         "width": "45%",
                                         'background' : '#ffefef',
                                         'textAlign': 'center'}),
                     html.Div([
                         dcc.Dropdown(id="heat_var_x_home", multi=False,
                                      # value=heat_options[1]["value"],
                                      value=heat_options[-1]["value"],
                           options=heat_options),
                     html.H6("Select X axis")],
                            className="column",
                             style={"display": "inline-block'",
                                     "width": "45%",
                                     'background' : '#ffefef',
                                     'textAlign': 'center'}),

                     html.Br(),
                     html.Div([dcc.Graph(id="heat_graph_home",
                                            style={"margin-top": "7rem",
                                                    "margin-right": "auto",
                                                    "margin-left": "auto",
                                                    "width": "80%"
                                                    })],
                      className="row")

                         ]),

             dcc.Markdown("""
                 Then we examined the training process of the best model we trained
                 against some contestants and the worst trained LSTM models. We
                 speculated that some of the better models might be very similar
                 to our final model.
                 """),

              html.Div(dcc.Graph(id="training_graph_home", clear_on_unhover=True,
                                 figure = figure_models_training),
                                      style={"width": "100%",
                                      },

                  className="container"),

              html.Div(dcc.Graph(id="training_graph_worst_home", clear_on_unhover=True,
                                 figure = figure_models_training_worst),
                                      style={"width": "100%",
                                      },

                  className="container"),


              dcc.Markdown("""
                  Finally we plotted the predictions of our main model and up to
                  next best 9 models next to historical "ground truth" data.

                  We confirmed our suspicion - the differences in models are not substantial.
                  As we will restate later, we can further improve the model using
                  the conclusions we drew from the previous plots.
                  """),

             html.Div(
                 [
                 html.Div([
                         dcc.Slider(id='time_series_slider_home', min=0, max=len(dirs), step=1, value=2),
                         html.Div(id='slider_ts_annotation_home')]),
                 html.Div(
                         [dcc.Graph(id="check_predictions_home")]),
                 ],
                 className="container"),


              html.H4(
                  children='Step 8: Deployment and next steps.',
                  style={
                      'textAlign': 'left',
                      # 'color': colors['text']
                  }
              ),

              dcc.Markdown("""
                  Finally, we concluded that other models need to be tested and evaluated
                  in addition to further training of more models using our initial findings
                  regarding dropping submeter data and using lags of 20-25.

                  We suggested that bringing model transformers together and deploying Flask
                  or Django apps to provide predictions based on API inputs. Even though a broad range of
                  options wasn't considered and we focused mainly on micro services
                  deployment sugestions, it's important to note that any model
                  deployment is determined by the goals - whether it's an ad-hoc
                  internal service or high-performance web server providing forecasts
                  for millions of users matters in the choice of deployment techniques.

                  We didn't consider re-building model in lower languages as a technique
                  for optimization, even though we didn't dedicate much attention to
                  technical optimization in our analysis.

                  Main suggested options involved Docker and Kuberneters - based
                  deployment methods.
                  """),

        ])




@app.callback(Output('my-graph_home', 'figure'),
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



@app.callback(
    dash.dependencies.Output('slider-output-container-home', 'children'),
    [dash.dependencies.Input('my-slider-home', 'value')])
def update_output(value):
    return 'Comparing {} days of power consumption'.format(value)



@app.callback(
    dash.dependencies.Output('line_graph_multiple-home', 'figure'),
    [dash.dependencies.Input('selected_variable-home', 'value'),
            dash.dependencies.Input('my-slider-home', 'value')])
def update_output(variable, days):
    trace = []
    for index in range(0,days):
        day = first_ten_days.day.unique()[index]
        df = first_ten_days.loc[first_ten_days.day == day,variable]
        trace.append(dict(x=df.index.time,
                                y=df,
                                name=day,
                                mode='lines',
                                line=dict(width=1,
                                          color=line_cols[index])
                                ))
    # figure = {"data": first_day_lines, "layout": layout_lines}
    return {"data": trace,
            "layout": dict(font=dict(family='Aleo'),
                            yaxis={"title": "Kilowatt per minute"},
                            title={'text':"Power consumption subset - {}".format(power_vars_dict[variable])},
                            xaxis={"showgrid": False,
                                    'automargin' : True,},
                            paper_bgcolor='#ffefef',
                            plot_bgcolor='#e6e1e1')}



@app.callback(Output('hist-graph_4-home', 'figure'), [Input('selected_variable_hist_final-home', 'value')])
def update_graph(selected):
    trace = go.Histogram(x=imputed_2_lags_outliers_aware[selected], opacity=0.5,
                        marker={"color": "#d95f02"},
                         xbins={"size": 0.5} )
    layout = go.Layout(title="{} distribution - final attempt".format(full_vars_dict[selected]),
                        xaxis={"title": "{}".format(full_vars_measure[selected]), "showgrid": False},
                       yaxis={"title": "Number of observations", "showgrid": False},
                       paper_bgcolor='#ffefef',
                       plot_bgcolor='#e6e1e1' )
    figure1 = {"data": [trace], "layout": layout}


    return figure1






@app.callback(Output('heat_var_x_home', 'options'),
    [Input('heat_var_y_home', 'value')])
def update_output(value):
    return [i for i in heat_options if i['value'] != value]

@app.callback(Output('heat_var_y_home', 'options'),
    [Input('heat_var_x_home', 'value')])
def update_output(value):
    return [i for i in heat_options if i['value'] != value]


@app.callback(Output('heat_graph_home', 'figure'),
    [Input('heat_var_x_home', 'value'), Input('heat_var_y_home', 'value')])
def update_figure(heat_var_x, heat_var_y):
    dff = scores_final.loc[:,[heat_var_x, heat_var_y, "avg_mse"]]
    dff = dff.sort_values([heat_var_y, heat_var_x])

    x_var = [i['label'] for i in heat_options if i['value'] == heat_var_x][0]
    y_var = [i['label'] for i in heat_options if i['value'] == heat_var_y][0]
    trace = go.Heatmap(x=dff[heat_var_x].values,
                        y=dff[heat_var_y].values,
                        z=dff['avg_mse'],
                        colorscale='Electric',
                        colorbar={"title": "MSE"},
                        showscale=True)
    return {"data": [trace],
            "layout": go.Layout(title=f"Test set MSE on 210 days - {y_var} vs {x_var}",
                                # xaxis={"title": x_var, "tickmode": "array",},
                                # yaxis={"title": y_var, "tickmode": "array",},
                                xaxis={"title": x_var,},
                                yaxis={"title": y_var,},
                                paper_bgcolor='#ffefef',
                                plot_bgcolor='#e6e1e1' )}




@app.callback(
    dash.dependencies.Output('slider_ts_annotation_home', 'children'),
    [dash.dependencies.Input('time_series_slider_home', 'value')])
def update_output(value):
    return 'Plotting {} best models'.format(value)


@app.callback(
    dash.dependencies.Output('check_predictions_home', 'figure'),
    [dash.dependencies.Input('time_series_slider_home', 'value')])
def update_output(comps):

    main_set = dirs[0][1]

    trace = [return_ts(main_set, variable = 'y_hat',
                                    name = "Final model", color = '#406a85', width = 3),
            return_ts(main_set, variable = 'y_true',
                                    name = "Ground truth", color = '#03070a', width = 0.3),]
    for i in range(1,comps+1):
        trace.append(return_ts(dirs[i][1], variable = 'y_hat',
                                name = "comp model № %d" % i, color = '#50a358', width = 0.5))
        # print(dirs[i])
        # print(i)

    return {"data": trace,
            "layout": dict(font=dict(family='Aleo'),
                            yaxis={"title": "Kilowatt per day", "showgrid": False},
                            title={'text':"Final model prediction and %d next best models" % comps},
                            xaxis={"showgrid": False,
                                    'automargin' : True,},
                            paper_bgcolor='#ffefef',
                            plot_bgcolor='#e6e1e1')}
