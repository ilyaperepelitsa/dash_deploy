import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table

from app import app, page_style
from dash.dependencies import Input, Output
import plotly.graph_objs as go
# table_1_index = pd.read_csv('data/table_1_index.csv')

line_cols = ['#e41a1c','#377eb8','#4daf4a',
            '#984ea3','#ff7f00','#ffff33',
            '#a65628','#f781bf','#999999']


first_day = pd.read_csv('data/first_day.csv')
first_day['Datetime'] = pd.to_datetime(first_day.loc[:,'Datetime'], infer_datetime_format=True)
first_day = first_day.set_index('Datetime')

first_ten_days = pd.read_csv('data/first_ten_days.csv')
first_ten_days['Datetime'] = pd.to_datetime(first_ten_days.loc[:,'Datetime'], infer_datetime_format=True)
first_ten_days = first_ten_days.set_index('Datetime')


data_original = pd.read_csv('./data_original_sampled.csv')
data_original['Datetime'] = pd.to_datetime(data_original.loc[:,'Datetime'], infer_datetime_format=True)
data_original = data_original.set_index('Datetime')


imputed_10_lags = pd.read_csv('../imputed_10_lags_sampled.csv')
imputed_10_lags['Datetime'] = pd.to_datetime(imputed_10_lags.loc[:,'Datetime'], infer_datetime_format=True)
imputed_10_lags = imputed_10_lags.set_index('Datetime')


imputed_2_lags_outliers_aware = pd.read_csv('../imputed_2_lags_outliers_aware_sampled.csv')
imputed_2_lags_outliers_aware['Datetime'] = pd.to_datetime(imputed_2_lags_outliers_aware.loc[:,'Datetime'], infer_datetime_format=True)
imputed_2_lags_outliers_aware = imputed_2_lags_outliers_aware.set_index('Datetime')

describe_original = pd.read_csv('data/eda/describe_original.csv')
describe_2 = pd.read_csv('data/eda/describe_2.csv')
describe_10 = pd.read_csv('data/eda/describe_10.csv')
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
# print(imput_10_resampled_daily.columns)
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

# go.Scatter(x=dataframe['data'].index,
#                         y=dataframe['data'][variable],
#                         name=dataframe['display_name'],
#                         mode='lines',
#                         marker={'size': 3, "opacity": 0.7,
#                                 "line": {'width': 0.2}},
#                         )
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




final_daily = pd.read_csv('data/resampled/final_daily.csv')
final_daily['Datetime'] = pd.to_datetime(final_daily.loc[:,'Datetime'], infer_datetime_format=True)
final_daily = final_daily.set_index('Datetime')
final_daily["Rest"] = final_daily['Global_active_power'] - \
                            final_daily.loc[:,['Sub_metering_1',
                                                    'Sub_metering_2',
                                                    'Sub_metering_3']].sum(axis = 1)
# print(imput_10_resampled_daily.columns)
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




layout = html.Div(style=page_style, children=[
    html.Br(),
    # dcc.Link('Go to App 2', href='/apps/app2'),
    html.H1(
        children='Exploratory Data Analysis',
        style={
            'textAlign': 'center',

        }
    ),


    html.H4(
        children='Step 3: Visualizing transformations.',
        style={
            'textAlign': 'left',
            # 'color': colors['text']
        }
    ),

    dcc.Markdown("""
        In the missing value visualization we presented data in a certain transformed format.

        We will list the variables and transformations for visualization purposes only -
        we will be performing predictions in the original format or the one fitting the models (standardization,
         normalization etc.)
         - Global Active Power - Seems to be Kilowatt/hr readings of current consumption. Given that these are one-minute
         sample estimates - if we divide them by 60 we will get a minute-sample estimate
         of that minute power consumption. If we want to estimate an hour consumption we can resample by day
         and sum the estimates - this wiil give an estimate of total daily consumption in Kilowatts.
         - Sub metering 1, 2 and 3 - these are in watt per minute so in order to compare to Global active
         power we will need to multiply by 1000 to get Kilowatt per minute estimates. The amount not covered
         by three sub meters is consumption in other areas of the apartment.
         - Voltage and Global intensity - we average these after resampling since adding up these measurements is
         meaningless from physics perspective.


         We plot the weekly total power consumption in order to examine such consumption across 3 submeters
         and the **Rest** - Global consumption minus all three submeters combined.

        """),
        html.Br(),

        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="hist-graph", clear_on_unhover=True,
                                figure = {"data": init_hist_traces, "layout": layout}),]),

            ],
            className="container"),

        dcc.Markdown("""
             We can see clear seasonality in weekly consumption. It seems like there is a dropoff
             in consumption every August - maybe our residents are going on vacation every August.
             An interesting detail emerges - in August 2009 the water heater and AC consumption
             doesn't drop off to the levels of the previous year. Two hypotheses emerge:
             - the AC was installed after the summer of 2008
                - it doesn't get turned off completely during vacations
             - the AC was turned off during the summer vacation of 2008 and wasn't in later years

             Also the power consumption doesn't drop off to 0 during vacation periods - either
             some appliances work in idle mode and consume power or some residents stay
             at the apartment while the rest leave.

            """),
        html.Br(),
        dcc.Markdown("""
            We further examine the minute-sampled data directly. We applied the transformations
            in order to examine our data of two kinds of meters in the same dimensions.

            We can see a resemblence of some story with this graph. The following is a speculation.
            - The laundry has something turning on every hour and a half. Most likely it's the refrigerator
            working in burst of cooling - it probably has a thermometer and every time the temperature
            rises above the setting it reacts by cooling.
            - For some reason the kitchen consumption stays at zero but we know that it does add up to weekly
            total of ~8 Kilowatt. If Laundry consumption is a consistent pattern it makes sense for Kitchen to be that
            low since it sums to less energy than Laundry.
            - Water heater and AC are the most interesting daily pattern. It seems like simeone gets up at 5-6 AM, uses some
            hot water (brushing teeth) or turns the AC on for half an hour. We see the second
            burst at 7 AM - maybe the rest of the residents wake up and start using water and/or turn the AC on.
            Maybe water is a better guess but we can also see smaller bursts in this meter through he night -
            seems like water heater is keeping the water cold in the same patter as we speculated
            with the refrigerator. It drops off at 14:30 - seems like these residents are leaving at that time
            until someone gets back home around 5 PM. Maybe the first resident comes back home and takes the shower
            or turns the AC back on. The last burst at this meter is at 11 PM - maybe the dishwasher is using some hot water.
            - Residual ("submeter 4") consumption bursts in the evening and from midnight till 5 AM - something is
            consuming power. Maybe a personal computer gets turned on and/or portable devices are being charged.
            """),

        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="line-graph", clear_on_unhover=True,
                                figure = {"data": first_day_lines, "layout": layout_lines}),],
                                style={"width": "100%"}),
            ],
            className="container"),



        html.Br(),

        dcc.Markdown("""
            Interacting with the exhibit below (by changing the number of days to display and
            variables to examine) we see that patterns do hold somewhat schedule-wise, however
            it ins't very consistent - schedule isn't very specific. However we can see that
            certain appliances do behave generally in  the way we speculated before.

            """),

        html.Div(
            [
            html.Div(dcc.Dropdown(id="selected_variable", multi=False,
                         value=power_vars[0]["value"],
                         options=power_vars)),
            html.Div([
                    dcc.Slider(id='my-slider', min=2, max=len(first_ten_days.day.unique()), step=1, value=2),
                    html.Div(id='slider-output-container')]),
            html.Div(
                    [dcc.Graph(id="line_graph_multiple")]),
            ],
            className="container"),
        html.H4(
            children='Step 4: Examining variables.',
            style={
                'textAlign': 'left',
                # 'color': colors['text']
            }
        ),

        dcc.Markdown(dangerously_allow_html=True,
            children=("""
            We will visualize provided variables and explore some relationships
            between them. We can also speculate that in general submeters data
            could probably turn out to be somewhat redundant since Global power
            consumption is simply a sum of three variables and the fourth one -
            the difference between the sum of three and global. Since we are predicting the
            forecasted values the "four" submeters of previous observation will basically
            be y<sub>t-1</sub>.

            We examine three datasets - the original one and the two imputed datasets that we created.
            There is a significant problem that we skipped. There are some outliers
            that don't make sense from the phisics perspective - the power consumption ones in particular.
            There is also a problem with global intensity. We didn't conduct thorough
            research on power grids but from basic researc it seems like such observations
            are unlikely. This is a major problem for our imputed missing values
            since the process relies on distributions of observed data. We will have to
            redo the part.

            All three tables are presented below.
            """)),
        html.H6(
            children='Original.',
            style={
                'textAlign': 'center',
                # 'color': colors['text']
            }
        ),

        dash_table.DataTable(
                id='describe_original',
                columns=[{"name": i, "id": i} for i in describe_original.columns],
                data=describe_original.to_dict("rows"),

            ),
        html.H6(
            children='Two-lags imputed.',
            style={
                'textAlign': 'center',
                # 'color': colors['text']
            }
        ),

        dash_table.DataTable(
                id='describe_2',
                columns=[{"name": i, "id": i} for i in describe_2.columns],
                data=describe_2.to_dict("rows"),
            ),

        html.H6(
            children='Ten-lags imputed.',
            style={
                'textAlign': 'center',
                # 'color': colors['text']
            }
        ),

        dash_table.DataTable(
                id='describe_10',
                columns=[{"name": i, "id": i} for i in describe_10.columns],
                data=describe_10.to_dict("rows"),
            ),
        dcc.Markdown(dangerously_allow_html=True,
            children=("""
            Due to observed distributions we now have imputed negative power consumption values.

            We will proceed to examination of original distributions.


            **Since the initial version we modified this example to include only a sample of
            data for these charts - 100k rows were chosen as a representation in order to
            not load each 100Mb dataset into Dash**
            """)),


        html.Div(
            [
            html.Div(dcc.Dropdown(id="selected_variable_hist", multi=False,
                                    value=full_vars[0]["value"],
                         options=full_vars),
                            className="row",
                            style={"display": "inline",
                                    "width": "30%",
                                    "margin-left": "auto",
                                    "margin-right": "auto"}),
            html.Div(
                    [dcc.Graph(id="hist-graph_2", clear_on_unhover=True)]),
            html.Div(
                    [dcc.Graph(id="hist-graph_3", clear_on_unhover=True)]),
            ],
            className="container"),

        html.H4(
            children='Step 5: Outlier-aware missing value imputing.',
            style={
                'textAlign': 'left',
                # 'color': colors['text']
            }
        ),

        dcc.Markdown(dangerously_allow_html=True,
            children=("""
            Since we identified a significant issue with missing value replacement we
            introduce a new algorithm for missing values replacement.

            First we load original data
            ```py
            data_original = pd.read_csv('data_original.csv')
            data_original['Datetime'] = pd.to_datetime(data_original.loc[:,'Datetime'], infer_datetime_format=True)
            data_original = data_original.set_index('Datetime')
            ```

            We select not-null data and fit an `IsolationForest` model on it to identify
            outliers. We examine the set with outliers:
            """)),

        dash_table.DataTable(
                id='describe_original_outliers',
                columns=[{"name": i, "id": i} for i in describe_original_outliers.columns],
                data=describe_original_outliers.to_dict("rows"),
            ),

        dcc.Markdown(dangerously_allow_html=True,
            children=("""
            And the set without outliers:
            """)),

        dash_table.DataTable(
                id='describe_original_no_outliers',
                columns=[{"name": i, "id": i} for i in describe_original_no_outliers.columns],
                data=describe_original_no_outliers.to_dict("rows"),
            ),
        dcc.Markdown(dangerously_allow_html=True,
            children=("""
            We can clearly see that for columns identified previously the values are quite high.

            We aggresively set them to `np.nan`, join it with the original data therefore
            significantly increasing the number of missing values and applying the iterative imputer
            as we did previously.

            ```py
            not_null_original.loc[preds_2 < 0,:] = np.nan
            new_data = pd.concat([data_original[data_original.isnull().any(axis=1)], not_null_original])


            pct_vars = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
            pct_vars_change = [i + "_pct_change" for i in pct_vars]
            other = [i for i in new_data.columns if i not in pct_vars]
            num_cols = new_data.columns

            df_lag = new_data[pct_vars]
            df_lag[pct_vars_change] = new_data[pct_vars].pct_change()
            df_lag[other] = new_data[other]

            lag_cols = df_lag.columns
            LAG = 3
            for i in range(1, LAG):
                for col in lag_cols:
                    df_lag[col + '__' + str(i)] = df_lag[col].shift(i)

            df_lag.replace([np.inf, -np.inf], np.nan, inplace = True)

            ii = IterativeImputer()
            data_imputed = ii.fit_transform(df_lag)
            from joblib import dump, load
            dump(ii, 'iterative_imputer_2_lags_with_outliers.joblib')

            imputed_2_lags = pd.DataFrame(data_imputed, columns = df_lag.columns, index = df_lag.index)
            imputed_2_lags = imputed_2_lags.loc[:,num_cols]
            imputed_2_lags.to_csv("imputed_2_lags_outliers_aware.csv")

            ```
            """)),

            dcc.Markdown(dangerously_allow_html=True,
                children=("""
                Now we examine the description of this final dataset. There are still substantial problems
                with submeter data - the minimum values are negative. We will consider
                dropping these variables as redundant. For now we will examine the variable
                distribution.
                """)),


            dash_table.DataTable(
                    id='describe_final',
                    columns=[{"name": i, "id": i} for i in describe_final.columns],
                    data=describe_final.to_dict("rows"),
                ),

            html.Div(
                [
                html.Div(dcc.Dropdown(id="selected_variable_hist_final", multi=False,
                                        value=full_vars[0]["value"],
                             options=full_vars),
                                className="row",
                                style={"display": "inline",
                                        "width": "30%",
                                        "margin-left": "auto",
                                        "margin-right": "auto"}),
                html.Div(
                        [dcc.Graph(id="hist-graph_4", clear_on_unhover=True)]),
                ],
                className="container"),

            dcc.Markdown(dangerously_allow_html=True,
                children=("""
                We resample the final dataset using the mentioned transformation technique in
                order to estimate the daily total Kilowatt consumption.

                We use the following function.

                ```py

                def transform_resample(data, frequency = 'd'):
                    transformed_data = pd.DataFrame(data.resample(frequency).apply(lambda x: pd.Series([(x['Global_active_power']/60).sum(),
                        (x['Global_reactive_power']/60).sum(),
                        x['Voltage'].mean(),
                        x['Global_intensity'].mean(),
                        (x['Sub_metering_1']/1000).sum(),
                        (x['Sub_metering_2']/1000).sum(),
                        (x['Sub_metering_3']/1000).sum()])))

                    transformed_data = transformed_data.rename(columns={i: x for i,x in
                                                                zip(transformed_data.columns,
                                                                data.columns)})
                    return transformed_data

                ```
                """)),

                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="hist-final", clear_on_unhover=True,
                                        figure = {"data": final_daily_traces, "layout": final_layout}),]),

                    ],
                    className="container"),



        ])


@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('my-slider', 'value')])
def update_output(value):
    return 'Comparing {} days of power consumption'.format(value)


@app.callback(
    dash.dependencies.Output('line_graph_multiple', 'figure'),
    [dash.dependencies.Input('selected_variable', 'value'),
            dash.dependencies.Input('my-slider', 'value')])
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


@app.callback([Output('hist-graph_2', 'figure'), Output('hist-graph_3', 'figure')], [Input('selected_variable_hist', 'value')])
def update_graph(selected):
    trace = go.Histogram(x=data_original[selected], opacity=0.5,
                        marker={"color": "#d95f02"},
                         xbins={"size": 0.5} )
    layout = go.Layout(title="{} distribution - original data".format(full_vars_dict[selected]),
                        xaxis={"title": "{}".format(full_vars_measure[selected]), "showgrid": False},
                       yaxis={"title": "Number of observations", "showgrid": False},
                       paper_bgcolor='#ffefef',
                       plot_bgcolor='#e6e1e1' )
    figure1 = {"data": [trace], "layout": layout}

    trace2 = go.Histogram(x=imputed_10_lags[selected], opacity=0.5,
                        marker={"color": "#984ea3"},
                         xbins={"size": 0.5} )
    layout2 = go.Layout(title="{} distribution - lag 10 data".format(full_vars_dict[selected]),
                        xaxis={"title": "{}".format(full_vars_measure[selected]), "showgrid": False},
                       yaxis={"title": "Number of observations", "showgrid": False},
                       paper_bgcolor='#ffefef',
                       plot_bgcolor='#e6e1e1' )
    figure2 = {"data": [trace2], "layout": layout2}

    return figure1, figure2


@app.callback(Output('hist-graph_4', 'figure'), [Input('selected_variable_hist_final', 'value')])
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
# "title": "{}".format(full_vars_measure[selected])
