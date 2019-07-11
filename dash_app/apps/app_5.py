import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table

from app import app, page_style
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import os
import itertools
# table_1_index = pd.read_csv('data/table_1_index.csv')

final_predictions_df = pd.read_csv('../final_model/final_prediction_df.csv')

dirs = [x[2] for x in os.walk('data/model/predictions')]
dirs = list(itertools.chain(*dirs))
dirs = [(int(i.split("_")[0]), 'data/model/predictions/' + i) for i in dirs]
dirs.sort(key = lambda x: x[0])
dirs = dirs[0:10]
# dirs = [os.path.join(x[0], i) for x in dirs for i in x[1]]
# print(len(dirs))
def return_ts(path, variable = 'y_hat', name = "", color = '#406a85', width = 1):
    data = pd.read_csv(path)
    data['Datetime'] = pd.to_datetime(data.loc[:,'Datetime'], infer_datetime_format=True)
    data = data.set_index('Datetime')

    trace = go.Scatter(
            x=data.index,
            y=data.loc[:, variable],
            name= name,
            mode='lines',
            line=dict(width=width, color=color))
    return trace

def return_line_trace(data, idx, train_or_test = 'train', color = '#252c36', width = 1):
    trace = go.Scatter(
            x=data.loc[data.id == idx,'row'],
            y=data.loc[data.id == idx, train_or_test],
            name= "%.2f" % round(data.loc[data.id == idx,'mse'].values[0],2) + " " + train_or_test,
            mode='lines',
            line=dict(width=width, color=color))
    return trace

line_cols = ['#e41a1c','#377eb8','#4daf4a',
            '#984ea3','#ff7f00','#ffff33',
            '#a65628','#f781bf','#999999']

model_scores = pd.read_csv('data/model/model_scores.csv', index_col = None)
model_scores = model_scores.drop(model_scores.columns[0], axis=1)

scores_final = pd.read_csv('data/model/scores_final.csv')
scores_final = scores_final.drop(scores_final.columns[0], axis=1)
# print(model_scores.head())

# print(len(range(1, 10)))

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

heat_options = [
                # {'value' : 'cols_drop', 'label' : 'Columns dropped'},
                {'value' : 'dropout', 'label' : 'LSTM dropout'},
                {'value' : 'lag_size', 'label' : 'Lag size'},
                {'value' : 'n_drop', 'label' : 'Number of variables dropped'}]

layout = html.Div(style=page_style, children=[
    html.Br(),
    # dcc.Link('Go to App 2', href='/apps/app2'),
    html.H1(
        children='Model',
        style={
            'textAlign': 'center',

        }
    ),


    html.H4(
        children='Step 7: Model evaluation.',
        style={
            'textAlign': 'left',
            # 'color': colors['text']
        }
    ),

    dcc.Markdown("""
        Our final LSTM model characteristics:
        - test set MSE of 21.45 on 210 days
        - three variables dropped - Sub meters 1-3
        - dropout - 0.7
        - using 26 previous days for prediction

        Here are some general observations on top models trained:
        - best models drop either sub meter 3 or all three sub meters
        - using around 20+ previous days for prediction
        - dropout is generally in the 0.1-0.4 range

        From the heatmap below we can see the following better models:
        - highest dropout rate along with more variables to drop
        - higher lags (10-30) with few variables dropped
        - 5-10 lag along all dropout rates

        """),
        html.Br(),

        html.Div([
                html.Div([
                     dcc.Dropdown(id="heat_var_y", multi=False,
                                  value=heat_options[0]["value"],
                       options=heat_options),
                html.H6("Select Y axis")],
                                className="column",
                             style={"display": "inline-block'",
                                     "width": "45%",
                                     'background' : '#ffefef',
                                     'textAlign': 'center'}),
                 html.Div([
                     dcc.Dropdown(id="heat_var_x", multi=False,
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
                 html.Div([dcc.Graph(id="heat_graph",
                                        style={"margin-top": "7rem",
                                                "margin-right": "auto",
                                                "margin-left": "auto",
                                                "width": "80%"
                                                })],
                  className="row")

                     ]),

         dcc.Markdown("""
                From the graph below we can see how our model outperforms
                the other best nine models. The difference in 0.13 MSE might not
                be that significant especially given that it's squared
                across 210 days and models are quite similar.

                Next we examine our model to the worst ones. We see that the
                difference is quite drastic.
             """),

         html.Div(dcc.Graph(id="training_graph", clear_on_unhover=True,
                            figure = figure_models_training),
                                 style={"width": "100%",
                                 },

             className="container"),

         html.Div(dcc.Graph(id="training_graph_worst", clear_on_unhover=True,
                            figure = figure_models_training_worst),
                                 style={"width": "100%",
                                 },

             className="container"),

         dcc.Markdown("""
                We can see that the next best models are basically identical in
                their accuracy. We achived quite a smooth prediction vector, model
                doesn't overfit but also follows the ground truth data when there
                are sudden dips in consumption.
             """),


         html.Div(
             [
             html.Div([
                     dcc.Slider(id='time_series_slider', min=0, max=len(dirs), step=1, value=2),
                     html.Div(id='slider_ts_annotation')]),
             html.Div(
                     [dcc.Graph(id="check_predictions")]),
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
                We did indeed limit our analysis to LSTM so one of the next steps
                would be to expand the range of models.

                In addition we could further train models by expanding training procedures.
                We saw a slow increase in accuracy over time so maybe increasing the number
                of training epochs would further improve the model.

                In terms of deployment we didn't quite organize our model components
                but basically these are the steps we would take:
                - Bring all transformers and anomaly detection models in one place
                - We know the number of lags our model requires to make predictions -
                    we even stored the learning hyperparameters such as number of lags
                    and variables to drop - therefore we can feed them as inputs to
                    deployed model
                - Design a REST api and deploy it using Django REST or Flask
                    - We feed data in json format
                    - Return payloads in either json or raw text format since our output
                        is only seven seven days ahead.
                - Push designed Flask/Django app to github
                - Containerize application using Docker automated builds
                - Deploy docker container(s) using either docker compose (for simple ad hoc models)
                    or Kubernetes

                Deployment and its design will strongly depend on the purpose of the application,
                its use and in what environments it's required. We will use Kubernetes for
                more reliable scalable services where high amount of traffic is expected.

                Below is the vector of seven next days in power consumption.
                ```
             """),

             dash_table.DataTable(
                     id='final_vector',
                     columns=[{"name": i, "id": i} for i in final_predictions_df.columns],
                     data=final_predictions_df.to_dict("rows"),

                 ),



        ])


@app.callback(Output('heat_var_x', 'options'),
    [Input('heat_var_y', 'value')])
def update_output(value):
    return [i for i in heat_options if i['value'] != value]

@app.callback(Output('heat_var_y', 'options'),
    [Input('heat_var_x', 'value')])
def update_output(value):
    return [i for i in heat_options if i['value'] != value]


@app.callback(Output('heat_graph', 'figure'),
    [Input('heat_var_x', 'value'), Input('heat_var_y', 'value')])
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
    dash.dependencies.Output('slider_ts_annotation', 'children'),
    [dash.dependencies.Input('time_series_slider', 'value')])
def update_output(value):
    return 'Plotting {} best models'.format(value)


@app.callback(
    dash.dependencies.Output('check_predictions', 'figure'),
    [dash.dependencies.Input('time_series_slider', 'value')])
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
