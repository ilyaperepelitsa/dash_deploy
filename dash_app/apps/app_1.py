import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from app import page_style
import dash_table

table_1_index = pd.read_csv('data/table_1_index.csv')
# table_missing_vals = pd.read_csv('data/missing_vals.csv')

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


layout = html.Div(style=page_style, children=[
    # dcc.Link('Go to App 2', href='/apps/app2'),
    html.Br(),
    html.H1(
        children='Dataset Preparation',
        style={
            'textAlign': 'center',
            # 'color': colors['text'],
        }
    ),

    html.H4(
        children='Step 1: Importing libraries.',
        style={
            'textAlign': 'left',
            # 'color': colors['text']
        }
    ),
    dcc.Markdown("""
        ```py
        from matplotlib import pyplot as plt
        import os
        import pandas as pd
        import re
        import numpy as np
        from joblib import dump, load
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        """),

    dcc.Markdown("""
        We os.walk the **data** directory and put all the files into a list.
        ```py
        dirs = [(x[0] , x[2]) for x in os.walk('data') if len(x[0].split("/")) > 2 ]
        dirs = [os.path.join(x[0], i) for x in dirs for i in x[1]]
        ```

        There are some hidden directories that we filter out because we don\'t want
        to deal with dropping duplicates at a later stage.
        ```py
        {'data/2008/10/.DS_Store', 'data/2008/10/.ipynb_checkpoints/10-checkpoint'}
        ```
        We filter the paths with regular expressions:
        ```py
        dirs = [i for i in dirs if re.match('data/\d{4}/\d+/\d+',i)]
        ```
        And concatenate the **`pd.read_csv`** of those paths into a DataFrame
        ```py
        data = pd.concat([pd.read_csv(i, na_values="?") for i in dirs])
        ```
        """),

    dcc.Markdown("""
        We create index and examine difference between timestamps and their lags
        to make sure that `freq=60s`
        ```py
        data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
        data['Datetime_1'] = data.groupby(['Date'])['Datetime'].shift(1)
        data['lag_time'] = (data['Datetime'] - data['Datetime_1']).dt.seconds
        data['lag_time'].describe()
        ```

        The frequency is indeed 1 minute meaning that all data is continuous and
        we loaded everything correctly.

        ```py
        count    1526602.0
        mean          60.0
        std            0.0
        min           60.0
        25%           60.0
        50%           60.0
        75%           60.0
        max           60.0
        Name: lag_time, dtype: float64
        ```
        We can drop intermediate variables in order to preserve the RAM.

        Since we specified `na_values="?"` at `pd.read_csv()` we can convert all columns
        to numeric because the `?` have been interpreted as `np.nan`

        ```py
        data = data.set_index('Datetime')
        data = data.drop(["Date", "Time", "Datetime_1", "lag_time"], axis = 1)
        data = data.apply(pd.to_numeric)
        ```
        """),

    dcc.Markdown("""
        We print the dataset and examine it for missing values in the next step.
        """),
    # generate_table(table_1_index),
    dash_table.DataTable(
            id='table_app_1_1',
            columns=[{"name": i, "id": i} for i in table_1_index.columns],
            data=table_1_index.to_dict("rows"),
        ),





    # dcc.Graph(
    #     id='example-graph-2',
    #     figure={
    #         'data': [
    #             {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
    #             {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
    #         ],
    #         'layout': {
    #             'plot_bgcolor': colors['background'],
    #             'paper_bgcolor': colors['background'],
    #             'font': {
    #                 'color': colors['text']
    #             }
    #         }
    #     }
    # )
])

# if __name__ == '__main__':
#     app.run_server(debug=True)
