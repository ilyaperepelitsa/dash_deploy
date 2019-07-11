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
        children='Step 6: Using LSTM to predict next 7 days.',
        style={
            'textAlign': 'left',
            # 'color': colors['text']
        }
    ),

    dcc.Markdown("""
        We will be using a basic LSTM model in order to estimate the next seven days of
        Global power consumption.

        Even though there are other models we could test, there are certain reasons to explore
        this particular technique.

        - We want to test how adding certain variables' lags affects performance
            of our model - univariate auto-regressions are quite limited in this case
            since we only look at lags of target variable and its moving average.
        - Walk-forward multistep linear models:
            - Take significant setup time for initial models
                - pipeline engineering for transformations and inverse transformations
                - pipeline engineering for basic estimators
                - incorporating model tuning into pipeline
            - After MVP pipelines are designed it takes significant time to train
                and test models to find optimal predictors
        - Advantage of neural networks is that they allow us to structure our outputs
        pretty clearly as arrays and we can predict the whole sequence based on model
        design. In addition this approach allows us to retrain models for new input-output
        shapes and with adaptive learning find models that are optimal. For example
        we may learn hyper-parameters in the training process in libraries like
        pytorch. We are currently more comfortable working with `keras` so we
        will be using this library.

        Ideally in production environment we would implement a multi-step model
            and test for a variety of estimators ranging from `LinearRegression`,
            `Ridge` and `Lasso` to `ElasticNet`, `BayesianRidge` and `PassiveAggressiveRegressor`

        We design our model in the following manner. Starting with imports.

        ```py
        import pandas as pd
        import numpy as np
        from matplotlib import pyplot as plt

        import keras
        from keras.layers import Dense
        from keras.models import Sequential
        import itertools
        from keras.layers import LSTM
        from keras.layers import Dropout

        from sklearn.preprocessing import MinMaxScaler

        ```
        Then we load data. Since the models were trained on the FloydHub cloud GPU
        instance - the path includes that.

        ```py
        final_daily = pd.read_csv('/floyd/input/electric_lstm/final_daily.csv')
        final_daily['Datetime'] = pd.to_datetime(final_daily.loc[:,'Datetime'], infer_datetime_format=True)
        final_daily = final_daily.set_index('Datetime')
        ```

        We structure our inputs and outputs in the following manner.

        We use the previous day's data including Global active power, lag it by
        a number of periods and drop training data containing missing values - for
        periods that don't have lag data.

        As target variable we use the current day's Global active power and forward shifts
        so that total number of days equals 7 : `t ... t+6`

        We use the following function to transform data in the manner described above:

        ```py

        def transform_dataset(data_in, x_lags=1, y_shift=1):
            n_vars = 1 if type(data_in) is list else data_in.shape[1]
            dff = pd.DataFrame(data_in)
            cols, names = list(), list()
            for i in range(x_lags, 0, -1):
                cols.append(dff.shift(i))
                names += [('%s(t-%d)' % (data_in.columns[j], i)) for j in range(n_vars)]
            for i in range(0, y_shift):
                cols.append(dff["Global_active_power"].shift(-i))
                if i != 0:
                    names.append("Global_active_power(t+%d)" % i)
                else:
                    names.append("Global_active_power(t)")
            data_out = pd.concat(cols, axis=1)
            data_out.columns = names
            data_out.dropna(inplace=True)
            return data_out

        ```

        We will be testing different options for lags ranging from 1 (previous day)
        up to 30 previous days in order to examine a broad range of options.


        Then we split our data to training and validation sets - we will be using
        210 days (30 weeks) of data as our validation set.

        ```py
        reframed = transform_dataset(final_daily, 20, 7)
        values = reframed.values

        n_train_time = 210
        train = values[:-n_train_time, :]
        test = values[-n_train_time:,

        train_X, train_y = train[:, :-7], train[:, -7:]
        test_X, test_y = test[:, :-7], test[:, -7:]

        ```

        Then we fit `MinMaxScaler` on our train X set and fit it on test X test
        as a standard procedure:
        - to optimize learning
        - to transform unseen data using previously observed data scale

        We then reshape our inputs into arrays.

        ```py
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_X = scaler.fit_transform(train)
        test_X = scaler.transform(test)

        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        ```

        Next we will train multiple models - a total of 108 in order to test combinations
        of hyperparameters of our model. Such parameters include:
        - lags 1 through 30 with a step of 5 - we want to test how our model works
            at various lags
        - 1 through 6 columns that we drop from final X test feature set - we will
            test how our models behave when we reduce the number of training features
        - 0.1 through 1 with a step of 0.3 - the dropout rate of our model

        We will use the following model:

        ```py

        from sklearn.metrics import mean_squared_error
        model_list = []
        for lag_size in range(1, 30, 5):
            for i, x in enumerate(final_daily.columns[1:]):

                cols_to_drop = list(final_daily.columns[1:][i : len(final_daily.columns[1:])])
                eframed = transform_dataset(final_daily.drop(cols_to_drop, axis = 1), lag_size, 7)
                values = reframed.values

                n_train_time = 210
                train = values[:-n_train_time, :]
                test = values[-n_train_time:,:]

                train_X, train_y = train[:, :-7], train[:, -7:]
                test_X, test_y = test[:, :-7], test[:, -7:]

                scaler = MinMaxScaler(feature_range=(0, 1))
                train_X = scaler.fit_transform(train)
                test_X = scaler.transform(test)

                train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
                test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

                for dropout in np.arange(0.1, 1, 0.2):
                    model = Sequential()
                    model.add(LSTM(150, input_shape=(train_X.shape[1], train_X.shape[2])))
                    model.add(Dropout(dropout))
                    model.add(Dense(7))
                    model.compile(loss='mean_squared_error', optimizer='adam')
                    history = model.fit(train_X, train_y,
                                        epochs=200, batch_size=200,
                                        validation_data=(test_X, test_y), verbose=0, shuffle=False)

                    yhat = model.predict(test_X)
                    avg_7_mse = mean_squared_error(test_y, yhat, multioutput='raw_values')
                    avg_mse = mean_squared_error(test_y, yhat)
                    print("MSE %f dropped %d lag %d" % (avg_mse, len(cols_to_drop), lag_size))

                    params = dict()
                    params['n_drop'] = len(cols_to_drop)
                    params['cols_drop'] = cols_to_drop
                    params['lag_size'] = lag_size
                    params['model'] = model
                    params['history'] = history
                    params['avg_mse'] = avg_mse
                    params['avg_7_mse'] = avg_7_mse
                    params['dropout'] = dropout

                    model_list.append(params)
        ```

        After we're done training we examine the best models and plot predicted and observed values of
        top models across the training set and validation set. We will also visualize different
        hyperparameters to learn more about model performance.

        *Upon examination of our results we found the following issues after fitting 180 models:
        - **eframed** - there is a typo in variable referencing and model was
            fit on data stored in the environment resulting in all variable drops
            not being executed at all*
        - we fit the scaler on the wrong dataset, retraining the model after another
            iteration of 108 models.

        We are restarting the previous kernel training session with errors corrected.
        We are going to evaluate the models in the next section.

        """),
        html.Br(),



        ])
