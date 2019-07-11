layout = html.Div([
    html.Div([html.H1("Animation Plot")],style={'textAlign': "center"}),
    html.Div([dcc.Graph(id="my-graph"),dcc.Interval(id='interval-component',interval=800,n_intervals=0, disabled=True),]),
    html.Div([html.Button(id='play-button', children="Play", className="six columns", n_clicks=0),
              html.Button(id='stop-button', children="Stop", className="six columns", n_clicks=0)],className="row",
             style={"width": "40%", "margin-left": "auto", "margin-right": "auto","display": "block",})
], className="container")


@app.callback(
    [Output('my-graph', 'figure'),
     Output('interval-component', 'disabled')],
    [Input('interval-component', 'n_intervals'),
     Input("play-button", 'n_clicks'), Input('stop-button', 'n_clicks')],
)
def update_figure(interval,play,stop):
    dff = df[df['Indicator'] == "AIDS-related deaths"]
    ctx = dash.callback_context
    trace = [go.Scatter(x=dff['Time Period'],y=dff['Data Value'],mode='lines',name="No of Deaths",
                        marker={"color": "#4daf4a"})]

    figure = {"data": trace,
              "layout": go.Layout(title="AIDS Related Deaths",height=600,showlegend=False,
                                  xaxis={"title": "Date"},yaxis={ "title": "AIDS Related Deaths (Number) ",
                                                                  "range": [50, 500000],},hovermode="closest",),}
    trace.append(go.Scatter(x=[dff['Time Period'][interval]],y=[dff['Data Value'][interval]],mode='markers',
                            name="Animation trace",marker={"size": 25, "color": "#e41a1c", "symbol": "star-triangle-up"}))
    if ctx.triggered[0]['prop_id'].split('.')[0] == 'play-button' and ctx.triggered[0]['prop_id'].split('.')[0] == 'interval-component':
        disable = False
    elif ctx.triggered[0]['prop_id'].split('.')[0] == 'stop-button' or interval >= 25 :
        disable = True
    elif play == 0 and stop == 0:
        disable = True
    else:
        disable = False

    return figure, disable

















    # layout_lines = dict(font=dict(family='Aleo'), yaxis={"title": "Kilowatt per minute"},
    #             title={'text':'Global power consumption, consumption in kitchen, laundry, by water heater and AC'},
    #                                         xaxis={"showgrid": False,
    #                                                 'automargin' : True,},
    #                                         paper_bgcolor='#ffefef',
    #                                         plot_bgcolor='#e6e1e1')
