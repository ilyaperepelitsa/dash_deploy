
html.Div([
        dcc.Dropdown(id="selected_variable_hist", multi=False,
                                value=dropdown_options[0]["value"],
                     options=dropdown_options)],
                        className="row",
                        style={"display": "inline",
                                "width": "30%",
                                "margin-left": "auto",
                                "margin-right": "auto"}),
html.Div(
    [dcc.Graph(id="hist-graph", clear_on_unhover=True),], className="container"),

@app.callback(Output('hist-graph', 'figure'), [Input('selected_variable_hist', 'value')])
def update_graph(selected):
    trace = go.Histogram(x=nan_difference[selected], opacity=0.5, name="Difference",
                        marker={"line": {"color": "#25232C"}},
                         xbins={"size": 1} )
    layout = go.Layout(title="Age Distribution", xaxis={"title": "Age (years)", "showgrid": False},
                       yaxis={"title": "Count", "showgrid": False}, )
    figure2 = {"data": [trace], "layout": layout}

    return figure2
