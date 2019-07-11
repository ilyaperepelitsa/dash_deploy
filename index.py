import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import app_1, app_2, app_3, app_4, app_5, home
from app import page_style

nav_style = {"margin" : "1em", 'text-decoration': 'none',
            'fontSize' : '1.5em', 'color' : '#0d669b'}

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Link('Home', href='/', style=nav_style),
    dcc.Link('Load Data', href='/loading', style=nav_style),
    dcc.Link('Missing Values', href='/imputing', style=nav_style),
    dcc.Link('Exploring Data', href='/eda', style=nav_style),
    dcc.Link('Model', href='/model', style=nav_style),
    dcc.Link('Evaluation', href='/model_eval', style=nav_style),
    html.Div(id='page-content'),
]
,style={'marginLeft' : 'auto',
'marginRight' : 'auto',
"width": "80%",}
)
#


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/loading':
        return app_1.layout
    elif pathname == '/imputing':
        return app_2.layout
    elif pathname == '/eda':
        return app_3.layout
    elif pathname == '/model':
        return app_4.layout
    elif pathname == '/model_eval':
        return app_5.layout
    elif pathname == '/':
        return home.layout
    else:
        return '404'

# server = app.run_server(debug=False)
# if __name__ == '__main__':
#     app.run_server(debug=False)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=False)
