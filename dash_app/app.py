import dash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://codepen.io/ilyaperepelitsa/pen/QXJNVJ.css']
# external_stylesheets = ['https://codepen.io/ilyaperepelitsa/pen/QXJNVJ.css']

colors = {
    'background': '#ffefef',
    'text': '#424856'
}


page_style = {'backgroundColor': colors['background'],
                            "width": "100%",
                            'marginLeft' : 'auto',
                            'marginRight' : 'auto',}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True
