from flask import Flask
from dash import Dash, html, callback
from dash_extensions.enrich import Output, Input, html, DashProxy, LogTransform, DashLogger

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import dash_auth

from apputil import _header, _footer, load_password, init_sqlite

# set up flask server
server = Flask(__name__)
app = Dash(
    __name__, server=server,
    url_base_pathname="/", use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# authentication
# USERNAME_PASSWORD_PAIRS = load_password("statics/pwd.json")
# auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)

# make layout
app.layout = html.Div(
    [
        _header(),
        dash.page_container,
        _footer(),
    ]
)

if __name__ == "__main__":
    # init_sqlite()
    # server.run(host="0.0.0.0", port=8020, debug=True)
    # app.run_server(debug=True)
    server.run(host="0.0.0.0", debug=True)
