from flask import Flask
from dash import Dash, html, callback

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

### long_callback related
from celery import Celery
from dash.long_callback import CeleryLongCallbackManager

from apputil import _header, _footer, load_password, init_sqlite

# set up flask server
server = Flask(__name__)

celery_app = Celery(
    __name__,
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)
LONG_CALLBACK_MANAGER = CeleryLongCallbackManager(celery_app)

app = Dash(
    __name__, server=server,
    url_base_pathname="/", use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    long_callback_manager=LONG_CALLBACK_MANAGER,
)

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
    server.run(host="0.0.0.0", port=8024, debug=True)
