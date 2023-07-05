from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import dash

import matplotlib.pyplot as plt
from io import BytesIO
import base64

import pandas as pd
import numpy as np
import plotly.express as px

### choose from file directly, select beamfile
### if you know 

dash.register_page(__name__, path="/beam", title="CRACO File Loader")

data = np.random.randn(100, 50)

# @callback(
#     Output("filterbank_heat_map_div", "children"),
#     Input("")
# )
def _filterbank_heatmap_layout():
    fig = px.imshow(data, origin="lower")
    return fig


layout = dbc.Container(
    html.Div(dbc.Row([
        dbc.Col(
            html.Div(dcc.Graph(figure=_filterbank_heatmap_layout(), id="filterbank_heatmap_plot")),
            width=6
        ),
        dbc.Col([
            dbc.Row([
                "Here is something",
            ], align="center", className="h-100"),
        ], width=6,
        )
    ], ),
))