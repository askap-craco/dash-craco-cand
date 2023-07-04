from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import dash

import pandas as pd
import numpy as np
import plotly.express as px

### choose from file directly, select beamfile
### if you know 

dash.register_page(__name__, path="/beam", title="CRACO File Loader")

layout = dbc.Container(
    html.H3("TEST")
)