from dash import Dash, html, callback
import dash_bootstrap_components as dbc
import dash

dash.register_page(__name__, path="/", title="CRACO Candidate HOME")

layout = dbc.Container(
    [
        html.P("This is an interactive tool for inspecting candidates")
    ]
)
