from dash import html
import dash_bootstrap_components as dbc
import json

# load password pairs
def load_password(fname):
    with open(fname) as fp:
        pwd = json.load(fp)
    return pwd

# web layout
def _header():
    return dbc.NavbarSimple(
        brand="CRACO Candidate Inspection",
        brand_href="#", color="primary", dark=True,
    )

def _footer():
    return dbc.Container(
        [dbc.Row(html.P("ASKAP-CRACO @ 2023"))]
    )