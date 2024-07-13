import os
import subprocess

from dash import Dash, html, callback, dcc, Input, Output,State, ctx, dash_table
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash
import plotly.express as px
import plotly.graph_objects as go

from apputil import load_figure_from_disk

from astropy.coordinates import SkyCoord, Angle
from astropy import units

import pandas as pd
import numpy as np

from craco.craco_run import auto_sched

dash.register_page(__name__, path="/calib", title="CRACO calibration")

app = dash.get_app()

### layout for calibration shower...
def calib_allbeam_layout(query):
    sbid = query.get("sbid")

    titlerow = dbc.Row([
        dbc.Col(html.B("SBID"), width=1, align="center"),
        dbc.Col(dbc.Input(value=sbid, type="number", placeholder="enter the sbid...", id="cal_sbid"), width=2, align="center"),
        dbc.Col(width=4),
        dbc.Col(dbc.Button("Plot All Beams", id="cal_plot_allbeam_btn", ), width=2, align="center"),
    ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'})
    plotrow = dbc.Row([
        # dbc.Col(width=2),
        dbc.Col(id="cal_allbeam_plot_col", width=10),
        # dbc.Col(width=2)
    ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'})

    return dbc.Container(
        [titlerow, plotrow],
        style={'marginBottom': '1.0em', 'marginTop': '1.0em'}
    )

def calib_beam_layout(query):
    beam = query.get("beam")

    titlerow = dbc.Row([
        dbc.Col(html.B("BEAM"), width=1, align="center"),
        dbc.Col(dbc.Input(value=beam, type="number", placeholder="enter the sbid...", min=0, max=35, step=1, id="cal_beam"), width=2, align="center",),
        dbc.Col(width=4),
        dbc.Col(dbc.Button("Plot All Antennas", id="cal_plot_allant_btn", ), width=3, align="center"),
    ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'})
    plotrow = dbc.Row([
        # dbc.Col(width=2),
        dbc.Col(id="cal_allant_plot_col", width=12),
        # dbc.Col(width=2)
    ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'})

    return dbc.Container(
        [titlerow, plotrow],
        style={'marginBottom': '1.0em', 'marginTop': '1.0em'}
    )

def calib_ant_layout(query):
    ak = query.get("ak")

    titlerow = dbc.Row([
        dbc.Col(html.B("ANTENNA"), width=1, align="center"),
        dbc.Col(dbc.Input(value=ak, type="number", placeholder="enter the antenna number...", min=1, max=36, step=1, id="cal_ak"), width=2, align="center"),
        dbc.Col(width=4),
        dbc.Col(dbc.Button("Plot One Antenna", id="cal_plot_sant_btn", ), width=3, align="center"),
    ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'})

    plotrow = dbc.Row([
        # dbc.Col(width=2),
        dbc.Col(id="cal_sant_plot_col", width=12),
        # dbc.Col(width=2)
    ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'})

    return dbc.Container(
        [titlerow, plotrow],
        style={'marginBottom': '1.0em', 'marginTop': '1.0em'}
    )

### callbacks...
@callback(
    output=[
        Output("cal_allbeam_plot_col", "children"),
    ],
    inputs = [
        Input("cal_plot_allbeam_btn", "n_clicks"),
        State("cal_sbid", "value")
    ],
    prevent_initial_call=True,
)
def cal_plot_allbeam(nclicks, sbid):
    if nclicks is None: raise PreventUpdate
    qcfigpath = f"/data/craco/craco/calibration/SB0{sbid}/calsol_qc.png"
    if not os.path.exists(qcfigpath):
        dashfig = [
            dbc.Row(html.H5(f"cannot load quality control figure for SB0{sbid}..."))
        ]
    else:
        dashfig = [
            html.Img(src=load_figure_from_disk(qcfigpath), style={"width": "100%"})
        ]
    return dashfig,


@callback(
    output=[
        Output("cal_allant_plot_col", "children"),
    ],
    inputs = [
        Input("cal_plot_allant_btn", "n_clicks"),
        State("cal_sbid", "value"),
        State("cal_beam", "value"),
    ],
    prevent_initial_call=True,
)
def cal_plot_allant(nclicks, sbid, beam):
    if nclicks is None: raise PreventUpdate
    
    calsol = auto_sched.CalSolBeam(sbid=sbid, beam=beam)
    binamp = np.abs(calsol.binbp)
    binpha = calsol.binph
    phadif = calsol.phdif

    nant, nchan = binamp.shape
    xaxis = np.arange(nchan)
    yaxis = np.arange(1, nant+1)

    ### plot the raw amplitude, raw phase
    ### difference between raw and smooth phase
    ampfig = px.imshow(
        binamp, x=xaxis, y=yaxis,
        aspect="auto", origin="lower",
    )
    phafig = px.imshow(
        binpha, x=xaxis, y=yaxis,
        aspect="auto", origin="lower",
    )
    phafig["layout"]["coloraxis"].update(dict(cauto=False, cmax=180, cmin=-180))

    diffig = px.imshow(
        phadif, x=xaxis, y=yaxis,
        aspect="auto", origin="lower",
    )
    diffig["layout"]["coloraxis"].update(dict(cauto=False, cmax=180, cmin=0))

    figrow = dbc.Container([
        dbc.Row([dbc.Col(dcc.Graph(figure=ampfig, id="cal_beam_amp_fig"), width=6), dbc.Col(dcc.Graph(figure=phafig, id="cal_beam_pha_fig"), width=6)]),
        dbc.Row(dbc.Col(dcc.Graph(figure=diffig, id="cal_beam_dif_fig"), width=12),)
    ])
    return figrow,

@callback(
    output=[
        Output("cal_ak", "value"),
    ],
    inputs = [
        Input("cal_beam_amp_fig", "clickData"),
        Input("cal_beam_pha_fig", "clickData"),
        Input("cal_beam_dif_fig", "clickData"),
    ],
    prevent_initial_call=True,
)
def cal_get_click_ak(ampclick, phaclick, difclick):
    if ctx.triggered_id == "cal_beam_amp_fig":
        clickdata = ampclick
    elif ctx.triggered_id == "cal_beam_pha_fig":
        clickdata = phaclick
    elif ctx.triggered_id == "cal_beam_dif_fig":
        clickdata = difclick

    if clickdata is None: raise PreventUpdate
    point = clickdata["points"][0]
    return point.get("y"),

@callback(
    output = [
        Output("cal_sant_plot_col", "children"),
    ],
    inputs = [
        Input("cal_plot_sant_btn", "n_clicks"),
        State("cal_sbid", "value"),
        State("cal_beam", "value"),
        State("cal_ak", "value"),
    ],
    prevent_initial_call=True,
)
def cal_plot_single_ant(nclicks, sbid, beam, ak):
    if nclicks is None: raise PreventUpdate

    calsol = auto_sched.CalSolBeam(sbid=sbid, beam=beam)
    binamp = np.abs(calsol.binbp)
    binpha = calsol.binph
    smoamp = np.abs(calsol.smobp)
    smopha = calsol.smoph

    nant, nchan = binamp.shape
    xaxis = np.arange(nchan)

    ampfig = go.Figure()
    ampfig.add_traces([
        go.Scatter(x=xaxis, y=binamp[int(ak)-1], name="raw"),
        go.Scatter(x=xaxis, y=smoamp[int(ak)-1], name="fit")
    ])

    phafig = go.Figure()
    phafig.add_traces([
        go.Scatter(x=xaxis, y=binpha[int(ak)-1], name="raw"),
        go.Scatter(x=xaxis, y=smopha[int(ak)-1], name="fit"),
    ])

    qcfigpath = f"/data/craco/craco/calibration/SB0{sbid}/{beam:0>2}/bp_smooth/bp_ak{ak}.png"
    if not os.path.exists(qcfigpath):
        dashfig = html.H5(f"cannot load quality control figure for SB0{sbid} BEAM{beam} AK{ak}...")
    else:
        dashfig = dbc.Col(html.Img(src=load_figure_from_disk(qcfigpath), style={"width": "100%"}), width=3, align="center")


    figrows = dbc.Row([
        dbc.Col(dcc.Graph(figure=ampfig, id="cal_ak_amp_fig"), width=4),
        dbc.Col(dcc.Graph(figure=phafig, id="cal_ak_pha_fig"), width=4),
        dashfig,
    ])

    return figrows,


def layout(**query):
    return [
        dbc.Container(
            html.H2("CRACO Calibration Inspection Page (Beta)"), 
            style={'marginBottom': '1.0em', 'marginTop': '1.0em'}
        ),
        calib_allbeam_layout(query),
        calib_beam_layout(query),
        calib_ant_layout(query),
    ]