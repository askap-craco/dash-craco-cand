from dash import Dash, html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash

import glob
import os

import pandas as pd
import numpy as np
import plotly.express as px

### choose from file directly, select beamfile
### if you know 

dash.register_page(__name__, path="/beamfile", title="CRACO File Loader")

def _workout_uniq_scans(scanlst):
    scans = ["/".join(scan.split("/")[-3:]) for scan in scanlst]
    scans = list(set(scans))

    defaultscan = None
    for scan in scans:
        if scan.endswith("/results"):
            defaultscan = scan

    return [{"label": scan, "value": scan} for scan in scans], defaultscan

### callback
# output scan
@callback(
    Output("beam_scan_dropdown", "options"),
    Output("beam_scan_dropdown", "value"),
    Output("beam_beam_input", "disabled"),
    Input("beam_sbid_input", "value"),
)
def load_scan4sbid(inputsbid):
    try:
        if inputsbid.isdigit():
            sbid = "SB{:0>6}".format(int(inputsbid))
        else:
            sbid = "SB{:0>6}".format(int(inputsbid[2:]))
    except:
        return ([], None, True)

    scanlst = glob.glob(f"/data/seren-??/big/craco/{sbid}/scans/??/*/*")
    scanlst = [f for f in scanlst if os.path.isdir(f)]
    scan_options, defaultscan = _workout_uniq_scans(scanlst)
    if scan_options:
        return (scan_options, defaultscan, False, )
    return ([], None, True)

# check if candidate file exists...
@callback(
    Output("beam_cand_files_div", "children"),
    Output("cand_file_paths", "data"),
    Input("beam_beam_input", "value"),
    State("beam_sbid_input", "value"),
    State("beam_scan_dropdown", "value"),
)
def load_candidate(beamid, inputsbid, beamscan, ):
    if beamid is None: raise PreventUpdate()
    try:
        serennode = "{:0>2}".format(int(beamid) % 10 + 1)
        beam = "{:0>2}".format(int(beamid))
    except:
        raise PreventUpdate()

    try:
        if inputsbid.isdigit():
            sbid = "SB{:0>6}".format(int(inputsbid))
        else:
            sbid = "SB{:0>6}".format(int(inputsbid[2:]))
    except:
        raise PreventUpdate()


    resultpath = f"/data/seren-{serennode}/big/craco/{sbid}/scans/{beamscan}"
    allcandfile = f"{resultpath}/candidates.txtb{beam}"
    unicandfile = f"{resultpath}/clustering_output/candidates.txtb{beam}.uniq"

    print(allcandfile, unicandfile)

    # add file if it is existing
    candrows = []; candfiles = {}
    if os.path.exists(allcandfile):
        candrows.append(dbc.Row([
            dbc.Col("unclustered", width=2),
            dbc.Col([
                dbc.FormText(allcandfile, id="beam_cand_all_fpath"),
                dcc.Clipboard(target_id="beam_cand_all_fpath", title="copy", style = {"display": "inline-block"}),
            ], width=10)
        ]))
        candfiles["allcand_fpath"] = allcandfile
    if os.path.exists(unicandfile):
        candrows.append(dbc.Row([
            dbc.Col("clustered", width=2),
            dbc.Col([
                dbc.FormText(unicandfile, id="beam_cand_uni_fpath"),
                dcc.Clipboard(target_id="beam_cand_uni_fpath", title="copy", style = {"display": "inline-block"}),
            ], width=10)
        ]))
        candfiles["unicand_fpath"] = unicandfile

    return (
        candrows, candfiles.__str__()
    )    

### layout
# for selection files...
def find_cand_file_layout():
    sbidinput_row = dbc.Row([
        dbc.Col(dbc.Label(html.B("SBID")), width=1),
        dbc.Col(dbc.Input(id="beam_sbid_input", placeholder="Enter SBID"), width=2),
        dbc.Col(dbc.FormText(
            "Input a SBID, it can be something like 49721, 049721 or SB049721",
            color="secondary",
        ), width=6)
    ], style={"margin": "5px"})
    scanresult_row = dbc.Row([
        dbc.Col(dbc.Label(html.B("SCAN")), width=1),
        dbc.Col(dcc.Dropdown(
            id="beam_scan_dropdown",
            options=[],
            placeholder="select your scan - 00/20230101000000/results"
        ), width=6),
        dbc.Col(dbc.FormText("select a scan for a given sbid", color="secondary"), width=3)
    ], style={"margin": "5px"})
    beaminput_row = dbc.Row([
        dbc.Col(dbc.Label(html.B("BEAM")), width=1),
        dbc.Col(dbc.Input(id="beam_beam_input", placeholder="Enter BEAM", disabled=True), width=2),
        dbc.Col(dbc.FormText("select a beam to proceed", color="secondary"), width=2)
    ], style={"margin": "5px"})
    candfile_row = dbc.Row([
        dbc.Col(dbc.Label(html.B("FILE")), width=1),
        dbc.Col(dbc.Container(id="beam_cand_files_div"), width=10),
    ], style={"margin": "5px"})
    return dbc.Container([
        sbidinput_row, scanresult_row, 
        beaminput_row, candfile_row
    ])

### final layout
def layout(**beam_query_strings):
    return dbc.Container([
            dcc.Store(id="cand_file_paths"),
            dbc.Row(html.H5("Candidate File Input")),
            dbc.Row(find_cand_file_layout())
    ])