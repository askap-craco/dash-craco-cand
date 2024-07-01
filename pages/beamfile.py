from dash import Dash, html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash

import glob
import os
import re

import pandas as pd
import numpy as np
import plotly.express as px

from astropy.coordinates import SkyCoord
import astropy.units as units

from apputil import find_file
from craco.datadirs import DataDirs, SchedDir, ScanDir, RunDir, format_sbid


### choose from file directly, select beamfile
### if you know 

dash.register_page(__name__, path="/beamfile", title="CRACO File Loader")

def _is_headdir_beamfolder(path):
    folder = path.split("/")[-1]
    if re.match(r"beam\d{2}", folder): return True
    return False

def _get_result_scans(scans):
    scanfolder = ["/".join(scan.split("/")[:-1]) for scan in scans]
    uniq_scan = list(set(scanfolder))
    return [f"{scan}/results" for scan in uniq_scan]

def _workout_uniq_scans(sched_head_dir, scanlst):
    # this is for skadi
    scans = [scan for scan in scanlst if os.path.exists(f"{sched_head_dir}/scans/{scan}")]
    scans.extend(_get_result_scans(scans))
    scans = sorted(set(scans))
    # remove scans that finish with beam??
    scans = [scan for scan in scans if not _is_headdir_beamfolder(scan)]
    
    if len(scans) == 0: return [], None

    defaultscan = None
    for scan in scans:
        if scan.endswith("/results"):
            defaultscan = scan
    scans = sorted(scans)
    if defaultscan is None: defaultscan = scans[0]

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

    scheddir = SchedDir(sbid)
    scanlst = []
    for scan in scheddir.scans:
        scandir = ScanDir(sbid, scan)
        for run in scandir.runs:
            scanlst.append(f"{scan}/{run}")

    scan_options, defaultscan = _workout_uniq_scans(scheddir.sched_head_dir, scanlst)
    if scan_options:
        return (scan_options, defaultscan, False, )
    return ([], None, True)

# check if candidate file exists...
@callback(
    Output("beam_cand_files_div", "children"),
    Output("cand_file_paths", "data"),
    Input("beam_beam_input", "value"),
    Input("beam_sbid_input", "value"),
    State("beam_scan_dropdown", "value"),
)
def load_candidate(beamid, inputsbid, beamscan, ):
    if beamid is None: raise PreventUpdate()
    try:
        serennode = "{:0>2}".format(int(beamid) % 10 + 1)
        beam = "{:0>2}".format(int(beamid))
    except:
        return None, ""

    try:
        if inputsbid.isdigit():
            sbid = "SB{:0>6}".format(int(inputsbid))
        else:
            sbid = "SB{:0>6}".format(int(inputsbid[2:]))
    except:
        return None, ""

    query_dict = dict(sbid=sbid, beam=beam, scanpath=beamscan)
    print(query_dict)
    uvfitsfile = find_file("uvfits", query_dict)
    calfile = find_file("cal", query_dict)
    allcandfile = find_file("cand_raw", query_dict)
    unicandfile = find_file("cand_cls", query_dict)

    # add file if it is existing
    candrows = []; candfiles = {}
    if allcandfile is not None:
        candrows.append(dbc.Row([
            dbc.Col("unclustered", width=2),
            dbc.Col([
                dbc.FormText(allcandfile, id="beam_cand_all_fpath"),
                dcc.Clipboard(target_id="beam_cand_all_fpath", title="copy", style = {"display": "inline-block"}),
            ], width=10)
        ]))
        candfiles["allcand_fpath"] = allcandfile
    if unicandfile is not None:
        candrows.append(dbc.Row([
            dbc.Col("clustered", width=2),
            dbc.Col([
                dbc.FormText(unicandfile, id="beam_cand_uni_fpath"),
                dcc.Clipboard(target_id="beam_cand_uni_fpath", title="copy", style = {"display": "inline-block"}),
            ], width=10)
        ]))
        candfiles["unicand_fpath"] = unicandfile
    if uvfitsfile is not None:
        candrows.append(dbc.Row([
            dbc.Col("uvfits", width=2),
            dbc.Col([
                dbc.FormText(uvfitsfile, id="beam_cand_uvfits_fpath"),
                dcc.Clipboard(target_id="beam_cand_uvfits_fpath", title="copy", style = {"display": "inline-block"})
            ])
        ]))
    if calfile is not None:
        candrows.append(dbc.Row([
            dbc.Col("calfile", width=2),
            dbc.Col([
                dbc.FormText(calfile, id="beam_cand_cal_fpath"),
                dcc.Clipboard(target_id="beam_cand_cal_fpath", title="copy", style = {"display": "inline-block"})
            ])
        ]))

    ### add click button for unclustered and clustered files...
    allcand_href = None; unicand_href = None
    allcand_color = "secondary"; unicand_color = "secondary"
    allcand_title = "unclustered (unavail)"; unicand_title = "clustered (unavail)"
    if unicandfile is not None:
        unicand_href = f"/beam?fname={unicandfile}"
        unicand_color = "success"
        unicand_title = "clustered (recom)"
    if allcandfile is not None:
        allcand_href = f"/beam?fname={allcandfile}"
        allcand_color = "danger"
        allcand_title = "unclustered (no)"

    candrows.append(dbc.Row([
        dbc.Col(html.A(dbc.Button(unicand_title, color=unicand_color), href=unicand_href, target="_blank"), width=4),
        dbc.Col(html.A(dbc.Button(allcand_title, color=allcand_color), href=allcand_href, target="_blank"), width=4),
    ], style={"margin": "15px"}))
    candrows.append(dbc.Row([
        dbc.FormText("unclustered candidates may take super long time to render... we suggest you use clustered version...")
    ]))

    return (
        candrows, candfiles.__str__()
    )    

@callback(
    Output("beam_cand_coord_btn", "color"),
    Output("beam_cand_coord_link", "href"),
    Input("beam_cand_ra_input", "value"),
    Input("beam_cand_dec_input", "value"),
)
def link_beam_cand_coord(rainput, decinput):
    if rainput is None or decinput is None:
        return "secondary", None
    coordstr = "{} {}".format(rainput.strip(), decinput.strip())
    try:
        coord = SkyCoord(coordstr)
    except:
        try:
            coord = SkyCoord(coordstr, (units.hourangle, units.degree))
        except:
            return "danger", None
    radeg = coord.ra.value
    decdeg = coord.dec.value

    return "success", "/candidate?ra={}&dec={}".format(radeg, decdeg)


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

### candidate coord layout
def cand_coord_layout():
    rarow = dbc.Row([
        dbc.Col(dbc.Label(html.B("RA")), width=3),
        dbc.Col(dbc.Input(id="beam_cand_ra_input", placeholder="Right Ascension", type="text"), width=9),
    ])
    decrow = dbc.Row([
        dbc.Col(dbc.Label(html.B("DEC")), width=3),
        dbc.Col(dbc.Input(id="beam_cand_dec_input", placeholder="Declination", type="text"), width=9),
    ])
    candbutton = dbc.Row(
        html.A(dbc.Button(
            "Go To Coord", color="secondary", id="beam_cand_coord_btn"
        ), target="_blank", id="beam_cand_coord_link")
    )
    return dbc.Row([
        dbc.Col([rarow, decrow], width=6),
        dbc.Col(candbutton, align="center", width=6)
    ])

### final layout
def layout(**beam_query_strings):
    return dbc.Container([
            dcc.Store(id="cand_file_paths"),
            dbc.Row(html.H5("Candidate File Input")),
            dbc.Row(find_cand_file_layout()),
            dbc.Row(html.H5("Candidate Coordinate Input")),
            dbc.Row(dbc.FormText("This may be helpful if you just want to check if there is any obvious match for a given coord")),
            dbc.Row(cand_coord_layout(), style={"margin": "15px"}),
            
    ])