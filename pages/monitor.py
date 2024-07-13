import os
import subprocess

from dash import Dash, html, callback, dcc, Input, Output,State, ctx, dash_table
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash
import plotly.express as px

from apputil import load_figure_from_disk

from astropy.coordinates import SkyCoord, Angle
from astropy import units

import pandas as pd

from craft import uvfits
from craco.craco_run.slackpost import SlackPostManager
from craco.datadirs import SchedDir, ScanDir
from craco.craco_run import auto_sched

dash.register_page(__name__, path="/monitor", title="CRACO Monitor")

app = dash.get_app()

### function to load dataframe...
def load_database_pandas(sqlquery):
    conn = auto_sched.get_psql_engine("dbreader")
    return pd.read_sql(sqlquery, con=conn)

def dataframe_to_dashtable(df, id, page_size=10, ):

    tab = dash_table.DataTable(
        id = id,
        columns = [{"name": i, "id": i} for i in df.columns],
        data = df.to_dict("records"),
        page_size = page_size,
        style_table={'overflowX': 'auto', 'overflowY': 'auto'},  # Enable horizontal scroll
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
    )
    return tab

def find_recent_sbid(table="observation"):
    df = load_database_pandas(f"select max(sbid) as sbid from {table}")
    return df.iloc[0]["sbid"]

### basic layout...
def calibration_layout():
    query_row = dbc.Row([
        dbc.Col(dbc.Button("Query Recent", id="op_query_cal_recent_btn"), width=2),
        dbc.Col(dbc.Button("Query SBID", id="op_query_sbid_btn"), width=2),
        dbc.Col(dbc.Input(placeholder="input sbid here...", id="op_query_sbid_input"), width=2),
    ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'})
    display_row = dbc.Row([
        dbc.Col(id="op_calib_tab_col", width=6),
        dbc.Col(id="op_calib_qc_col", width=6),
    ],  style={'marginBottom': '1.0em', 'marginTop': '1.0em'})
    verify_row = dbc.Row(id="op_verify_calib_row")

    return dbc.Container([query_row, display_row, verify_row])

### callback for loading table only
# @app.long_callback(
@callback(
    output=[
        Output("op_calib_tab_col", "children"),
    ],
    inputs = [
        Input("op_query_cal_recent_btn", "n_clicks"),
        Input("op_query_sbid_btn", "n_clicks"),
        State("op_query_sbid_input", "value"),
    ],
    running = [
        (Output("op_query_cal_recent_btn", "disabled"), True, False),
        (Output("op_query_sbid_btn", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def op_show_calib_tab(recent_click, query_click, sbid):
    NLIMIT = 10
    if ctx.triggered_id == "op_query_cal_recent_btn":
        if recent_click is None: raise PreventUpdate
        mode = "recent"
    elif ctx.triggered_id == "op_query_sbid_btn":
        if query_click is None: raise PreventUpdate
        mode = "query"
    else: raise PreventUpdate

    ### table region
    if mode == "recent":
        df = load_database_pandas(
            f"select * from calibration order by sbid desc limit {NLIMIT};"
        )
        dashtab = dataframe_to_dashtable(df, id="op_cal_tab", page_size=NLIMIT)
    if mode == "query":
        df = load_database_pandas(
            f"select * from calibration where sbid <= {sbid} order by sbid desc limit {NLIMIT};"
        )
        dashtab = dataframe_to_dashtable(df, id="op_cal_tab", page_size=NLIMIT)

    return[dashtab]

### for plotting qc image and show verification button
# # @app.long_callback(
@callback(
    output=[
        Output("op_calib_qc_col", "children"),
        Output("op_verify_calib_row", "children"),
        Output("op_query_sbid_input", "value")
    ],
    inputs = [
        Input("op_cal_tab", "active_cell"),
        State("op_cal_tab", "data")
    ],
    # running = [
    #     (Output("op_query_cal_recent_btn", "disabled"), True, False),
    #     (Output("op_query_sbid_btn", "disabled"), True, False),
    # ],
    prevent_initial_call=True,
)
def op_update_qcfig(active_cell, caltab_data):
    if active_cell is None: raise PreventUpdate
    datarow = caltab_data[active_cell["row"]]
    sbid = datarow["sbid"]
    valid = datarow["valid"]

    qcfigpath = f"/data/craco/craco/calibration/SB0{sbid}/calsol_qc.png"
    if not os.path.exists(qcfigpath):
        dashfig = [
            dbc.Row(html.H5(f"cannot load quality control figure for SB0{sbid}..."))
        ]
    else:
        dashfig = [
            html.Img(src=load_figure_from_disk(qcfigpath), style={"width": "100%"})
        ]

    ### show the verification row
    if valid: verify_disabled = True; reject_disabled = False
    else: verify_disabled = False; reject_disabled = True

    verifyrow = [
        dbc.Col(html.B("SELECTED SBID"), width=2, align="center"),
        dbc.Col(dbc.Input(value=int(sbid), type="number", readonly=True), width=2),
        dbc.Col(dbc.Button("VERIFY", color="success", id="op_calib_verify_btn", disabled=verify_disabled), width=2),
        dbc.Col(dbc.Button("REJECT", color="danger", id="op_calib_reject_btn", disabled=reject_disabled), width=2),
    ]
    return (dashfig, verifyrow, sbid)

### callback to active the modal
@callback(
    output=[
        Output("op_verify_cal_sbid_modal", "children"),
        Output("op_verify_cal_sbid_modal", "is_open"),
    ],
    inputs = [
        Input("op_calib_verify_btn", "n_clicks"),
        Input("op_calib_reject_btn", "n_clicks"),
        Input("op_calib_modal_proceed_btn", "n_clicks"),
        Input("op_calib_modal_cancel_btn", "n_clicks"),
        State("op_query_sbid_input", "value"),
        State("op_calib_modal_mode", "data"),
        State("op_calib_modal_note", "value")
    ],
    prevent_initial_call=True,
)
def op_control_modal(
    verify_nclick, reject_nclick, 
    proceed_nclick, cancel_nclick,
    calsbid, calibmode, calibnote,
):
    if ctx.triggered_id == "op_calib_verify_btn":
        if verify_nclick is None: raise PreventUpdate
        mode = "verify"
    elif ctx.triggered_id == "op_calib_reject_btn":
        if reject_nclick is None: raise PreventUpdate
        mode = "reject"
    elif ctx.triggered_id == "op_calib_modal_proceed_btn":
        if proceed_nclick is None: raise PreventUpdate
        mode = "proceed"
    elif ctx.triggered_id == "op_calib_modal_cancel_btn":
        if cancel_nclick is None: raise PreventUpdate
        mode = "cancel"

    if mode in ["verify", "reject"]:
        modal_children = [
            dbc.ModalHeader(dbc.ModalTitle("Confirm!"), close_button=True),
            dbc.ModalBody([
                html.P(f"You are going to {mode} the calibration solution for SBID {calsbid}"),
                dbc.Row([
                    dbc.Col(html.B(f"note for {mode}ing"), width=6, align="center"),
                    dbc.Col(dbc.Input(value="manual", id="op_calib_modal_note"), width=6, align="center"),
                ])
            ]),
            dcc.Store(id="op_calib_modal_mode", data=mode),
            dbc.ModalFooter([
                dbc.Button("Proceed", color="danger", id="op_calib_modal_proceed_btn"),
                dbc.Button("Cancel", color="light", id="op_calib_modal_cancel_btn"),
            ])
        ]
        
        return (modal_children, True)
    
    ### this is for proceeding...
    if mode == "proceed":
        if calibmode == "verify": newvalid = True
        else: newvalid = False
        auto_sched.update_table_single_entry(int(calsbid), "valid", newvalid, "calibration")
        auto_sched.update_table_single_entry(int(calsbid), "note", calibnote, "calibration")

        return (None, False)
    
    if mode == "cancel":
        return (None, False)


def op_calib_modal_layout():
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Confirm!"), close_button=True),
        dbc.ModalBody([
            html.P(id="op_calib_modal_paragraph"),
            dbc.Row([
                dbc.Col(html.B(f"Here are some prompt messages"), width=6, align="center"),
                dbc.Col(dbc.Input(value="manual", id="op_calib_modal_note"), width=6, align="center"),
            ])
        ]),
        dcc.Store(id="op_calib_modal_mode", ),
        dbc.ModalFooter([
            dbc.Button("Proceed", color="danger", id="op_calib_modal_proceed_btn"),
            dbc.Button("Cancel", color="light", id="op_calib_modal_cancel_btn"),
        ])
    ], id="op_verify_cal_sbid_modal", keyboard=False, backdrop="static",)


##### section for observation
def observation_layout():
    query_row = dbc.Row([
        dbc.Col(dbc.Button("Query Recent", id="op_query_obs_recent_btn"), width=2),
        dbc.Col(dbc.Button("Query SBID", id="op_query_obs_sbid_btn"), width=2),
        dbc.Col(dbc.Input(placeholder="input sbid here...", id="op_query_obs_sbid_input"), width=2),
    ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'})
    display_row = dbc.Row([
        dbc.Col(id="op_obs_tab_col", width=12),
        # dbc.Col(id="op_calib_qc_col", width=6),
    ],  style={'marginBottom': '1.0em', 'marginTop': '1.0em'})
    verify_row = dbc.Row(id="op_find_calib_row")

    return dbc.Container([query_row, display_row, verify_row])

# @app.long_callback(
@callback(
    output=[
        Output("op_obs_tab_col", "children"),
    ],
    inputs = [
        Input("op_query_obs_recent_btn", "n_clicks"),
        Input("op_query_obs_sbid_btn", "n_clicks"),
        State("op_query_obs_sbid_input", "value"),
    ],
    running = [
        (Output("op_query_obs_recent_btn", "disabled"), True, False),
        (Output("op_query_obs_sbid_btn", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def op_show_calib_tab(recent_click, query_click, sbid):
    NLIMIT = 10
    if ctx.triggered_id == "op_query_obs_recent_btn":
        if recent_click is None: raise PreventUpdate
        mode = "recent"
    elif ctx.triggered_id == "op_query_obs_sbid_btn":
        if query_click is None: raise PreventUpdate
        mode = "query"
    else: raise PreventUpdate

    ### table region
    if mode == "recent":
        df = load_database_pandas(
            f"select * from observation order by sbid desc limit {NLIMIT};"
        )
        dashtab = dataframe_to_dashtable(df, id="op_obs_tab", page_size=NLIMIT)
    if mode == "query":
        df = load_database_pandas(
            f"select * from observation where sbid <= {sbid} order by sbid desc limit {NLIMIT};"
        )
        dashtab = dataframe_to_dashtable(df, id="op_obs_tab", page_size=NLIMIT)

    return [dashtab]

### this is for finding calibration stuff
@callback(
    output = [
        Output("op_find_calib_row", "children")
    ],
    inputs = [
        Input("op_obs_tab", "active_cell"),
        State("op_obs_tab", "data")
    ],
    prevent_initial_call=True,
)
def op_update_findcal(active_cell, obstab_data):
    if active_cell is None: raise PreventUpdate
    datarow = obstab_data[active_cell["row"]]
    sbid = datarow["sbid"]

    row = [
        dbc.Col(html.B("Find Calibration Solution for"), width=3, align="center"),
        dbc.Col(dbc.Input(value=sbid, id="op_findcal_sbid_input", type="number"), width=2, align="center"),
        dbc.Col([
            dbc.Select(
                options=[
                    dict(label="calibration pipeline used", value="0"),
                    dict(label="best calibration in the database", value="1"),
                ],
                id="op_findcal_method", value="0",
            )
        ], width=3, align="center"),
        dbc.Col(dbc.Button("Find", color="success", id="op_findcal_btn"), width=1, align="center"),
        dbc.Col(id="op_findcal_result_col", width=3, align="center")
    ]

    return row,

# find calibration button
# @app.long_callback(
@callback(
    output = [
        Output("op_findcal_result_col", "children"),
    ],
    inputs = [
        Input("op_findcal_btn", "n_clicks"),
        State("op_findcal_sbid_input", "value"),
        State("op_findcal_method", "value"),
    ],
    running = [
        (Output("op_findcal_btn", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def op_findcal_call(nclick, sbid, method):
    if nclick is None: raise PreventUpdate

    result = f"no calibration found..."
    if method == "0":
        ### get head node...
        calpath = f"/data/craco/craco/SB{sbid:0>6}/cal"
        if os.path.exists(calpath):
            caltar = os.readlink(calpath)
            try: 
                calsbid = int(caltar[-6:])
                result = f"calsbid SB{calsbid}"
            except: result = "error in getting calsbid..."
    elif method == "1":
        calfinder = auto_sched.CalFinder(sbid)
        calsbid, status = calfinder.query_calib_table()
        if calsbid is not None:
            result = f"calsbid SB{calsbid}"

    return result,

def layout():
    return [
        dbc.Container(
            html.H2("CRACO Monitor Page (Beta)"), 
            style={'marginBottom': '1.0em', 'marginTop': '1.0em'}
        ),
        ##### for calibration table
        dbc.Container([
            html.H4("Calibration Table", style={'marginBottom': '1.0em', 'marginTop': '1.0em'}),
            calibration_layout(),
        ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'}),
        # modal here
        op_calib_modal_layout(),
        html.Hr(),
        ##### for observation table
        dbc.Container([
            html.H4("Observation Table", style={'marginBottom': '1.0em', 'marginTop': '1.0em'}),
            observation_layout(),
        ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'}),
    ]