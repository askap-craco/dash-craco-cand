from dash import Dash, html, callback, dcc, Input, Output,State, ctx, dash_table
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash

import plotly.express as px

import pandas as pd
import numpy as np

import os
import glob
from io import StringIO

from apputil import (
    load_candidate,
    circular_mean,
    construct_beaminfo,
)

dash.register_page(__name__, path="/beam", title="CRACO File Loader")

def _workout_uniq_scans_plaintxt(scanlst):
    scans = ["/".join(scan.split("/")[-3:]) for scan in scanlst]
    scans = list(set(scans))

    defaultscan = None
    for scan in scans:
        if scan.endswith("/results"):
            defaultscan = scan

    return scans, defaultscan

### callback for loading candidate files and identify sbid etc.
@callback(
    Output("beam_query_strings", "data"),
    Output("beam_warning_match_msg", "children"),
    Input("beam_query_strings", "data"),
)
def update_beam_query_strings(beam_query_strings):
    beam_query_dict = eval(beam_query_strings)
    msg = None
    ### the beam_query_strings should be either
    # fname ### or # sbid, scanpath, beam
    ### update the beam_query_dict
    beam_query_dict = construct_beaminfo(beam_query_dict)
    if msg: return beam_query_dict.__str__(), dbc.Row(dbc.FormText(msg))
    return beam_query_dict.__str__(), msg

## callback for inspection
@callback(
    Output("beamcand_snr_value", "data"),
    Output("timestamp_slider", "max"),
    Output("beam_warning_load_msg", "children"),
    # Output("beamcand_center", "data"),
    Input("snr_slider", "value"),
    Input("beam_query_strings", "data"),
    prevent_initial_call=True
)
def filterdf_snr(snrthres, beam_query_strings):
    try:
        beam_query_dict = eval(beam_query_strings)
        # print(beam_query_dict)
        beamcanddf = load_candidate(beam_query_dict["fname"])
        beamcand_snrdf = beamcanddf[beamcanddf["SNR"] >= snrthres]
        beamcand_snrdf = beamcand_snrdf.reset_index(drop=True)

        # ra_center = circular_mean(beamcand_snrdf["ra"].to_numpy())
        # dec_center = np.mean(beamcand_snrdf["dec"].to_numpy())
        
        return (
            beamcand_snrdf.to_json(orient="split"), 
            beamcanddf["totalsample"].max(), None,
        )
    except:
        if "fname" in beam_query_dict:
            return (
                None, 100, dbc.Row(dbc.FormText(f"Warning... error in loading {beam_query_dict['fname']}...")),
                # None.__str__(),
            )
        return (
            None, 100, dbc.Row(dbc.FormText(f"Warning... no candidate file found...")),
            # None.__str__(),
        )

def beam_info_row(query_dict, label, key, width, labelwidth=2):
    if key not in query_dict: value = None
    else: value = dbc.FormText(query_dict[key])

    return dbc.Row([
            dbc.Col(dbc.Label(html.B(label), width=labelwidth)),
            dbc.Col(value, width=width),
        ])

### load basic information for the candidate file...
@callback(
    Output("beam_info_div", "children"),
    Input("beam_query_strings", "data"),
    prevent_initial_call=True
)
def beam_load_file_info(beam_query_strings):
    beam_query_dict = eval(beam_query_strings)
    try:
        return dbc.Row([
            dbc.Row([
                dbc.Col(beam_info_row(beam_query_dict, "SBID", "sbid", 8), width=4),
                dbc.Col(beam_info_row(beam_query_dict, "SCAN", "scan", 8), width=4),
                dbc.Col(beam_info_row(beam_query_dict, "TSTART", "tstart", 8), width=4),
            ]),
            dbc.Row([
                dbc.Col(beam_info_row(beam_query_dict, "RUNNAME", "runname", 8), width=4),
                dbc.Col(beam_info_row(beam_query_dict, "BEAM", "beam", 8), width=4),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label(html.B("CANDFILE")), width=2),
                dbc.Col([
                    dbc.FormText(beam_query_dict["fname"], id="beam_file_load_fpath"),
                    dcc.Clipboard(target_id="beam_file_load_fpath", title="copy", style = {"display": "inline-block"}),
                ]),
            ]),
        ])
    except:
        PreventUpdate()



def crossfilter_df(df, selections):
    selectedpoints = df.index
    for selected_data in selections:
        if selected_data and selected_data["points"]:
            selectedpoints = np.intersect1d(
                selectedpoints, [p["pointIndex"] for p in selected_data["points"]]
            )
    return selectedpoints

@callback(
    Output("time_snr_figure", "figure"),
    Output("time_dm_figure", "figure"),
    Output("lpix_mpix_static_figure", "figure"),
    Input("time_snr_figure", "selectedData"),
    Input("time_dm_figure", "selectedData"),
    Input("lpix_mpix_static_figure", "selectedData"),
    Input("beamcand_snr_value", "data"),
)
def crossfilter_plot(
        timesnrfigure_selection, 
        timedmfigure_selection, 
        lpixmpixstatic_selection,
        beamcand_snrdf_cache
    ):
    # print("crossfilter_plot is fired!!! Due to {}".format(ctx.triggered_id))
    try:
        beamcand_snrdf = pd.read_json(StringIO(beamcand_snrdf_cache), orient="split")
    except:
        raise PreventUpdate()
    selectedpoints = beamcand_snrdf.index
    
    if ctx.triggered_id != "beamcand_snr_value":
        selectedpoints = crossfilter_df(
            beamcand_snrdf, 
            [timesnrfigure_selection, timedmfigure_selection, lpixmpixstatic_selection]
        )

    # time-snr plot
    time_snr_scatter = px.scatter(
        beamcand_snrdf, x="totalsample", y="SNR",
        color="dm",  hover_data=["lpix", "mpix"],
    )

    time_snr_scatter.update_layout(
        coloraxis=dict(cmax=250, cmin=0),
        dragmode="select",
        newselection_mode="gradual",
    )

    #time-dm plot
    time_dm_scatter = px.scatter(
        beamcand_snrdf, x="totalsample", y="dm",
        color="boxcwidth",  hover_data=["lpix", "mpix"],
    )

    time_dm_scatter.update_layout(
        coloraxis=dict(cmax=7, cmin=0),
        dragmode="select",
        newselection_mode="gradual",
    )

    # location plot
    location_scatter = px.scatter(
        beamcand_snrdf, x="lpix", y="mpix", color="dm", 
        marginal_x="histogram", marginal_y="histogram",
    )
    location_scatter.update_xaxes(range=(-0.5, 256.5), )
    location_scatter.update_yaxes(range=(-0.5, 256.5), )
    location_scatter.update_layout(
        coloraxis=dict(cmax=250, cmin=0),
        dragmode="select",
        newselection_mode="gradual",
    )


    if ctx.triggered_id != "beamcand_snr_value":
        time_snr_scatter.update_traces(
            selectedpoints=selectedpoints,
            unselected={"marker": {"opacity": 0.0}}
        )
        time_dm_scatter.update_traces(
            selectedpoints=selectedpoints,
            unselected={"marker": {"opacity": 0.0}}
        )
        location_scatter.update_traces(
            selectedpoints=selectedpoints,
            unselected={"marker": {"opacity": 0.0}}
        )

    return time_snr_scatter, time_dm_scatter, location_scatter

def format_datatable(df, numercols, precision=3, ignorecols=[]):
    datacolumns = []
    for col in df.columns:
        if col in ignorecols: continue
        if col in numercols:
            datacolumns.append(
                {"name": col, "id": col, "type": "numeric", "format": Format(precision=precision, scheme=Scheme.fixed)}
            )
        else:
            datacolumns.append({"name": col, "id": col,})
    
    return datacolumns

@callback(
    Output("filtered_candidate_table_div", "children"),
    Output("filtered_table_index", "data"),
    Input("time_snr_figure", "selectedData"),
    Input("time_dm_figure", "selectedData"),
    Input("lpix_mpix_static_figure", "selectedData"),
    Input("beamcand_snr_value", "data"),
)
def crossfilter_table(
        timesnrfigure_selection, 
        timedmfigure_selection, 
        lpixmpixstatic_selection,
        beamcand_snrdf_cache
    ):
    try:
        beamcand_snrdf = pd.read_json(StringIO(beamcand_snrdf_cache), orient="split")
    except:
        raise PreventUpdate()
    beamcand_snrdf["index"] = beamcand_snrdf.index
    selectedpoints = beamcand_snrdf.index
    
    if ctx.triggered_id != "beamcand_snr_value":
        selectedpoints = crossfilter_df(
            beamcand_snrdf, 
            [timesnrfigure_selection, timedmfigure_selection, lpixmpixstatic_selection]
        )

    selected_df = beamcand_snrdf.iloc[selectedpoints]

    crosstable = dash_table.DataTable(
        id="filtered_candidate_table",
        columns=format_datatable(
            selected_df, precision=3,
            numercols=["SNR", "obstimesec", "dmpccm"],
            ignorecols=["rawsn", "obstimesec", "dm"],
        ),
        data = selected_df.to_dict("records"),
        row_selectable="multi",
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
        selected_rows=[],
        page_current=0,
        page_size=10,
    )

    return crosstable, list(selectedpoints).__str__()

# click the data and return the corresponding pixel plot
@callback(
    Output("lpix_mpix_slider_figure", "figure"),
    Input("timestamp_slider", "value"),
    Input("timestamp_slider_width_input", "value"),
    Input("beamcand_snr_value", "data"),
)
def loc_sample_plot(
        timestamp, width, beamcand_snrdf_cache
    ):
    try:
        beamcand_snrdf = pd.read_json(StringIO(beamcand_snrdf_cache), orient="split")
    except:
        raise PreventUpdate()
    sample_df = beamcand_snrdf[np.abs(beamcand_snrdf["totalsample"]-timestamp) <= width]

    loctime_scatter = px.scatter(
        sample_df, x="lpix", y="mpix", color="dm",
    )
    loctime_scatter.update_xaxes(range=(-0.5, 256.5), )
    loctime_scatter.update_yaxes(range=(-0.5, 256.5), )
    loctime_scatter.update_layout(
        coloraxis=dict(cmax=250, cmin=0),
    )

    return loctime_scatter

@callback(
    Output("clicked_candidate_table_div", "children"),
    Output("clicked_candidate_push", "href"),
    # Output("click_id_value", "data"),
    Output("timestamp_slider", "value"),
    Input("time_snr_figure", "clickData"),
    Input("time_dm_figure", "clickData"),
    Input("lpix_mpix_static_figure", "clickData"),
    Input("beamcand_snr_value", "data"),
    State("beam_query_strings", "data")
    # State("click_id_value", "data")
)
def clicked_candidate_table(
        timesnrfigure_click, 
        timedmfigure_click, 
        lpixmpixstatic_click,
        beamcand_snrdf_cache,
        beam_query_strings,
        # click_cache,
    ):
    try:
        beamcand_snrdf = pd.read_json(StringIO(beamcand_snrdf_cache), orient="split")
    except:
        raise PreventUpdate()
    beam_query_dict = eval(beam_query_strings)

    # clickstatus = eval(click_cache)

    if ctx.triggered_id == "beamcand_snr_value":
        clicked_df = beamcand_snrdf.iloc[[]] # create an empty dataframe
        clickstatus = None
    if ctx.triggered_id == "time_snr_figure":
        clicked_df = beamcand_snrdf.iloc[[timesnrfigure_click["points"][0]["pointIndex"]]]
        clickstatus = timesnrfigure_click["points"][0]["pointIndex"]
    if ctx.triggered_id == "time_dm_figure":
        clicked_df = beamcand_snrdf.iloc[[timedmfigure_click["points"][0]["pointIndex"]]]
        clickstatus = timedmfigure_click["points"][0]["pointIndex"]
    if ctx.triggered_id == "lpix_mpix_static_figure":
        clicked_df = beamcand_snrdf.iloc[[lpixmpixstatic_click["points"][0]["pointIndex"]]]
        clickstatus = lpixmpixstatic_click["points"][0]["pointIndex"]

    if len(clicked_df) == 0: 
        nsample = 0
        candurl = None
    else: 
        clickedrow = clicked_df.iloc[0]
        nsample = clicked_df.iloc[0]["totalsample"]
    
        ### what is the url?
        candurl = "/candidate?sbid={}&beam={}&scan={}&tstart={}&runname={}&dm={}&boxcwidth={}&lpix={}&mpix={}&totalsample={}&ra={}&dec={}".format(
            beam_query_dict["sbid"], beam_query_dict["beam"], beam_query_dict["scan"], beam_query_dict["tstart"],
            beam_query_dict["runname"], clickedrow["dmpccm"], int(clickedrow["boxcwidth"]), int(clickedrow["lpix"]), int(clickedrow["mpix"]),
            int(clickedrow["totalsample"]), clickedrow["ra"], clickedrow["dec"],
        )

    return (
        dash_table.DataTable(
            id="clicked_candidate_table",
            columns=format_datatable(
                clicked_df, precision=3, 
                numercols=["SNR", "obstimesec", "dmpccm"],
                ignorecols=["rawsn", "obstimesec", "dm"],
            ),
            data = clicked_df.to_dict("records")
        ), 
        candurl, 
        nsample,
    )

@callback(
    Output("selected_timesnr_btn", "color"),
    Output("selected_timedm_btn", "color"),
    Output("selected_lpixmpix_btn", "color"),
    Input("time_snr_figure", "selectedData"),
    Input("time_dm_figure", "selectedData"),
    Input("lpix_mpix_static_figure", "selectedData"),
)
def indicate_selection_status(
        timesnr_selected, timedm_selected, locstatics_selected,
    ):

    colors = []
    for selected in [timesnr_selected, timedm_selected, locstatics_selected,]:
        color = "success"
        if selected: color="danger"
        colors.append(color)
    return colors

@callback(
    Output("selected_candidate_table_div", "children"),
    Output("selected_candidate_push", "href"),
    # Output("select_id_value", "data"),
    Input("filtered_candidate_table", "selected_rows"),
    Input("beamcand_snr_value", "data"),
    State("beam_query_strings", "data"),
    State("filtered_table_index", "data"),
    # State("select_id_value", "data"),
)
def selected_rows_table(
        selected_row_ids, beamcand_snrdf_cache, 
        beam_query_strings, filtered_table_index_cache
    ):

    try:
        beamcand_snrdf = pd.read_json(StringIO(beamcand_snrdf_cache), orient="split")
    except:
        raise PreventUpdate()
    beam_query_dict = eval(beam_query_strings)
    filtered_table_index = eval(filtered_table_index_cache)

    selected_row = beamcand_snrdf.iloc[filtered_table_index].iloc[selected_row_ids]
    if selected_row_ids: 
        clickedrow = selected_row.iloc[0]
        candurl = "/candidate?sbid={}&beam={}&scan={}&tstart={}&results={}&dm={}&boxcwidth={}&lpix={}&mpix={}&totalsample={}&ra={}&dec={}".format(
            beam_query_dict["sbid"], beam_query_dict["beam"], beam_query_dict["scan"], beam_query_dict["tstart"],
            beam_query_dict["runname"], clickedrow["dmpccm"], int(clickedrow["boxcwidth"]), int(clickedrow["lpix"]), int(clickedrow["mpix"]),
            int(clickedrow["totalsample"]), clickedrow["ra"], clickedrow["dec"],
        )
    else:
        candurl = None

    return (
        dash_table.DataTable(
            id="selected_candidate_table",
            columns=format_datatable(
                selected_row, precision=3,
                numercols=["SNR", "obstimesec", "dmpccm"],
                ignorecols=["rawsn", "obstimesec", "dm"],
            ),
            data=selected_row.to_dict("records")
        ), 
        candurl,
    )

### layout related
def snr_slider(default=6):
    header = dbc.Row([html.H5("Signal to Noise Ratio (SNR) Threshold")])
    slidermarks = [6, 8, 10, 12, 15]
    slider = dbc.Row(
        [
            dcc.Slider(6, 15, value=default,
                marks={i: {"label": str(i)} for i in slidermarks},
                tooltip={"placement": "bottom", "always_visible": True},
                id="snr_slider"
            )
        ]
    )

    return dbc.Container(
        dbc.Row([
            dbc.Col(header),
            dbc.Col(slider),
        ], style = {"margin": "15px"})
    )

def figure_container(figuredivid, figureid):
    return dbc.Container([
        dbc.Row(html.Div(
            dcc.Graph(id=figureid),
            id=figuredivid,
        ))
    ])

# max=beamcanddf["totalsample"].max()
def lpix_mpix_slider_container():
    slider = dcc.Slider(
        min=0, max=100, step=1, marks=None,
        tooltip={"placement": "bottom", "always_visible": True}, value=0,
        id="timestamp_slider",
    )
    inputbox = dcc.Input(
        type="number", id="timestamp_slider_width_input", value=3
    )
    sliderrow = html.Div(
        [
            html.P("totalsample"), slider,
            html.P("width"), inputbox,
        ],
        style={
            "display": "grid",
            "grid-template-columns": "15% 50% 15% 15%",
        }
    )
    return dbc.Container([
        figure_container("lpix_mpix_slider_div", "lpix_mpix_slider_figure"),
        sliderrow,
    ])

def selected_region_status_container():
    buttons = dbc.Row([
        dbc.Col(html.P("Selection Region Indicator"), width=2),
        dbc.Col([
            dbc.Button("Time-SNR", id="selected_timesnr_btn"),
            dbc.Button("Time-DM", id="selected_timedm_btn"),
            dbc.Button("Lpix-Mpix", id="selected_lpixmpix_btn"),
        ], width=4,
        ),
    ])

    return dbc.Container(dbc.Row(buttons))

def candidate_plots_container():
    return dbc.Container([
        dbc.Container(dbc.Row(dbc.Col([
            dbc.Row([
                dbc.Col(html.H5("Clicked Candidate"), width=2),
                dbc.Col(html.A(dbc.Button("Push", color="success"), id="clicked_candidate_push", target="_blank"), width=4),
            ]),
            dbc.Row(html.Div(id="clicked_candidate_table_div")),
        ]))),
        dbc.Container(dbc.Row([
            dbc.Col(figure_container("time_snr_figure_div", "time_snr_figure")),
            dbc.Col(figure_container("time_dm_figure_div", "time_dm_figure")),
        ])),
        dbc.Container(dbc.Row([
            dbc.Col(figure_container("lpix_mpix_static_figure_div", "lpix_mpix_static_figure")),
            dbc.Col(lpix_mpix_slider_container()),
        ])),
        selected_region_status_container(),
        dbc.Container(
            dbc.Row([
                dbc.Col(figure_container("radec_figure_div", "radec_figure"), width=6),
                dbc.Col([
                    dbc.Row(html.Div(id="psrcat_beam_table_div")),
                    dbc.Row(html.Div(id="racs_beam_table_div")),
                ], width=6, className="h-100"),
            ]),
        ),
        dbc.Container(dbc.Row(dbc.Col([
            dbc.Row(html.H5("Filtered Candidate Table")),
            dbc.Row(html.Div(id="filtered_candidate_table_div")),
        ]))),
        dbc.Container(dbc.Row(dbc.Col([
            dbc.Row([
                dbc.Col(html.H5("Selected Candidate"), width=2),
                dbc.Col(html.A(dbc.Button("Push", color="success"), id="selected_candidate_push", target="_blank"), width=4),
            ]),
            dbc.Row(html.Div(id="selected_candidate_table_div")),
        ]))),
    ])
    
### final layout
def layout(**beam_query_strings):
    return dbc.Container(
        [
            dcc.Location(id="beam_url", refresh=False),
            dbc.Container(html.H5("Loading Messages")),
            dcc.Store(id="beam_query_strings", data=beam_query_strings.__str__()),
            dbc.Container(id="beam_warning_match_msg"), dbc.Container(id="beam_warning_load_msg"),
            dbc.Container(html.H5("Basic Information")),
            dbc.Container(id="beam_info_div"),
            snr_slider(default=8),
            candidate_plots_container(),
            dcc.Store(id="beamcand_snr_value"),
            dcc.Store(id="click_id_value", data=None.__str__()),
            dcc.Store(id="select_id_value", data=None.__str__()),
            dcc.Store(id="filtered_table_index", data="None"),
            dcc.Store(id="beamcand_center")
        ]
    )