from dash import Dash, html, callback, dcc, Input, Output,State, ctx, dash_table
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash

import plotly.express as px

import pandas as pd
import numpy as np

from apputil import FileTree, load_candidate

dash.register_page(__name__, path="/beamfile", title="CRACO File Loader")

beamcanddf = load_candidate(
    "/data/fast/wan342/work/cracorun/50705/candidates.txtb31.uniq"
)

## callback for inspection
@callback(
    Output("beamcand_snr_value", "data"),
    Input("snr_slider", "value"),
)
def filterdf_snr(snrthres):
    beamcand_snrdf = beamcanddf[beamcanddf["SNR"] >= snrthres]
    beamcand_snrdf = beamcand_snrdf.reset_index(drop=True)
    return beamcand_snrdf.to_json(orient="split")

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
    beamcand_snrdf = pd.read_json(beamcand_snrdf_cache, orient="split")
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
        coloraxis=dict(cmax=20, cmin=0),
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
        coloraxis=dict(cmax=20, cmin=0),
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
    beamcand_snrdf = pd.read_json(beamcand_snrdf_cache, orient="split")
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

    return crosstable

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
    beamcand_snrdf = pd.read_json(beamcand_snrdf_cache, orient="split")
    sample_df = beamcand_snrdf[np.abs(beamcand_snrdf["totalsample"]-timestamp) <= width]

    loctime_scatter = px.scatter(
        sample_df, x="lpix", y="mpix", color="dm",
    )
    loctime_scatter.update_xaxes(range=(-0.5, 256.5), )
    loctime_scatter.update_yaxes(range=(-0.5, 256.5), )
    loctime_scatter.update_layout(
        coloraxis=dict(cmax=20, cmin=0),
    )

    return loctime_scatter

@callback(
    Output("clicked_candidate_table_div", "children"),
    Output("timestamp_slider", "value"),
    Input("time_snr_figure", "clickData"),
    Input("time_dm_figure", "clickData"),
    Input("lpix_mpix_static_figure", "clickData"),
    Input("beamcand_snr_value", "data"),
)
def clicked_candidate_table(
        timesnrfigure_click, 
        timedmfigure_click, 
        lpixmpixstatic_click,
        beamcand_snrdf_cache
    ):
    beamcand_snrdf = pd.read_json(beamcand_snrdf_cache, orient="split")

    if ctx.triggered_id == "beamcand_snr_value":
        clicked_df = beamcand_snrdf.iloc[[]] # create an empty dataframe
    if ctx.triggered_id == "time_snr_figure":
        clicked_df = beamcand_snrdf.iloc[[timesnrfigure_click["points"][0]["pointIndex"]]]
    if ctx.triggered_id == "time_dm_figure":
        clicked_df = beamcand_snrdf.iloc[[timedmfigure_click["points"][0]["pointIndex"]]]
    if ctx.triggered_id == "lpix_mpix_static_figure":
        clicked_df = beamcand_snrdf.iloc[[lpixmpixstatic_click["points"][0]["pointIndex"]]]

    if len(clicked_df) == 0: nsample = 0
    else: nsample = clicked_df.iloc[0]["totalsample"]

    return dash_table.DataTable(
        id="clicked_candidate_table",
        columns=format_datatable(
            clicked_df, precision=3, 
            numercols=["SNR", "obstimesec", "dmpccm"],
            ignorecols=["rawsn", "obstimesec", "dm"],
        ),
        data = clicked_df.to_dict("records")
    ), nsample

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
    Input("filtered_candidate_table", "selected_rows"),
    Input("beamcand_snr_value", "data"),
)
def selected_rows_table(
        selected_row_ids, beamcand_snrdf_cache
    ):

    beamcand_snrdf = pd.read_json(beamcand_snrdf_cache, orient="split")
    selected_row = beamcand_snrdf.iloc[selected_row_ids]

    return dash_table.DataTable(
        id="selected_candidate_table",
        columns=format_datatable(
            selected_row, precision=3,
            numercols=["SNR", "obstimesec", "dmpccm"],
            ignorecols=["rawsn", "obstimesec", "dm"],
        ),
        data=selected_row.to_dict("records")
    )


### layout related
def snr_slider(default=6):
    header = dbc.Row([html.H5("Signal to Noise Ratio (SNR) Threshold")])
    slidermarks = [6, 8, 10, 12, 15]
    slider = dbc.Row(
        [
            dcc.Slider(6, 15, value=6,
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
        ])
    )

def figure_container(figuredivid, figureid):
    return dbc.Container([
        dbc.Row(html.Div(
            dcc.Graph(id=figureid),
            id=figuredivid,
        ))
    ])

def lpix_mpix_slider_container():
    slider = dcc.Slider(
        min=0, max=beamcanddf["totalsample"].max(), step=1, marks=None,
        tooltip={"placement": "bottom", "always_visible": True}, value=0,
        id="timestamp_slider",
    )
    inputbox = dcc.Input(
        type="number", id="timestamp_slider_width_input", value=1
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

### final layout
layout = dbc.Container(
    [
        snr_slider(),
        dbc.Container(dbc.Row(dbc.Col([
            dbc.Row([
                dbc.Col(html.H5("Clicked Candidate"), width=2),
                dbc.Col(dbc.Button("Push", id="clicked_candidate_push", color="success"), width=4),
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
        dbc.Container(dbc.Row(dbc.Col([
            dbc.Row(html.H5("Filtered Candidate Table")),
            dbc.Row(html.Div(id="filtered_candidate_table_div")),
        ]))),
        dbc.Container(dbc.Row(dbc.Col([
            dbc.Row(html.H5("Selected Candidate")),
            dbc.Row(html.Div(id="selected_candidate_table_div")),
        ]))),
        dbc.Row(html.P(id="test_region")),
        dcc.Store(id="beamcand_snr_value")
    ]
)