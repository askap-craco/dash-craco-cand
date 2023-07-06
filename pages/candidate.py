from dash import Dash, html, callback, dcc, Input, Output,State, ctx, dash_table
from dash_extensions.enrich import DashProxy, LogTransform, DashLogger
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash

import os
import glob

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as units
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad

from apputil import fig_to_uri

from craco import craco_candidate

dash.register_page(__name__, path="/candidate", title="CRACO candidate Plotter")

### functions

### callbacks
@callback(
    Output("cand_query_strings", "data"),
    Input("cand_query_strings", "data"),
)
def update_cand_query_strings(cand_query_strings):
    cand_query_strings = eval(cand_query_strings)
    ### set some default value in cand_query_strings
    default_values = {
        "sbid": "000000", "beam": "99", "scan": "00", "tstart": None,
        "results": "results", "dm": None,
        "lpix": None, "mpix": None, "boxcwidth": None,
        "totalsample": None, "ra": None, "dec": None,
    }

    for key, value in default_values.items():
        if key not in cand_query_strings:
            cand_query_strings[key] = value
    
    ### do we really need this?
    cand_query_strings["npzfname"] = "SB{}_B{}_{}_{}_{}_t{}.npz".format(
        cand_query_strings["sbid"], cand_query_strings["beam"],
        cand_query_strings["results"], cand_query_strings["lpix"],
        cand_query_strings["mpix"], cand_query_strings["totalsample"],
    )
    ###########################

    if cand_query_strings["sbid"] is None:
        print("A SBID must be provided...")
        raise PreventUpdate()

    if cand_query_strings["tstart"] is not None:
        cand_uvfits_path = "/data/seren-{:0>2}/big/craco/SB{:0>6}/scans/{}/{}/{}/b{:0>2}.uvfits".format(
            int(cand_query_strings["beam"]) % 10 + 1, cand_query_strings["sbid"],
            cand_query_strings["scan"], cand_query_strings["tstart"],
            cand_query_strings["results"], int(cand_query_strings["beam"])
        )
    else:
        cand_uvfits_pattern =  "/data/seren-{:0>2}/big/craco/SB{:0>6}/scans/{}/*/{}/b{:0>2}.uvfits".format(
            int(cand_query_strings["beam"]) % 10 + 1, cand_query_strings["sbid"], cand_query_strings["scan"], 
            cand_query_strings["results"], int(cand_query_strings["beam"])
        )
        cand_uvfits_paths = glob.glob(cand_uvfits_pattern)
        cand_uvfits_path = cand_uvfits_paths[0] if cand_uvfits_paths else None

    cand_cal_path = "/data/seren-01/big/craco/SB{:0>6}/cal/{:0>2}/b{:0>2}.aver.4pol.smooth.npy".format(
        cand_query_strings["sbid"], int(cand_query_strings["beam"]), int(cand_query_strings["beam"]),
    )

    if cand_query_strings["tstart"] is not None:
        cand_clust_path = "/data/seren-{:0>2}/big/craco/SB{:0>6}/scans/{}/{}/{}/clustering_output/candidates.txtb{:0>2}.uniq".format(
            int(cand_query_strings["beam"]) % 10 + 1, cand_query_strings["sbid"],
            cand_query_strings["scan"], cand_query_strings["tstart"],
            cand_query_strings["results"], int(cand_query_strings["beam"])
        )
    else:
        cand_clust_pattern =  "/data/seren-{:0>2}/big/craco/SB{:0>6}/scans/{}/*/{}/clustering_output/candidates.txtb{:0>2}.uniq".format(
            int(cand_query_strings["beam"]) % 10 + 1, cand_query_strings["sbid"], cand_query_strings["scan"], 
            cand_query_strings["results"], int(cand_query_strings["beam"])
        )
        cand_clust_paths = glob.glob(cand_clust_pattern)
        cand_clust_path = cand_clust_paths[0] if cand_clust_paths else None
    
    ### unclustered
    if cand_query_strings["tstart"] is not None:
        cand_unclust_path = "/data/seren-{:0>2}/big/craco/SB{:0>6}/scans/{}/{}/{}/candidates.txtb{:0>2}".format(
            int(cand_query_strings["beam"]) % 10 + 1, cand_query_strings["sbid"],
            cand_query_strings["scan"], cand_query_strings["tstart"],
            cand_query_strings["results"], int(cand_query_strings["beam"])
        )
    else:
        cand_unclust_pattern =  "/data/seren-{:0>2}/big/craco/SB{:0>6}/scans/{}/*/{}/candidates.txtb{:0>2}".format(
            int(cand_query_strings["beam"]) % 10 + 1, cand_query_strings["sbid"], cand_query_strings["scan"], 
            cand_query_strings["results"], int(cand_query_strings["beam"])
        )
        cand_unclust_paths = glob.glob(cand_unclust_pattern)
        cand_unclust_path = cand_unclust_paths[0] if cand_unclust_paths else None

    if cand_uvfits_path: 
        if not os.path.exists(cand_uvfits_path): cand_uvfits_path = None
    cand_query_strings["uvfitspath"] = cand_uvfits_path

    if not os.path.exists(cand_cal_path): cand_cal_path = None
    cand_query_strings["calpath"] = cand_cal_path

    if cand_clust_path:
        if not os.path.exists(cand_clust_path): cand_clust_path = None
    cand_query_strings["clustpath"] = cand_clust_path

    if cand_unclust_path:
        if not os.path.exists(cand_unclust_path): cand_unclust_path = None
    cand_query_strings["unclustpath"] = cand_unclust_path

    # print(cand_query_strings)

    return cand_query_strings.__str__()

def info_table_row(key, value):
    return html.Tr([
        html.Td(html.B(key)),
        html.Td(value),
    ])

def coord_table_row(title, coord, textid):
    return html.Tr([
        html.Td(html.P(html.B(title))),
        html.Td(html.P(coord, id=textid)),
        html.Td(
            dcc.Clipboard(
                target_id=textid, title="copy", 
                style = {"display": "inline-block"}
            )
        ),
    ])

def _decimal_add_unit(coord):
    coord_ = coord.split()
    return "{}d {}d".format(*coord_)


# table for basic information (TODO: add clip-board)
@callback(
    Output("cand_info_table_div", "children"),
    Input("cand_query_strings", "data")
)
def candidate_pipe_info(cand_query_strings):
    cand_query_dict = eval(cand_query_strings)
    # SBID, BEAM, SCAN, TSTART
    basic_info_rows = [
        info_table_row("SBID", cand_query_dict["sbid"]),
        info_table_row("BEAM", "{:0>2}".format(cand_query_dict["beam"])),
        info_table_row("SCAN", cand_query_dict["scan"]),
        info_table_row("TSTART", cand_query_dict["tstart"]),
        info_table_row("RUNNAME", cand_query_dict["results"]),
    ]

    basic_info_tab = dbc.Table(html.Tbody(basic_info_rows), borderless=True, color="primary")

    # DM, boxcar, (l,m), (totalsample)
    search_info_rows = [
        info_table_row("DM", cand_query_dict["dm"]),
        info_table_row("BOXCWIDTH", cand_query_dict["boxcwidth"]),
        info_table_row("PIXEL (l, m)", "({}, {})".format(cand_query_dict["lpix"], cand_query_dict["mpix"])),
        info_table_row("TOTALSAMPLE", cand_query_dict["totalsample"]),
    ]

    search_info_tab = dbc.Table(html.Tbody(search_info_rows), borderless=True, color="secondary")

    try:
        candcoord = SkyCoord("{}d {}d".format(
            cand_query_dict["ra"], cand_query_dict["dec"]
        ))
    except:
        candcoord = None
    
    if candcoord:
        coord_rows = [
            html.Tr([html.Td(html.B("Coordinate")), html.Td(), html.Td()]),
            coord_table_row("RA, DEC ", candcoord.to_string("decimal"), "radec_decimal"),
            coord_table_row("RA, DEC ", candcoord.to_string("hmsdms"), "radec_hmsdms"),
            coord_table_row("GL, GB  ", _decimal_add_unit(candcoord.galactic.to_string("decimal")), "glgb_decimal"),
        ]
    else:
        coord_rows = [
            html.Tr(html.Td(html.B("Coordinate"))),
            html.Tr(html.Td(" ")), html.Tr(html.Td("No available Coordinate")),
            html.Tr(html.Td(" ")), html.Tr(html.Td(" ")),
        ]

    coord_info_tab = dbc.Table(html.Tbody(coord_rows), borderless=True, color="success")

    # coordinate (ra, dec => sexe and degr; gl, gb)

    return dbc.Row([
        dbc.Col(basic_info_tab, width=4),
        dbc.Col(search_info_tab, width=4),
        dbc.Col(coord_info_tab, width=4),
    ])

def _match_psrcat(candcoord):
    """
    check if there is a known pulsar at the source location...
    """
    psrdf = pd.read_json("data/psrcat.json")
    psrcoord = SkyCoord(psrdf["RAJ"], psrdf["DECJ"], unit=(units.hourangle, units.degree))

    psrsep = candcoord.separation(psrcoord).value
    nearby_bool = psrsep < 30./3600.

    nearby_df = psrdf[nearby_bool].copy()
    nearby_df["sep"] = psrsep[nearby_bool] * 3600

    nearby_df = nearby_df.sort_values("sep")

    return nearby_df

def _match_racs(candcoord):
    """
    check if there is a known source in RACS-low catalogue
    """
    v = Vizier(columns=["RAJ2000", "DEJ2000", "Ftot", "Fpk", "+_r"], catalog=["J/other/PASA/38.58/gausscut", "J/other/PASA/38.58/gaussreg"])
    v.ROW_LIMITS = -1

    try:
        racs_query = v.query_region(candcoord, radius="30s")
    except:
        return pd.DataFrame({"NO_INTERNET": []})
    
    if racs_query:
        return racs_query[0].to_pandas()
    return pd.DataFrame({"_r":[], "RAJ2000": [], "DEJ2000": [], "Ftot": [], "Fpk": []})

def _match_simbad(candcoord):
    """
    perform a naive match with simbad
    """
    try:
        sim_query = Simbad.query_region(candcoord, radius="30s")
    except:
        return pd.DataFrame({"NO_INTERNET": []})

    if sim_query:
        sim_df = sim_query.to_pandas()[["MAIN_ID", "RA", "DEC"]].copy()
        sep = candcoord.separation(SkyCoord(sim_df["RA"], sim_df["DEC"], unit=(units.hourangle, units.degree)))
        sim_df["sep"] = sep.value * 3600

        sim_df = sim_df.sort_values("sep")
        return sim_df
    return pd.DataFrame({"MAIN_ID":[], "RA": [], "DEC": [], "sep": []})

def _atnf_psrcat_url(ra, dec, radius=0.01):
    return f"https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?version=1.70&Jname=Jname&Name=Name&P0=P0&DM=DM&W50=W50&S400=S400&S1400=S1400&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names=&ephemeris=short&coords_unit=rajd%2Fdecjd&radius={radius}&coords_1={ra}&coords_2={dec}&raddist=raddist&style=Long+with+last+digit+error&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query&table_bottom.x=66&table_bottom.y=13"

def _vizier_url(candcoord):
    pass

def _simbad_url(ra, dec, radius=30):
    return f"http://simbad.cds.unistra.fr/simbad/sim-coo?Coord={ra}+{dec}&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&Radius={radius}&Radius.unit=arcsec&submit=submit+query&CoordList="

def _format_query_datatable(df, **extra_param):
    return dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in df.columns],
        data = df.to_dict("records"),
        **extra_param
    )


@callback(
    Output("cand_cross_table_div", "children"),
    Input("cand_query_strings", "data")
)
def candidate_cross_info(cand_query_strings):
    cand_query_dict = eval(cand_query_strings)
    try:
        candcoord = SkyCoord("{}d {}d".format(
            cand_query_dict["ra"], cand_query_dict["dec"]
        ))
    except:
        return html.P("No coordinate information available...")

    ### PSR CATALOG
    psrcatdf = _match_psrcat(candcoord)
    if len(psrcatdf) == 0:
        psrcatbtn = dbc.Button("ATNF PSRCAT", color="danger")
    else:
        psrcatbtn = dbc.Button("ATNF PSRCAT", color="success", href=_atnf_psrcat_url(cand_query_dict["ra"], cand_query_dict["dec"]), target="_blank")
    psrcattab = _format_query_datatable(psrcatdf, )

    ### RACS CATALOG
    racsdf = _match_racs(candcoord)
    if len(racsdf) == 0:
        racsbtn = dbc.Button("RACS LOW DR1 CAT", color="danger")
    else:
        racsbtn = dbc.Button("RACS LOW DR1 CAT", color="success")
    racstab = _format_query_datatable(racsdf, )

    ### SIMBAD CATALOG
    simdf = _match_simbad(candcoord)
    if len(simdf) == 0:
        simbtn = dbc.Button("SIMBAD", color="danger", href=_simbad_url(cand_query_dict["ra"], cand_query_dict["dec"]), target="_blank")
    else:
        simbtn = dbc.Button("SIMBAD", color="success", href=_simbad_url(cand_query_dict["ra"], cand_query_dict["dec"]), target="_blank")
    simtab = _format_query_datatable(simdf, page_size=3)
    
    return dbc.Container([
        dbc.Row(psrcatbtn), dbc.Row(psrcattab),
        dbc.Row(html.P("")),
        dbc.Row(racsbtn), dbc.Row(racstab),
        dbc.Row(html.P(["For HIPS MAP: ", dcc.Link("RACS HIPS MAP", href="https://www.atnf.csiro.au/research/RACS/CRACSlow1_I2/", target="_blank")])),
        dbc.Row(simbtn), dbc.Row(simtab),
    ])

def _pltfig2img(fig, **extra_param):
    return html.Img(
        src=fig_to_uri(fig), **extra_param,
    )

# craco candidate related plotting
@callback(
    Output("craco_candidate_filterbank", "children"),
    Output("craco_candidate_images", "children"),
    # Output("cand_filterbank_store", "data"),
    Output("craco_cand_plot_status", "children"),
    Input("craco_cand_plot_btn", "n_clicks"),
    State("cand_query_strings", "data"),
    prevent_initial_call=True
)
def craco_cand_plot(nclick, cand_query_strings):
    cand_query_dict = eval(cand_query_strings)
    try:
        crow = {
            "ra_deg": float(cand_query_dict["ra"]), "dec_deg": float(cand_query_dict["dec"]),
            "dm_pccm3": float(cand_query_dict["dm"]), "total_sample": int(cand_query_dict["totalsample"]),
            "boxc_width": int(cand_query_dict["boxcwidth"]), 
            "lpix": int(cand_query_dict["lpix"]), "mpix": int(cand_query_dict["mpix"])
        }
    except Exception as err:
        # print(err)
        return None, None, "Not enough info..."

    padding = 100

    cand = craco_candidate.Candidate(
        crow = crow,
        uvsource = cand_query_dict["uvfitspath"],
        calibration_file = cand_query_dict["calpath"],
        workdir="./test", padding=padding
    )
    cand.search_output["obstime_sec"] = cand.search_output["total_sample"] * cand.tsamp

    ### make filterbank, image
    cand._calibrate_data(cand.calibration_file)
    cand._rotate_vis()
    cand._normalise_vis()
    cand._normalise_vis(target=False)
    cand._load_burst_filterbank()
    # for images
    cand._dedisperse_block(dm=cand.search_output["dm_pccm3"])
    cand._grid_image_data()

    ### filterbank related plot
    fig, ax = cand.plot_filterbank(dm=0)
    filterbank_zerodm = dbc.Col([
        dbc.Row("dedispered at DM=0", justify="center"),
        dbc.Row(_pltfig2img(fig, style={"width": "100%"})),
        ], width=4
    )

    fig, ax = cand.plot_filterbank(dm=cand.search_output["dm_pccm3"], keepnan=True)
    filterbank_searchdm = dbc.Col([
        dbc.Row("dedispered at DM={:.2f}".format(cand.search_output["dm_pccm3"]), justify="center"),
        dbc.Row(_pltfig2img(fig, style={"width": "100%"})), 
        ], width=4
    )

    fig, ax = cand.plot_dmt()
    filterbank_butterfly = dbc.Col([
        dbc.Row("butterfly plot", justify="center"),
        dbc.Row(_pltfig2img(fig, style={"width": "100%"})), 
        ], width=4
    )

    # interactive filterbank plot
    filterbank_plot, trange_ = cand._dedisperse2tf(dm=cand.search_output["dm_pccm3"], keepnan=True)
    taxis = np.linspace(*trange_, filterbank_plot.shape[1]) * cand.tsamp
    faxis = np.linspace(cand.freqs[0]/1e6, cand.freqs[-1]/1e6, filterbank_plot.shape[0])

    # print()

    heatmapfig = dbc.Col(html.Div(dcc.Graph(
        figure= px.imshow(
            filterbank_plot, x=taxis, y=faxis, aspect="auto", origin="lower"
        ),
        id="cand_filterbank_interactive", 
    )), width=6, className="h-100")

    # lightcurve
    lcfig = go.Figure()
    lcfig.add_traces([
        go.Scatter(x=taxis, y=np.nanmean(filterbank_plot, axis=0), name="mean"),
        go.Scatter(x=taxis, y=np.nanmax(filterbank_plot, axis=0), name="max"),
        go.Scatter(x=taxis, y=np.nanmin(filterbank_plot, axis=0), name="min"),
    ])
    burstlcfig = dbc.Col(html.Div(dcc.Graph(
        figure=lcfig, id="cand_filterbank_interlc", responsive=True
    )))

    pixelmax = np.round(np.nanmax(filterbank_plot), 2) 
    pixelmin = np.round(np.nanmin(filterbank_plot), 2)
    pixelrange = np.round((pixelmax - pixelmin), 2)

    burstlcdiv=dbc.Col(dbc.Row([
        dbc.Row(burstlcfig),
    ], align="center", className="h-100"), width=6)

    colorscale = dbc.Col([
        dbc.Row(dbc.Button(
            "Mannual Color Scale Disabled", id="cand_filterbank_cscale_btn", color="danger", n_clicks=0
        ), style={"margin": "10px"}, justify="center",),
        dbc.Row(dcc.RangeSlider(
            np.round(pixelmin - 0.5*pixelrange,2), np.round(pixelmax + 0.5*pixelrange, 2),
            value=[pixelmin, pixelmax], id="cand_filterbank_slider",
            tooltip={"placement": "bottom", "always_visible": True}
        ), justify="center")
    ], width=4)

    ######### for images
    ### dignostic plots => median and standard deviation
    fig = cand._make_field_image(save=False)
    imgdigplot = dbc.Row([
        _pltfig2img(fig, style={"width": "100%"}),
    ])

    ### make interactive snr image, and interactive zoom image
    fig = px.imshow(
        cand.imgcube.std(axis=0), origin='lower',
    )
    fig.update_layout(title=dict(text="std image (inter)", x=0.5, xanchor="center"))
    snrfig = dbc.Col([
        # dbc.Row("std image", justify="center"),
        dbc.Row(html.Div(dcc.Graph(figure=fig, ))),
    ], width=4)

    # work out the limits during the detection...
    # detection is through imgidx_s to imgidx_e
    _dets = cand.search_output["total_sample"]
    _viss = cand.visrange[0]

    imgidx_e = _dets - _viss + 1
    imgidx_s = imgidx_e - cand.search_output["boxc_width"]
    ## detections are ...
    img_detected = cand.imgcube[imgidx_s:imgidx_e + 1]
    imagestd = cand.imgcube.std()

    fig = px.imshow(
        cand.imgzoomcube, animation_frame=0, 
        zmax=imagestd * 8, zmin=-imagestd,
        origin="lower",
    )
    fig.update_layout(title=dict(text="zoom-in images (inter)\ndetected {}-{}".format(imgidx_s, imgidx_e), x=0.5, xanchor="center"))
    zoomfig = dbc.Col([
        # dbc.Row("zoom-in images", justify="center"),
        dbc.Row(html.Div(dcc.Graph(figure=fig, ))),
    ], width=4)

    fig = px.imshow(
        img_detected.mean(axis=0), # take the mean image over detected period
        origin="lower",
    )
    fig.update_layout(title=dict(text="detection image (inter)", x=0.5, xanchor="center"))
    detectfig = dbc.Col([
        # dbc.Row("detection image", justify="center"),
        dbc.Row(html.Div(dcc.Graph(figure=fig, ))),
    ], width=4)

    os.system("rm uv_data.*.txt")

    return (
        dbc.Container([
            dbc.Row(html.P(html.B("Filterbank Plots"))),
            dbc.Row([filterbank_zerodm, filterbank_searchdm, filterbank_butterfly]),
            dbc.Row([heatmapfig, burstlcdiv]),
            dbc.Row(colorscale, justify="center"),
        ]), # this corresponding to filterbank plots
        dbc.Container([
            dbc.Row(html.P(html.B("Synthesized Images"))),
            dbc.Row(imgdigplot),
            dbc.Row([snrfig, zoomfig, detectfig]),
        ]),
        "done..."
    )

# callback to enable mannual color scales
@callback(
    Output("cand_filterbank_interactive", "figure"),
    Output("cand_filterbank_cscale_btn", "color"),
    State("cand_filterbank_interactive", "figure"),
    Input("cand_filterbank_cscale_btn", "n_clicks"),
    Input("cand_filterbank_slider", "value"),
)
def rescale_filterbank(heatmapfig, nclicks, slidervalue):
    ### update color
    if nclicks == 0: raise PreventUpdate()
    if nclicks % 2 == 0: 
        color="danger"
        if ctx.triggered_id != "cand_filterbank_cscale_btn": 
            raise PreventUpdate()
        heatmapfig["layout"]["coloraxis"].update(dict(cauto=True))
    else: 
        color="success"
        heatmapfig["layout"]["coloraxis"].update(dict(
            cauto=False, cmax=slidervalue[-1], cmin=slidervalue[0]
        ))

    return heatmapfig, color

@callback(
    Output("candidate_all_files", "children"),
    Input("cand_query_strings", "data"),
)
def printout_files(cand_query_strings):
    cand_query_dict = eval(cand_query_strings)
    uvfitsrow = html.Tr([
        html.Td(html.B("UVFITS")),
        html.Td([
            html.Abbr(cand_query_dict["uvfitspath"], id="cand_uvfits_path"),
            dcc.Clipboard(target_id="cand_uvfits_path", title="copy", style = {"display": "inline-block"})
        ]),
    ])
    calrow = html.Tr([
        html.Td(html.B("CAL")),
        html.Td([
            html.Abbr(cand_query_dict["calpath"], id="cand_cal_path"),
            dcc.Clipboard(target_id="cand_cal_path", title="copy", style = {"display": "inline-block"})
        ]),
    ])
    clustrow = html.Tr([
        html.Td(html.B("CLUST")),
        html.Td([
            html.Abbr(cand_query_dict["clustpath"], id="cand_clust_path"),
            dcc.Clipboard(target_id="cand_clust_path", title="copy", style = {"display": "inline-block"})
        ]),
    ])
    unclustrow = html.Tr([
        html.Td(html.B("UNCLUST")),
        html.Td([
            html.Abbr(cand_query_dict["unclustpath"], id="uncand_clust_path"),
            dcc.Clipboard(target_id="cand_unclust_path", title="copy", style = {"display": "inline-block"})
        ]),
    ])
    return dbc.Col(dbc.Table(html.Tbody([
        uvfitsrow, calrow, clustrow, unclustrow
    ]), borderless=True, color="light"), width=10)

### final layout
def layout(**cand_query_strings):

    return html.Div([
        dcc.Location(id="cand_url", refresh=False),
        dcc.Store(id="cand_query_strings", data=cand_query_strings.__str__()),
        dcc.Store(id="cand_filterbank_store"), # make this available later perhaps
        dbc.Container(dbc.Col([
            dbc.Row([
                html.H5("Candidate Pipeline Information")
            ]),
            dbc.Row([
                dbc.Col(html.Div(id="cand_info_table_div"), width=12), # basic information from pipeline
            ]),
                dbc.Row(id="candidate_all_files", justify="center"),
            dbc.Row([
                html.H5("Candidate External Information")
            ]),
            dbc.Row([
                dbc.Col(html.Div(id="cand_cross_table_div"), width=12), # cross check from PSRCAT, RACS, SIMBAD
            ]),
            dbc.Row([
                dbc.Col(html.H5("Candidate CRACO Data"), width=3),
                dbc.Col(dbc.Button("Process", id="craco_cand_plot_btn", color="success"), width=3),
                dbc.Col(dcc.Loading(id="craco_cand_plot_status", fullscreen=False)),
            ]),
            dbc.Row(id="craco_candidate_filterbank"),
            dbc.Row(id="craco_candidate_images"),
        ])),
        dbc.Container(dbc.Row([
            html.Div(id="cand_test_div"),
        ])),
    ])