from dash import (
    Dash, html, callback, 
    dcc, Input, Output, State, 
    ctx, dash_table,
    )

from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash

import subprocess
import os
import glob
import json

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as units
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad

from apputil import (
    fig_to_uri, load_filterbank,
    construct_candinfo,
)

# from craco import craco_candidate
from craft import craco_plan
from craco import craco_cand, fixuvfits
from craco.datadirs import DataDirs, SchedDir, ScanDir, RunDir, format_sbid
from craft.cmdline import strrange

dash.register_page(__name__, path="/candidate", title="CRACO candidate Plotter")

### functions
app = dash.get_app()

def _load_flag_chans(cand_query_strings):
    return None

def _load_flag_ant(cand_query_strings):
    cand_query_dict = eval(cand_query_strings)

    rundir = RunDir(
        cand_query_dict["sbid"], 
        f"""{cand_query_dict["scan"]}/{cand_query_dict["tstart"]}""",
        cand_query_dict["runname"],
    )

    rundir.get_run_params()
    return rundir.flagant # in strrange
    
def intlst_to_strrange(intlst):
    """
    convert an integer list to a strrange to be readable...
    """
    intlst = sorted(list(intlst))

    if len(intlst) == 0: return ""

    splitlst = []
    for i in range(len(intlst)):
        if i == 0: tmp = [intlst[0]]
        elif intlst[i] - intlst[i-1] > 1:
            splitlst.append(tmp)
            tmp = [intlst[i]]
        else: tmp.append(intlst[i])
            
    splitlst.append(tmp)

    strrange_ = []
    for slst in splitlst:
        if len(slst) == 1: strrange_.append(f"{str(slst[0])}")
        else: strrange_.append(f"{str(slst[0])}-{slst[-1]}")
    return ",".join(strrange_)

def _combine_ant_flag(flag1, flag2):
    if isinstance(flag1, str):
        flag1 = strrange(flag1)
    if isinstance(flag2, str):
        flag2 = strrange(flag2)
    
    if flag1 is None: flag1 = []
    if flag2 is None: flag2 = []

    aflag = list(set(flag1 + flag2))
    return [a for a in aflag if a <= 30]

### callbacks
@callback(
    Output("cand_query_strings", "data"),
    Output("cand_flagchan", "value"),
    Output("cand_flagant", "value"),
    Input("cand_query_strings", "data"),
)
def update_cand_query_strings(cand_query_strings):
    cand_query_strings = eval(cand_query_strings)
    ### set some default value in cand_query_strings
    default_values = {
        "sbid": "000000", "beam": "99", "scan": None, "tstart": None,
        "runname": None, "dm": None,
        "lpix": None, "mpix": None, "boxcwidth": None,
        "totalsample": None, "ra": None, "dec": None,
    }

    for key, value in default_values.items():
        if key not in cand_query_strings:
            cand_query_strings[key] = value

    if cand_query_strings["sbid"] is None:
        print("A SBID must be provided...")
        raise PreventUpdate()

    cand_query_strings = construct_candinfo(cand_query_strings)

    ### add antenna flagging
    try:
        if cand_query_strings["uvfitspath"] is not None:
            flagchan = _load_flag_chans(cand_query_strings)
        else:
            flagchan = None
    except:
        flagchan = None
    cand_query_strings["flagchan"] = flagchan

    try:
        if cand_query_strings["uvfitspath"] is not None:
            flagant = _load_flag_ant(cand_query_strings)
        else:
            flagant = "25-30"
    except:
        flagant = "25-30"

    cand_query_strings["flagant"] = flagchan

    # print(cand_query_strings.__str__()) # for debugging purposes

    return cand_query_strings.__str__(), flagchan, flagant

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
        info_table_row("RUNNAME", cand_query_dict["runname"]),
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
    v = Vizier(columns=["GID","RAJ2000", "DEJ2000", "Ftot", "Fpk", "+_r"], catalog=["J/other/PASA/38.58/gausscut", "J/other/PASA/38.58/gaussreg"])
    v.ROW_LIMITS = -1

    try:
        racs_query = v.query_region(candcoord, radius="30s")
    except:
        return pd.DataFrame({"NO_INTERNET": []})
    
    if racs_query:
        return racs_query[0].to_pandas()
    return pd.DataFrame({"_r":[], "GID": [], "RAJ2000": [], "DEJ2000": [], "Ftot": [], "Fpk": []})

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

### function to add rect shape
def _dash_rect_region(x, y, radius):
    linedict = dict(
        type="rect", 
        line=dict(width=5, dash="dot", color="cyan"),
    )
    selection_bound = dict(
        x0=x-radius, x1=x+radius,
        y0=y-radius, y1=y+radius,
    )
    return linedict, selection_bound

# cas/ics candidate related plotting
@app.long_callback(
    output = [
        Output("craco_icscas", "children"),
        Output("craco_icscas_plot_status", "children"),
    ],
    inputs = [
        Input("craco_icscas_plot_btn", "n_clicks"),
        State("cand_query_strings", "data"),
    ],
    running = [
        (Output("craco_icscas_plot_btn", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
# @callback(
#     output = [
#         Output("craco_icscas", "children"),
#         Output("craco_icscas_plot_status", "children"),
#     ],
#     inputs = [
#         Input("craco_icscas_plot_btn", "n_clicks"),
#         State("cand_query_strings", "data"),
#     ],
#     prevent_initial_call=True,
# )
def craco_icscas_plot(nclick, cand_query_strings):
    cand_query_dict = eval(cand_query_strings)

    ### check if ics/cas exists
    icspath = cand_query_dict["icspath"]
    caspath = cand_query_dict["caspath"]
    totalsample = cand_query_dict["totalsample"]

    if totalsample is None: return "totalsample need to be provided to proceed..."
    totalsample = int(float(totalsample))

    if icspath is not None:
        icsdata, taxis, faxis = load_filterbank(icspath, totalsample-75, 150)
        icsfig = px.imshow(
            icsdata[:, 0, :].T, x=taxis, y=faxis, aspect="auto", origin="lower",
        )
        icsfig.add_vline(
            x=totalsample, line_width=1, line_dash="dash", line_color="black",
        )
        icsfig.update_layout(title=dict(text="ICS", x=0.5, xanchor="center"))
        icsfigcol = dbc.Col(html.Div(dcc.Graph(
            figure=icsfig, id="cand_ics_interactive",
        )), width=6, className="h-100")
    else:
        icsfigcol = dbc.Col(width=6, className="h-100")

    if caspath is not None:
        casdata, taxis, faxis = load_filterbank(caspath, totalsample-75, 150)
        casfig = px.imshow(
            casdata[:, 0, :].T, x=taxis, y=faxis, aspect="auto", origin="lower",
        )
        casfig.add_vline(
            x=totalsample, line_width=1, line_dash="dash", line_color="black",
        )
        casfig.update_layout(title=dict(text="CAS", x=0.5, xanchor="center"))
        casfigcol = dbc.Col(html.Div(dcc.Graph(
            figure=casfig, id="cand_cas_interactive",
        )), width=6, className="h-100")
    else:
        casfigcol = dbc.Col(width=6, className="h-100")

    return dbc.Row([icsfigcol, casfigcol]), "Done..."


# craco candidate related plotting
# https://stackoverflow.com/a/75437616
@app.long_callback(
# @callback( # for debug purposes...
    output=[
        Output("craco_candidate_filterbank", "children"),
        Output("craco_candidate_images", "children"),
        # Output("craco_candidate_larger_images", "children"),
        # Output("cand_filterbank_store", "data"),
        Output("craco_cand_plot_status", "children"),
    ],
    inputs=[
        Input("craco_cand_plot_btn", "n_clicks"),
        State("cand_query_strings", "data"),
        State("cand_flagchan", "value"),
        State("cand_flagant", "value"),
        State("cand_pad", "value"),
    ],
    running=[
        (Output("craco_cand_plot_btn", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def craco_cand_plot(nclick, cand_query_strings, flagchan, flagant, padding):
    cand_query_dict = eval(cand_query_strings)
    # print(cand_query_dict)
    try:
        candrow = {
            "ra_deg": float(cand_query_dict["ra"]), "dec_deg": float(cand_query_dict["dec"]),
            "dm_pccm3": float(cand_query_dict["dm"]), "total_sample": int(float(cand_query_dict["totalsample"])),
            "boxc_width": int(float(cand_query_dict["boxcwidth"])), 
            "lpix": int(float(cand_query_dict["lpix"])), "mpix": int(float(cand_query_dict["mpix"]))
        }
    except Exception as err:
        # print(err)
        return None, None, "Not enough info..."

    if padding is None: padding = 50

    ### flag channels...
    if flagchan is not None:
        if flagchan == "": flagchan = None
    if flagant is not None:
        if flagant == "": flagant = None

    rundir = RunDir(
        cand_query_dict["sbid"], 
        f"""{cand_query_dict["scan"]}/{cand_query_dict["tstart"]}""",
        cand_query_dict["runname"],
    )
    try: 
        rundir.get_run_params()
        metafant = rundir.scheddir.flagant # this is a list
    except: 
        metafant = [] # nothing
    # note - wrong antenna will be flagged behind the scene
    # get run antenna based on the metadata
    flagant = intlst_to_strrange(_combine_ant_flag(metafant, flagant))
    # print(flagant)


    #TODO - get metadata path
    try: start_mjd = Time(eval(rundir.startmjd), format="jd", scale="tai")
    except: start_mjd = None

    print(f"running fixuvfits on {cand_query_dict['uvfitspath']}")
    try: fixuvfits.fix_length(cand_query_dict['uvfitspath'])
    except: pass

    cand = craco_cand.Cand(
        uvfits = cand_query_dict['uvfitspath'], 
        metafile = rundir.scheddir.metafile, # if there is nothing, it should give you None
        calfile = cand_query_dict['calpath'],
        # start_mjd = Time(rundir.scheddir.start_mjd, format="jd", scale="tai"),
        start_mjd = start_mjd,
        flagant = flagant, flagchan=flagchan,
        **candrow,
    )

    ### make filterbank, image
    cand.extract_data(padding=padding)
    cand.process_data(zoom_r=10)

    ### filterbank related plot
    fig, ax = cand.plot_filtb(dm=0)
    filterbank_zerodm = dbc.Col([
        dbc.Row("dedispered at DM=0", justify="center"),
        dbc.Row(_pltfig2img(fig, style={"width": "100%"})),
        ], width=4
    )

    fig, ax = cand.plot_filtb(dm=cand.dm_pccm3, keepnan=True)
    filterbank_searchdm = dbc.Col([
        dbc.Row("dedispered at DM={:.2f}".format(cand.dm_pccm3), justify="center"),
        dbc.Row(_pltfig2img(fig, style={"width": "100%"})), 
        ], width=4
    )

    fig, ax = cand.plot_dmt(dmfact=30, ndm=30)
    filterbank_butterfly = dbc.Col([
        dbc.Row("butterfly plot", justify="center"),
        dbc.Row(_pltfig2img(fig, style={"width": "100%"})), 
        ], width=4
    )

    # interactive filterbank plot
    filterbank_plot, trange_ = cand.datasnippet.dedisp_filtb(
        filtb=cand.filtb, dm=cand.dm_pccm3, keepnan=True,
        tstart=cand.canduvfits.datarange[0],
    )
    taxis = np.linspace(*trange_, filterbank_plot.shape[1]) * cand.canduvfits.tsamp
    faxis = np.linspace(
        cand.canduvfits.fmin/1e6, cand.canduvfits.fmax/1e6, filterbank_plot.shape[0]
    )

    fig = px.imshow(
        filterbank_plot, x=taxis, y=faxis, aspect="auto", origin="lower"
    )
    ### add hover data...
    nt = taxis.shape[0]; nfreq = faxis.shape[0]
    fig.update(
        data = [dict(
            customdata=np.repeat(np.arange(nfreq).reshape(-1, 1), nt, axis=1),
            hovertemplate="tsec %{x:.0f} <br>freq %{y:.3f} <br>chan %{customdata}",
        )]
    )
    fig.add_vline(
        x=cand.total_sample * cand.canduvfits.tsamp,
        line_width=1, line_dash="dash", line_color="black",
    ) # add vertical line to indicate the burst time
    heatmapfig = dbc.Col(html.Div(dcc.Graph(
        figure=fig,
        id="cand_filterbank_interactive", 
    )), width=6, className="h-100")

    # lightcurve
    lcfig = go.Figure()
    lcfig.add_traces([
        go.Scatter(x=taxis, y=np.nanmean(filterbank_plot, axis=0), name="mean"),
        go.Scatter(x=taxis, y=np.nanmax(filterbank_plot, axis=0), name="max"),
        go.Scatter(x=taxis, y=np.nanmin(filterbank_plot, axis=0), name="min"),
    ])
    lcfig.add_vline(
        x=cand.total_sample * cand.canduvfits.tsamp,
        line_width=1, line_dash="dash", line_color="red",
    )
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
    fig = cand.plot_diagnostic_images()
    imgdigplot = dbc.Row([
        _pltfig2img(fig, style={"width": "100%"}),
    ])

    ### make interactive snr image, and interactive zoom image
    ### make standard deviation normalisation
    stdimg = cand.imgcube.std(axis=0)

    linedict, selection_bound_large = _dash_rect_region(
        cand.lpix, cand.mpix, 10
    )
    linedict, selection_bound_small = _dash_rect_region(10, 10, 5)
    fig = px.imshow(stdimg, origin='lower', )
    fig.add_shape(linedict, **selection_bound_large)
    fig.update_layout(title=dict(text="std image (inter)", x=0.5, xanchor="center"))
    snrfig = dbc.Col([
        # dbc.Row("std image", justify="center"),
        dbc.Row(html.Div(dcc.Graph(figure=fig, ))),
    ], width=4)

    # work out the limits during the detection...
    # detection is through imgidx_s to imgidx_e
    ## detections are ...
    imagestd = cand.imgcube.std()
    stdmed = np.median(stdimg)

    fig = px.imshow(
        cand.imgzoomcube, animation_frame=0, 
        zmax=imagestd * 8, zmin=-imagestd,
        origin="lower",
    )
    fig.add_shape(linedict, **selection_bound_small)
    fig.update_layout(
        title=dict(text="zoom-in images (inter)\ndetected {}-{}".format(
            cand.image_start_index, cand.image_end_index
        ), 
        x=0.5, xanchor="center")
    )
    zoomfig = dbc.Col([
        # dbc.Row("zoom-in images", justify="center"),
        dbc.Row(html.Div(dcc.Graph(figure=fig, ))),
    ], width=4)

    img_detected = cand.imgcube[
        cand.image_start_index:cand.image_end_index + 1
    ]

    fig = px.imshow(
        img_detected.mean(axis=0) / stdmed / np.sqrt(img_detected.shape[0]), # take the mean image over detected period
        origin="lower",
    )
    fig.add_shape(linedict, **selection_bound_large)
    fig.update_layout(title=dict(text="detection image (inter)", x=0.5, xanchor="center"))
    detectfig = dbc.Col([
        # dbc.Row("detection image", justify="center"),
        dbc.Row(html.Div(dcc.Graph(figure=fig, ))),
    ], width=4)

    ### once we have a light version craco_plan,
    ### considering move larger image to here

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

### plot for larger region
@app.long_callback(
    output = [
        Output("craco_candidate_larger_images_div", "children"),
        Output("craco_cand_large_plot_status", "children"),
    ],
    inputs = [
        Input("craco_cand_large_plot_btn", "n_clicks"),
        State("cand_query_strings", "data"),
        State("cand_flagchan", "value"),
        State("cand_flagant", "value"),
        State("cand_pad", "value"),
    ],
    running = [
        (Output("craco_cand_large_plot_btn", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def craco_cand_large_plot(nclick, cand_query_strings, flagchan, flagant, padding):
    # again move it to the top callback once we have a light version plan
    cand_query_dict = eval(cand_query_strings)
    candrow = {
        "ra_deg": float(cand_query_dict["ra"]), "dec_deg": float(cand_query_dict["dec"]),
        "dm_pccm3": float(cand_query_dict["dm"]), "total_sample": int(float(cand_query_dict["totalsample"])),
        "boxc_width": int(float(cand_query_dict["boxcwidth"])), 
        "lpix": 128 + int(float(cand_query_dict["lpix"])), 
        "mpix": 128 + int(float(cand_query_dict["mpix"]))
    }

    if padding is None: padding = 50

    ### flag channels...
    if flagchan is not None:
        if flagchan == "": flagchan = None
    if flagant is not None:
        if flagant == "": flagant = None

    rundir = RunDir(
        cand_query_dict["sbid"], 
        f"""{cand_query_dict["scan"]}/{cand_query_dict["tstart"]}""",
        cand_query_dict["runname"],
    )
    rundir.get_run_params()
    # note - wrong antenna will be flagged behind the scene
    # get run antenna based on the metadata
    metafant = rundir.scheddir.flagant # this is a list
    flagant = intlst_to_strrange(_combine_ant_flag(metafant, flagant))
    # print(flagant)


    cand = craco_cand.Cand(
        uvfits = cand_query_dict['uvfitspath'], 
        metafile = rundir.scheddir.metafile, 
        calfile = cand_query_dict['calpath'],
        # start_mjd = Time(rundir.scheddir.start_mjd, format="jd", scale="tai"),
        start_mjd = Time(eval(rundir.startmjd), format="jd", scale="tai"),
        flagant = flagant, flagchan=flagchan,
        **candrow,
    )

    ### make filterbank, image
    cand.extract_data(padding=padding)
    lplan = craco_plan.PipelinePlan(
        cand.canduvfits.datauvsource, "--ndm 2 --npix 512 --fov 2.2d",
    )
    cand.process_data(plan=lplan, zoom_r=10)

    ### three interactive images again... but a larger version
    # _mpix = 128 + cand.mpix; _lpix = 128 + cand.lpix
    linedict, selection_bound_large = _dash_rect_region(cand.lpix,cand.mpix, 10)
    linedict, selection_bound_small = _dash_rect_region(10, 10, 5)

    ### snr image
    stdimg = cand.imgcube.std(axis=0)

    fig = px.imshow(stdimg, origin="lower")
    fig.add_shape(linedict, **selection_bound_large)
    fig.update_layout(title=dict(text="std image (inter)", x=0.5, xanchor="center"))
    snrfig = dbc.Col([dbc.Row(html.Div(dcc.Graph(figure=fig)))], width=4)

    ### detection things

    img_detected = cand.imgcube[
        cand.image_start_index:cand.image_end_index + 1
    ]
    imagestd = cand.imgcube.std()
    stdmed = np.median(stdimg)

    ### make new cube...
    # cand.imgzoomcube = cand.imgcube[
    #     :, cand._workout_slice_w_center(_mpix, 512, 10), cand._workout_slice_w_center(_lpix, 512, 10)
    # ] # this is a bit hard coding here...

    fig = px.imshow(
        cand.imgzoomcube, animation_frame=0, zmax=imagestd * 8, zmin=-imagestd, origin="lower"
    )
    fig.add_shape(linedict, **selection_bound_small)
    fig.update_layout(title=dict(text="zoom-in images (inter)\ndetected {}-{}".format(cand.image_start_index, cand.image_end_index), x=0.5, xanchor="center"))
    zoomfig = dbc.Col([dbc.Row(html.Div(dcc.Graph(figure=fig, )))], width=4)

    fig = px.imshow(
        img_detected.mean(axis=0) / stdmed / np.sqrt(img_detected.shape[0]), # take the mean image over detected period
        origin="lower",
    )
    fig.add_shape(linedict, **selection_bound_large)
    fig.update_layout(title=dict(text="detection image (inter)", x=0.5, xanchor="center"))
    detectfig = dbc.Col([dbc.Row(html.Div(dcc.Graph(figure=fig, ))),], width=4)

    os.system("rm uv_data.*.txt")

    return dbc.Row([snrfig, zoomfig, detectfig]), "done..."

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
    metarow = html.Tr([
        html.Td(html.B("METAFILE")),
        html.Td([
            html.Abbr(cand_query_dict["metapath"], id="cand_meta_path"),
            dcc.Clipboard(target_id="cand_meta_path", title="copy", style = {"display": "inline-block"})
        ]),
    ])
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
            html.Abbr(cand_query_dict["unclustpath"], id="cand_unclust_path"),
            dcc.Clipboard(target_id="cand_unclust_path", title="copy", style = {"display": "inline-block"})
        ]),
    ])
    return dbc.Col(dbc.Table(html.Tbody([
        metarow, uvfitsrow, calrow, clustrow, unclustrow
    ]), borderless=True, color="light"), width=10)

def _back_cand_btn(cand_query_strings, unique=True):
    title = "UNIQUE CANDPAGE" if unique else "UNCLUST CANDPAGE"
    try:
        sbid = cand_query_strings["sbid"]
        beam = cand_query_strings["beam"]
        runname = cand_query_strings["runname"]
        if "tstart" not in cand_query_strings:
            scanpath = None
        else:
            scanpath = "{}/{}/{}".format(
                cand_query_strings["scan"], cand_query_strings["tstart"],
                cand_query_strings["runname"]
            )
            return dcc.Link(
                title,
                href="/beam?sbid={}&beam={}&unique={}&scanpath={}".format(
                    sbid, beam, unique, scanpath
                ),
                target="_blank",
            )
        return dcc.Link(
            title, href="/beam?sbid={}&beam={}&results={}&unique={}".format(
                sbid, beam, runname, unique
            ), target="_blank",
        )
    except Exception as error:
        # print(error)
        return None

###### keep and rclone stuff here
@callback(
    inputs = [
        Input("craco_keep_btn", "n_clicks"),
        State("craco_keep_option", "value"),
        State("archive_comment", "value"),
        State("cand_query_strings", "data"),
    ],
    output = [
        Output("craco_keep_status", "children"),
    ],
    prevent_initial_call=True,
)
# def keep_files(nclick, optvalue, cand_query_strings):
def archive_candidate_data(nclick, optvalue, comment, cand_query_strings):
    cand_query_dict = eval(cand_query_strings)
    ### push candidate to candidate file
    _store_candidate(cand_query_dict)
    ### update archive scans
    # scans = _check_archive_file(cand_query_dict, keepopt=optvalue)
    scans = _make_archive_file(cand_query_dict, keepopt=optvalue)
    # print(cand_query_dict)
    print(scans)
    ### here is the part you are gonna launch several tsp jobs...
    for scan in scans:
        pid = _rclone_scan(cand_query_dict, scan, comment)

    return [f"""{len(scans)} scans to be archived - {", ".join(scans)}"""]

# functions for archiving files
def _rclone_scan(cand_query_dict, scan, comment):
    ecopy = os.environ.copy()
    ecopy['TS_SOCKET'] = "/data/craco/craco/tmpdir/queues/archive"

    sbid = cand_query_dict["sbid"]
    # scan = f"""{cand_query_dict["scan"]}/{cand_query_dict["tstart"]}"""
    beam = cand_query_dict["beam"]
    node = int(cand_query_dict["uvfitspath"].split("/")[2][-2:])
    cmd = "/CRACO/SOFTWARE/craco/craftop/softwares/craco_run/archive_scan.sh"

    if comment is None:
        comment = "NO COMMENT..."

    archive_cmd = f"tsp {cmd} {sbid} {scan} {beam} {node} '{comment}'"

    p = subprocess.run([archive_cmd], shell=True, capture_output=True, text=True, env=ecopy)
    return int(p.stdout.strip())

def _make_archive_file(cand_query_dict, keepopt=1):
    """
    the format of the archive file should be stored under 
    /CRACO/DATA_00/craco/SB0xxxxx/ARCHIVE

    ARCHIVE file should be in a format of csv, following are the columns
    scan,tstart,beam
    """
    scheddir = SchedDir(cand_query_dict["sbid"])
    beam = cand_query_dict["beam"]
    archive_fname = f"{scheddir.sched_head_dir}/ARCHIVE"
    if not os.path.exists(archive_fname):
        fp = open(archive_fname, "w")
        fp.write("scan,tstart,beam\n")
        fp.close()
    fp = open(archive_fname, "a")
    scans = _check_archive_file(cand_query_dict, keepopt=keepopt)
    for scan in scans:
        _write_scan(fp, scan, beam)
    ### check if this scan are already there
    fp.close()
    return scans

def __get_archive_scans(cand_query_dict, keepopt=1):
    if keepopt == 1:
        return [f"""{cand_query_dict["scan"]}/{cand_query_dict["tstart"]}"""]
    scan = cand_query_dict["scan"]
    scheddir = SchedDir(cand_query_dict["sbid"])
    scans = [i for i in scheddir.scans if i.startswith(scan)]
    return scans

def __check_df_scans(df, beam, scans):
    beamdf = df[df["beam"] == int(beam)]
    return [scan for scan in scans if not __check_df_scan(beamdf, scan)]

def __check_df_scan(df, scan):
    scan, tstart = scan.split("/")
    dfscan = df[(df["scan"] == int(scan)) & (df["tstart"] == int(tstart))]
    if len(dfscan) == 0: return False
    return True

def _check_archive_file(cand_query_dict, keepopt=1):
    """
    this function checks if there is any new scan need to be archived
    """
    scheddir = SchedDir(cand_query_dict["sbid"])
    beam = cand_query_dict["beam"]
    archive_fname = f"{scheddir.sched_head_dir}/ARCHIVE"
    ### get a list of scans should be archived first
    scans = __get_archive_scans(cand_query_dict, keepopt=keepopt)
    ### check if these scans are in database
    scankept_df = pd.read_csv(archive_fname)
    return __check_df_scans(scankept_df, beam, scans)

def _write_scan(fp, scan, beam):
    """
    write scan information to the ARCHIVE file...
    """
    scan, tstart = scan.split("/")
    fp.write(f"{scan},{tstart},{beam}\n")

def _store_candidate(cand_query_dict):
    scheddir = SchedDir(cand_query_dict["sbid"])
    cand_fname = f"{scheddir.sched_head_dir}/KEEP_CAND"
    if not os.path.exists(cand_fname):
        fp = open(cand_fname, "w")
        fp.write("scan,tstart,beam,dm,boxcwidth,lpix,mpix,totalsample,ra,dec\n")
        fp.close()
    
    fp = open(cand_fname, "a")
    d = cand_query_dict
    fp.write(f"""{d["scan"]},{d["tstart"]},{d["beam"]},{d["dm"]},{d["boxcwidth"]},{d["lpix"]},{d["mpix"]},{d["totalsample"]},{d["ra"]},{d["dec"]}\n""")
    fp.close()

### final layout
def layout(**cand_query_strings):
    # print(cand_query_strings)

    return html.Div([
        dcc.Location(id="cand_url", refresh=False),
        dcc.Store(id="cand_query_strings", data=cand_query_strings.__str__()),
        dcc.Store(id="cand_filterbank_store"), # make this available later perhaps
        dbc.Container(dbc.Col([
            dbc.Row([
                dbc.Col(html.H5("Candidate Pipeline Information"), width=3),
                dbc.Col(_back_cand_btn(cand_query_strings, unique=True), width=3),
                dbc.Col(_back_cand_btn(cand_query_strings, unique=False), width=3),
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
                dbc.Col(html.H5("ICS/CAS"), width=3),
                dbc.Col(dbc.Button("Process", id="craco_icscas_plot_btn", color="success"), width=3),
                dbc.Col(dcc.Loading(id="craco_icscas_plot_status", fullscreen=False)),
            ]),
            dbc.Row(id="craco_icscas"),
            dbc.Row([
                dbc.Col(html.H5("Candidate CRACO Data"), width=3),
                dbc.Col(dbc.Button("Process", id="craco_cand_plot_btn", color="success"), width=3),
                dbc.Col(dcc.Loading(id="craco_cand_plot_status", fullscreen=False)),
            ]),
            dbc.Row([
                dbc.Col(dbc.Row([
                    dbc.Col(html.P("FANT"), width=3), 
                    dbc.Col(dcc.Input(id="cand_flagant", type="text", placeholder="strrange"), width=6),
                ]), width=3),
                dbc.Col(dbc.Row([
                    dbc.Col(html.P("FCHAN"), width=3), 
                    dbc.Col(dcc.Input(id="cand_flagchan", type="text", placeholder="strrange"), width=6),
                ]), width=3),
                dbc.Col(dbc.Row([
                    dbc.Col(html.P("PADDING"), width=3), 
                    dbc.Col(dcc.Input(id="cand_pad", type="number", value=75, placeholder="number - default 50"), width=6),
                ]), width=3),
            ]),
            dbc.Row(id="craco_candidate_filterbank"),
            dbc.Row(id="craco_candidate_images"),
            dbc.Row([
                dbc.Col(html.P(html.B("Synthesized Images (npix=512)")), width=3),
                dbc.Col(dbc.Button("Process", id="craco_cand_large_plot_btn", color="success"), width=3),
                dbc.Col(dcc.Loading(id="craco_cand_large_plot_status", fullscreen=False)),
            ]),
            dbc.Row(dbc.Container([
                dbc.Container(id="craco_candidate_larger_images_div"),
            ]), id="craco_candidate_larger_images"),
            ##### layout for keep buttons
            html.Hr(),
            dbc.Row(dbc.Container([
                dbc.Col(dbc.Button("KEEP", id="craco_keep_btn", color="success"), width=3),
                dbc.Col(dbc.Row([
                    dbc.Col(html.P("comment"), width=3),
                    dbc.Col(dcc.Input(id="archive_comment", type="text", placeholder="FRB..."), width=6),
                ])),
                dbc.Col(dbc.RadioItems(
                    options=[
                        {"label": "scan", "value": 1},
                        {"label": "sbid", "value": 2},
                    ],
                    value=1, id="craco_keep_option", inline=True,
                 ),width=9),
                dbc.Col(dcc.Loading(id="craco_keep_status", fullscreen=False)),
            ]), )
        ])),
        dbc.Container(dbc.Row([
            html.Div(id="cand_test_div"),
        ])),
    ])