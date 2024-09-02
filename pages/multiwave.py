import os
import subprocess

from dash import Dash, html, callback, dcc, Input, Output,State, ctx, dash_table
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash
import plotly.express as px

from apputil import skyview_allsurveys, fig_to_uri, decam_allsurveys

from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units
from astropy.visualization import SqrtStretch, LogStretch, SinhStretch, PowerStretch
from astropy.visualization import ZScaleInterval, PercentileInterval, MinMaxInterval

from astroquery.skyview import SkyView

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np

dash.register_page(__name__, path="/multiwave", title="CRACO Multiwave Overlay")

app = dash.get_app()

### on for DES
def mul_input_layout(query):
    ra = query.get("ra")
    dec = query.get("dec")
    ra_err = query.get("raerr")
    dec_err = query.get("decerr")

    input_row = dbc.Row([
        dbc.Col(html.B("RA"), width=1, align="center"),
        dbc.Col(dbc.Input(id="mul_ra_input", value=ra), width=2, align="center"),
        dbc.Col(html.B("RAERR (arcsec)"), width=2, align="center"),
        dbc.Col(dbc.Input(id="mul_raerr_input", value=ra_err), width=1, align="center"),
        dbc.Col(html.B("DEC"), width=1, align="center"),
        dbc.Col(dbc.Input(id="mul_dec_input", value=dec), width=2, align="center"),
        dbc.Col(html.B("DECERR (arcsec)"), width=2, align="center"),
        dbc.Col(dbc.Input(id="mul_decerr_input", value=dec_err), width=1, align="center"),
    ], style={'marginBottom': '0.5em', "marginTop": "1.0em"})

    return input_row

### callback to validate inputs...
@callback(
    output=[
        Output("mul_ra_input", "invalid"),
        Output("mul_raerr_input", "invalid"),
        Output("mul_dec_input", "invalid"),
        Output("mul_decerr_input", "invalid"),
        Output("mul_input_store", "data"),
    ],
    inputs = [
        Input("mul_ra_input", "value"),
        Input("mul_raerr_input", "value"),
        Input("mul_dec_input", "value"),
        Input("mul_decerr_input", "value"),
    ],
    # prevent_initial_call=True,
)
def mul_verify_inputs(ra, raerr, dec, decerr):
    try: ra = float(ra); ra_unit = units.degree
    except: ra_unit = units.hourangle
    try: ra_value = Angle(ra, unit=ra_unit).degree; rainvalid = False
    except: rainvalid = True

    try: raerr = float(raerr); raerrinvalid = False
    except: raerrinvalid = True

    try: dec_value = Angle(dec, unit=units.degree).degree; decinvalid = False
    except: decinvalid = True
    
    try: decerr = float(decerr); decerrinvalid = False
    except: decerrinvalid = True

    if rainvalid or raerrinvalid or decinvalid or decerrinvalid:
        data = "None" 
    else:
        data = dict(ra=ra_value, raerr=raerr, dec=dec_value, decerr=decerr)
    # print(data)
    return rainvalid, raerrinvalid, decinvalid, decerrinvalid, data.__str__()

### add skyview input, survey, radius etc.
def mul_skyview_layout():
    # surveys = SkyView.list_surveys() # I plan to put it to apputil...
    surveys = skyview_allsurveys.keys()
    selectsurvey = [dict(label=survey, value=survey) for survey in surveys]

    input_row = dbc.Row([
        dbc.Col(html.B("Wavelength"), width=2, align="center"),
        dbc.Col(dbc.Select(id="mul_skyview_wavelength_input", options=selectsurvey, value="Optical:DSS"), width=4),
        dbc.Col(html.B("Survey"), width=2, align="center"),
        dbc.Col(dbc.Select(id="mul_skyview_survey_input", ), width=4),
    ], style={'marginBottom': '0.5em'})

    display_row = dbc.Row([
        dbc.Col(id="mul_skyview_display_small_col", width=6),
        dbc.Col(id="mul_skyview_display_large_col", width=6),
    ])

    return input_row, display_row

@callback(
    output = [
        Output("mul_skyview_survey_input", "options"),
        Output("mul_skyview_survey_input", "value"),
    ],
    inputs = [
        Input("mul_skyview_wavelength_input", "value"),
    ],
    # prevent_initial_call=True,
)
def mul_get_survey_opt(wavelength_input):
    surveys = skyview_allsurveys[wavelength_input]
    selectsurvey = [dict(label=survey, value=survey) for survey in surveys]
    default_value = surveys[0]

    return selectsurvey, default_value

### download image from internet
def download_skyview(ra, dec, radius, survey):
    hdulists = SkyView.get_images(position=f'{ra} {dec}',survey=[survey], radius=radius*units.arcsec,)
    return hdulists[0]

def get_decam_url(ra, dec, radius, survey, size=512):
    pixscale= 2 * radius / size
    if pixscale < decam_allsurveys[survey]["res"]:
        pixscale = decam_allsurveys[survey]["res"]
        size = int(2 * radius / pixscale)
    baseurl = "https://www.legacysurvey.org/viewer/cutout.fits"
    query = f"ra={ra}&dec={dec}&layer={survey}&size={size}&pixscale={pixscale}"
    return f"{baseurl}?{query}"

def download_decam(ra, dec, radius, survey, size=512):
    url = get_decam_url(ra, dec, radius, survey, size=size)
    hdulst = fits.open(url)
    return hdulst

### plot scripts...
def mul_plot_overlay(hdulst, ra, dec, raerr=None, decerr=None, ):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection=WCS(hdulst[0].header).celestial)

    if raerr is None: raerr = 10.
    if decerr is None: decerr = 10.

    raerr = raerr / 3600.; decerr = decerr / 3600.

    ell = Ellipse(
        (ra, dec), 2*raerr / np.cos(np.deg2rad(dec)),
        2*decerr, angle=0, # this is for position angle
        edgecolor="white", lw=1, facecolor="none",
        transform = ax.get_transform("fk5")
    )

    ax.imshow(np.squeeze(hdulst[0].data), origin="lower", aspect="auto", cmap="inferno",)
    ax.add_patch(ell)

    ax.coords[0].set_axislabel("Right Ascension (J2000)")
    ax.coords[1].set_axislabel("Declination (J2000)")
    ax.coords[0].set_ticklabel_position("top")
    ax.coords[1].set_ticklabel_position("right")

    return fig

def normalise_data(data):
    interval = PercentileInterval(100.)
    stretch = LogStretch(100)
    transform = interval + stretch
    return transform(data)

def mul_plot_decam_overlay(hdulst, ra, dec, raerr=None, decerr=None, ):
    ### if NAXIS is more than 2, use tricolor...
    header = hdulst[0].header
    naxis = header["NAXIS"]
    if naxis == 2:
        return mul_plot_overlay(hdulst, ra, dec, raerr=raerr, decerr=decerr, )
    ### check how many band are there...
    nband = header["NAXIS3"]
    if nband == 1:
        return mul_plot_overlay(hdulst, ra, dec, raerr=raerr, decerr=decerr, )
    bands = [header[f"BAND{i}"] for i in range(nband)]
    plotband = ""; plotdata = []
    for i in range(3):
        plotdata.append(hdulst[0].data[i%nband])
        plotband += bands[i%nband]
    ### normalise the data
    normdata = [normalise_data(data) for data in plotdata]
    rgbdata = np.stack(normdata, axis=-1)

    ### plotting here
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection=WCS(header).celestial)

    if raerr is None: raerr = 10.
    if decerr is None: decerr = 10.

    raerr = raerr / 3600.; decerr = decerr / 3600.

    ell = Ellipse(
        (ra, dec), 2*raerr / np.cos(np.deg2rad(dec)),
        2*decerr, angle=0, # this is for position angle
        edgecolor="cyan", lw=1, facecolor="none",
        transform = ax.get_transform("fk5")
    )

    ax.imshow(rgbdata, origin="lower", aspect="auto",)
    ax.add_patch(ell)

    ax.coords[0].set_axislabel("Right Ascension (J2000)")
    ax.coords[1].set_axislabel("Declination (J2000)")
    ax.coords[0].set_ticklabel_position("top")
    ax.coords[1].set_ticklabel_position("right")

    ax.text(
        0.05, 0.95, plotband, color="cyan", transform=ax.transAxes,
        va="center", ha="left", weight="bold"
    )

    return fig

@app.long_callback(
# @callback(
    output = [
        Output("mul_skyview_display_small_col", "children"),
        Output("mul_skyview_display_large_col", "children"),
        Output("mul_skyview_plot_load", "children"),
    ],
    inputs = [
        Input("mul_skyview_btn", "n_clicks"),
        State("mul_skyview_radius_input", "value"),
        State("mul_skyview_survey_input", "value"),
        State("mul_input_store", "data"),
    ],
    running = [
        (Output("mul_skyview_btn", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def mul_skyview_plot(nclick, radius, survey, inputs):

    if nclick is None: raise PreventUpdate
    inputs = eval(inputs)
    ra = float(inputs.get("ra")); dec = float(inputs.get("dec"))
    raerr = float(inputs.get("raerr")); decerr = float(inputs.get("decerr"))
    radius = float(radius)

    message = "download failed..."
    try:
        hdulst_small = download_skyview(ra, dec, radius, survey)
        message = "plot failed..."
        fig = mul_plot_overlay(hdulst_small, ra, dec, raerr, decerr, )
        small_fig = html.Img(src=fig_to_uri(fig), style={"width": "100%"})
    except Exception as err:
        small_fig = html.B(f"cannot plot small figure... {message}\n with error: {err}")

    message = "download failed..."
    try:
        hdulst_large = download_skyview(ra, dec, 2 * radius, survey)
        message = "plot failed..."
        fig = mul_plot_overlay(hdulst_large, ra, dec, raerr, decerr, )
        large_fig = html.Img(src=fig_to_uri(fig), style={"width": "100%"})
    except Exception as err:
        large_fig = html.B(f"cannot plot large figure... {message}\n with error: {err}")

    return small_fig, large_fig, "Done!"

### add skyview input, survey, radius etc.
def mul_decam_layout():
    # surveys = SkyView.list_surveys() # I plan to put it to apputil...
    surveys = decam_allsurveys.keys()
    selectsurvey = [dict(label=survey, value=survey) for survey in surveys]

    input_row = dbc.Row([
        dbc.Col(html.B("Please select avaiable survey on DECAM site"), width=6, align="center"),
        dbc.Col(html.B("Survey"), width=2, align="center"),
        dbc.Col(dbc.Select(id="mul_decam_survey_input", options=selectsurvey, value="ls-dr10"), width=4, align="center"),
    ], style={'marginBottom': '0.5em'})

    display_row = dbc.Row([
        dbc.Col(id="mul_decam_display_small_col", width=6),
        dbc.Col(id="mul_decam_display_large_col", width=6),
    ])

    return input_row, display_row

@app.long_callback(
# @callback(
    output = [
        Output("mul_decam_display_small_col", "children"),
        Output("mul_decam_display_large_col", "children"),
        Output("mul_decam_plot_load", "children"),
    ],
    inputs = [
        Input("mul_decam_btn", "n_clicks"),
        State("mul_decam_radius_input", "value"),
        State("mul_decam_survey_input", "value"),
        State("mul_input_store", "data"),
    ],
    running = [
        (Output("mul_decam_btn", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def mul_decam_plot(nclick, radius, survey, inputs):

    if nclick is None: raise PreventUpdate
    inputs = eval(inputs)
    ra = float(inputs.get("ra")); dec = float(inputs.get("dec"))
    raerr = float(inputs.get("raerr")); decerr = float(inputs.get("decerr"))
    radius = float(radius)

    message = "download failed..."
    try:
        hdulst_small = download_decam(ra, dec, radius, survey)
        message = "plot failed..."
        fig = mul_plot_decam_overlay(hdulst_small, ra, dec, raerr, decerr, )
        small_fig = html.Img(src=fig_to_uri(fig), style={"width": "100%"})
    except Exception as err:
        downurl = get_decam_url(ra, dec, radius, survey)
        msg = [
            html.P(f"cannot plot small figure... {message}"),
            html.P(f"with error: {err}"),
            html.P("please check decam viewer: https://www.legacysurvey.org/viewer"),
            html.P(["OR ", html.A("CLICK FOR CUTOUT URL", href=downurl, target="_blank")])
        ]
        small_fig = html.B(msg)

    message = "download failed..."
    try:
        hdulst_large = download_decam(ra, dec, 2 * radius, survey)
        message = "plot failed..."
        fig = mul_plot_decam_overlay(hdulst_large, ra, dec, raerr, decerr, )
        large_fig = html.Img(src=fig_to_uri(fig), style={"width": "100%"})
    except Exception as err:
        downurl = get_decam_url(ra, dec, 2*radius, survey)
        msg = [
            html.P(f"cannot plot large figure... {message}"),
            html.P(f"with error: {err}"),
            html.P("please check decam viewer: https://www.legacysurvey.org/viewer"),
            html.P(["OR ", html.A("CLICK FOR CUTOUT URL", href=downurl, target="_blank")])
        ]
        large_fig = html.B(msg)

    return small_fig, large_fig, "Done!"


def layout(**query):
    return [
        dcc.Store(id="mul_input_store"),
        dbc.Container([
            html.H2("CRACO Multiwavelength Overlay Page (Beta)"), 
            mul_input_layout(query)
        ], style={'marginBottom': '2.0em', 'marginTop': '1.0em'},),
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H4("Query Survey From SkyView"), width=5, align="center"), 
                dbc.Col(dbc.Button("Go!", color="success", id="mul_skyview_btn"), width=2, align="center"),
                dbc.Col(dcc.Loading(id="mul_skyview_plot_load", fullscreen=False), width=1, align="center"),
                dbc.Col(html.B("radius (arcsec)"), width=2, align="center"),
                dbc.Col(dbc.Input(value=30, id="mul_skyview_radius_input"), width=2, align="center")
            ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'}),
            *mul_skyview_layout(),
        ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'},),
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H4("Query Survey From DECAM"), width=5, align="center"), 
                dbc.Col(dbc.Button("Go!", color="success", id="mul_decam_btn"), width=2, align="center"),
                dbc.Col(dcc.Loading(id="mul_decam_plot_load", fullscreen=False), width=1, align="center"),
                dbc.Col(html.B("radius (arcsec)"), width=2, align="center"),
                dbc.Col(dbc.Input(value=30, id="mul_decam_radius_input"), width=2, align="center")
            ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'}),
            *mul_decam_layout(),
            dbc.Row(
                dbc.Col("For more details, please check DECAM viewer at https://www.legacysurvey.org/viewer", align="center"), 
                style={'marginBottom': '1.0em', 'marginTop': '0.5em'}
            )
        ], style={'marginBottom': '1.0em', 'marginTop': '2.0em'},),
    ]


# example link - http://localhost:8025/multiwave?ra=05:34:11.56&raerr=0.786&dec=-59:40:25.086&decerr=0.581