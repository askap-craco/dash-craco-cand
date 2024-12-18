import os
import subprocess

from dash import Dash, html, callback, dcc, Input, Output,State, ctx, dash_table
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash
import plotly.express as px

from apputil import construct_candinfo

from astropy.coordinates import SkyCoord, Angle
from astropy import units

from craft import uvfits
from craco.craco_run.slackpost import SlackPostManager

dash.register_page(__name__, path="/localise", title="Source Localisation")

SLACK_CHANNEL = "C07BEEDLX8W"

@callback(
    output=[
        Output("loc_ra_input", "invalid"),
        Output("loc_dec_input", "invalid"),
        Output("loc_file_info", "children"),
        Output("loc_job_title", "children"),
        Output("local_cand_query_strings", "data"),
    ],
    inputs=[
        Input("loc_verify_btn", "n_clicks"),
        State("loc_sbid_input", "value"),
        State("loc_scan_input", "value"),
        State("loc_tstart_input", "value"),
        State("loc_beam_input", "value"),
        State("loc_runname_input", "value"),
        State("loc_dm_input", "value"),
        State("loc_boxcwidth_input", "value"),
        State("loc_totalsample_input", "value"),
        State("loc_ra_input", "value"),
        State("loc_dec_input", "value"),
    ],
    prevent_initial_call=True,
)
def loc_construct_info(nclick, sbid, scan, tstart, beam, runname, dm, boxcwidth, totalsample, ra, dec):
    ### first verify the coordinate
    try: ra = float(ra); ra_unit = units.degree
    except: ra_unit = units.hourangle
    
    try: ra_value = Angle(ra, unit=ra_unit).degree; rainvalid = False
    except: rainvalid = True

    try: dec_value = Angle(dec, unit=units.degree).degree; decinvalid = False
    except: decinvalid = True

    if rainvalid or decinvalid:
        return rainvalid, decinvalid, "please provide a correct coordinate...", "", ""

    cand_dict = dict(
        sbid=sbid, beam=beam, scan=scan, tstart=tstart, runname=runname,
        dm=dm, boxcwidth=boxcwidth, totalsample=totalsample, ra = ra_value,
        dec = dec_value
    )

    try:
        cand_dict = construct_candinfo(cand_dict)
    except:
        return rainvalid, decinvalid, "the information you provided is insufficient...", None, cand_dict.__str__()
    
    # print(cand_dict)
    ### print out information like uvfitspath, calibration path, and wd
    casabins = {
        "casa 5.4.1": "/CRACO/SOFTWARE/craco/craftop/softwares/casa-release-5.4.1-31.el7/bin/casa",
        "casa 6.3.0": "/CRACO/SOFTWARE/craco/craftop/softwares/casa-6.3.0-48/bin/casa"
    }

    ### get workdir automatically...
    candfolder = f"TS{int(float(cand_dict['totalsample']))}"
    candfolder += f"W{int(float(cand_dict['boxcwidth']))}"
    candfolder += f"DM{int(float(cand_dict['dm']))}"
    defaultworkdir = "/".join(cand_dict["uvfitspath"].split("/")[:-1])
    defaultworkdir += f"/pipeloc/{candfolder}"
    ###

    fileinfo = [
        dbc.Container([
            dbc.Row([
                dbc.Col(html.B("UVFITS"), width=3, align="center"),
                dbc.Col(cand_dict["uvfitspath"], width=6, align="center"),
            ], style={'marginBottom': '0.5em'}),
            dbc.Row([
                dbc.Col(html.B("CALPATH"), width=3, align="center"),
                dbc.Col(cand_dict["calpath"], width=6, align="center"),
            ], style={'marginBottom': '0.5em'}),
            dbc.Row([
                dbc.Col(html.B("WORKING DIR"), width=3, align="center"),
                dbc.Col(dbc.Input(value=defaultworkdir, id="loc_workdir", ), width=6, align="center"),
            ], style={'marginBottom': '0.5em'}),
            dbc.Row([
                dbc.Col(html.B("CASA EXECUTABLE"), width=3, align="center"),
                dbc.Col(
                    dbc.Select(
                        options=[{"label": casav, "value": casapath} for casav, casapath in casabins.items()],
                        id = "loc_casa_path", value=casabins["casa 5.4.1"],
                    ), width=6, align="center"),
            ]),
        ], style={'marginBottom': '2.0em'}),
    ]
    
    return rainvalid, decinvalid, fileinfo, job_layout(), cand_dict.__str__()

### functions for make uvfits snippet...
def find_field_snippet_range(length, tpulse, uvnsamp, uvtstart=0, ):
    ### for realtime snippet
    tpulse = tpulse - uvtstart
    #######################
    idxs, idxe = tpulse - length // 2, tpulse + length // 2

    if idxs < 0: idxs = 0; idxe = length
    if idxe > uvnsamp: idxe = uvnsamp; idxs = max(0, uvnsamp - length)

    return idxs, idxe

def snippet_command(
        uvpath, calib=None, outname=None, workdir=None,
        idxs=0, idxe=-1, dm=0.0, 
        snippet_cmd = "`which uvfits_snippet.py`",
        extra = ""
    ):
        uvfolder = "/".join(uvpath.split("/")[:-1])
        if workdir is None: workdir = uvfolder
        cmd = f"{snippet_cmd} {uvpath}"
        cmd += f" -tstart {idxs} -tend {idxe}"
        if calib is not None:
            cmd += f" -calib {calib}"
        cmd += f" -dedisp_pccc {int(dm)}"
        if outname is None:
            uvfname = uvpath.split("/")[-1]
            uvfname = uvfname.replace(".uvfits", "")
            outname = uvfname + ".t{}_{}".format(idxs, idxe)
            if calib is not None:
                outname = outname + ".calib"
            outname = outname + ".dm{:.0f}".format(dm)
            outname = outname + ".uvfits"
        cmd += f" -outname {workdir}/{outname}"
        cmd += f" {extra}"
        return cmd, outname

@callback(
    output=[
        Output("loc_job_info", "children"),
        Output("loc_exe_title", "children"),
        Output("local_cmd_strings", "data"),
    ],
    inputs = [
        Input("loc_job_btn", "n_clicks"),
        State("local_cand_query_strings", "data"),
        State("loc_workdir", "value"),
        State("loc_casa_path", "value"),
        State("loc_snippet_length_input", "value"),
        State("loc_snippet_extra_cmd", "value"),
    ],
    prevent_initial_call = True,
)
def job_construct_info(
    nclick, cand_query_str, workdir, casapath, length, extra
):
    print("Here is the number of the click...", nclick)
    if nclick is None: raise PreventUpdate
    cand_query = eval(cand_query_str)
    print(cand_query)

    uvpath = cand_query.get("uvfitspath")
    calpath = cand_query.get("calpath")
    tpulse = int(cand_query.get("totalsample"))
    boxcwidth = int(cand_query.get("boxcwidth"))
    dm = int(float(cand_query.get("dm")))

    uvsource = uvfits.open(uvpath)
    uvnsamp = uvsource.nsamps
    uvtstart = uvsource.header.get("SMPSTRT", default=0)
    uvsource.close()

    tpulse_ = tpulse - uvtstart

    fidxs, fidxe = find_field_snippet_range(length, tpulse_, uvnsamp, )
    bidxs, bidxe = tpulse_ - boxcwidth, tpulse_

    ### get foutname and boutname here...
    uvfname = uvpath.split("/")[-1]
    uvfname = uvfname.replace(".uvfits", "")
    foutname = uvfname + ".t{}_{}".format(fidxs, fidxe) + f".calib.dm{dm}.uvfits"
    boutname = uvfname + ".t{}_{}".format(bidxs, bidxe) + f".calib.dm{dm}.uvfits"


    fcmd, foutname = snippet_command(
        uvpath, calib=calpath, workdir=workdir, outname=foutname,
        idxs=fidxs, idxe=fidxe, dm=dm, extra=extra
    )
    bcmd, boutname = snippet_command(
        f"{workdir}/{foutname}", workdir=workdir, outname=boutname,
        idxs=bidxs-fidxs, idxe=bidxe-fidxs, dm=0, extra=extra
    ) # use the previous one to make a new snippet...

    cand_query["foutname"] = foutname
    cand_query["boutname"] = boutname

    ### form a list of commands...
    cmds = [
        "#!/bin/bash\n",
        "### cd to the correct work directory",
        f"cd {workdir}",
        "\n### activate craco environment",
        "source /home/craftop/.conda/.remove_conda.sh",
        "source /home/craftop/.conda/.activate_conda.sh",
        "conda activate craco", 
        "\n### fix uvfits", f"`which fixuvfits` {uvpath}",
        "\n### make uvfits snippet", fcmd, bcmd,
        "\n### make casa images",
        f"{casapath} --log2term --nologger -c `which casa_quick_clean.py` {workdir}/{foutname} {workdir} f t",
        f"{casapath} --log2term --nologger -c `which casa_quick_clean.py` {workdir}/{boutname} {workdir} f t",
        "\n### copy fits file to the work directory...",
        f"cp {workdir}/clean/*t{fidxs}_{fidxe}*/*.image.fits {workdir}/field.image.fits",
        f"cp {workdir}/clean/*t{bidxs}_{bidxe}*/*.image.fits {workdir}/burst.image.fits",
        "\n### run source finder",
        f"`which source_find.py` -fits {workdir}/field.image.fits",
        f"`which source_find.py` -fits {workdir}/burst.image.fits",
        "\n###TODO: ADD CROSSMATCHING STEPS",
        f"`which localise_source.py` --fieldfpath {workdir}/field.image.fits --burstfpath {workdir}/burst.image.fits  --ra {cand_query['ra']} --dec {cand_query['dec']} --workdir {workdir}",
        f"`which slack_post_localisation.py` -workdir {workdir} -channel {SLACK_CHANNEL}"
    ]
    
    return job_outputs(cmds), exe_layout(), cand_query.__str__()


@callback(
    output=[
        Output("loc_exe_info", "children"),
    ],
    inputs = [
        Input("loc_exe_btn", "n_clicks"),
        State("local_bash_scripts", "value"),
        State("loc_workdir", "value"),
        State("local_cand_query_strings", "data")
    ],
    prevent_initial_call = True,
)
def exe_construct_info(nclick, cmds, workdir, cand_query_string):
    if nclick is None: raise PreventUpdate
    ### write commands to a file...
    logs = []
    if not os.path.exists(workdir):
        os.makedirs(workdir)
        logs.append(f"making directory... - {workdir}")
    cmdpath = f"{workdir}/localise.sh"
    if os.path.exists(cmdpath):
        logs.append(f"command file, {cmdpath}, already exists... aborting...")
        return exe_outputs(logs),
    
    with open(cmdpath, "w") as fp:
        fp.write(cmds)
        logs = [f"writing command to file - {cmdpath}"]
    os.system(f"chmod +x {cmdpath}")
    logs.append("add executing permission to the file...")
    ### post message to slack...
    slackmsg = f'*[LOCALISER]* Submit localisation Job in the Queue\n'
    slackmsg += "\n".join(logs)
    ### tsp the job...
    ecopy = os.environ.copy()
    ecopy.update(dict(TS_SOCKET="/data/craco/craco/tmpdir/queues/local"))
    p = subprocess.run([f"tsp {cmdpath}"], shell=True, capture_output=True, text=True, env=ecopy)
    slackmsg += f"\n_tsp output_ - {p.stdout.strip()}"
    slackpost = SlackPostManager(test=False, channel=SLACK_CHANNEL)
    slackpost.post_message(slackmsg)
    return exe_outputs(logs),

### layout for the input cells...
def table_inputs(cand_query):
    sbid = cand_query.get("sbid")
    scan = cand_query.get("scan")
    tstart = cand_query.get("tstart")
    beam = cand_query.get("beam")
    runname = cand_query.get("runname")
    dm = cand_query.get("dm")
    boxcwidth = cand_query.get("boxcwidth")
    totalsample = cand_query.get("totalsample")
    ra = cand_query.get("ra")
    dec = cand_query.get("dec")
    return dbc.Container([
        dbc.Row(
            html.H5("Candidate Information"), align="center",
            style={'marginBottom': '1.0em'}
        ),
        dbc.Row([
            dbc.Col(html.B("SBID"), width=1, align="center"),
            dbc.Col(dbc.Input(id="loc_sbid_input", value=sbid), width=2),
            dbc.Col(html.B("SCAN"), width=1, align="center"),
            dbc.Col(dbc.Input(id="loc_scan_input", value=scan), width=2),
            dbc.Col(html.B("TSTART"), width=1, align="center"),
            dbc.Col(dbc.Input(id="loc_tstart_input", value=tstart), width=2),
            dbc.Col(html.B("BEAM"), width=1, align="center"),
            dbc.Col(dbc.Input(id="loc_beam_input", value=beam), width=2),
        ], style={'marginBottom': '0.5em'}),
        dbc.Row([
            dbc.Col(html.B("RUNNAME"), width=1, align="center"),
            dbc.Col(dbc.Input(id="loc_runname_input", value=runname), width=2),
            dbc.Col(html.B("DM"), width=1, align="center"),
            dbc.Col(dbc.Input(id="loc_dm_input", value=dm), width=2),
            dbc.Col(html.B("BOXCWIDTH"), width=1, align="center"),
            dbc.Col(dbc.Input(id="loc_boxcwidth_input", value=boxcwidth), width=2),
            dbc.Col(html.B("TSAMPLE"), width=1, align="center"),
            dbc.Col(dbc.Input(id="loc_totalsample_input", value=totalsample), width=2),
        ], style={'marginBottom': '0.5em'}),
        dbc.Row([
            dbc.Col(html.B("RA"), width=2, align="center"),
            dbc.Col(dbc.Input(id="loc_ra_input", value=ra), width=4),
            dbc.Col(html.B("DEC"), width=2, align="center"),
            dbc.Col(dbc.Input(id="loc_dec_input", value=dec), width=4),
        ], style={'marginBottom': '3.0em'}),
    ])

def file_outputs():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H5("File Information"), width=2, align="center",),
            dbc.Col(dbc.Button("Verify Candidate Information", color="success", id="loc_verify_btn"), width=3, align="center", ),
            dbc.Col("Field Image Length (samples)", width=3, align="center"),
            dbc.Col(dbc.Input(id="loc_snippet_length_input", value=3000, type='number'), width=2),
        ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'}),
    ])

def job_layout():
    return [
        dbc.Row([
            dbc.Col(html.H5("Job Information"), width=2, align="center",),
            dbc.Col(dbc.Button("Verify File Information", color="success", id="loc_job_btn"), width=3, align="center", ),
            dbc.Col("extra command", width=2, align="center"),
            dbc.Col(dbc.Input(placeholder="extra command for uvfits_extract.py", id="loc_snippet_extra_cmd", value=""), width=4),
        ]),
        dbc.Container(id="loc_job_info", style={'marginBottom': '1.0em', 'marginTop': '1.0em'})
    ]

def job_outputs(cmds):
    cmdstyle = {"font-family": "Dejavu", "fontsize": 10}
    rows = [
        dbc.Row("The following commands will be executed...", style={'marginBottom': '0.5em'}),
        dbc.Row(dbc.Textarea(value="\n".join(cmds), style=cmdstyle, id="local_bash_scripts"), style={"height": "30vh", 'marginBottom': '2.0em'})
    ]
    
    return dbc.Container(rows)

def exe_layout():
    return [
        dbc.Row([
            dbc.Col(html.H5("Execution Information"), width=2, align="center",),
            dbc.Col(dbc.Button("Confirm and Submit", color="success", id="loc_exe_btn"), width=3, align="center", ),
            # dbc.Col("extra command", width=2, align="center"),
            # dbc.Col(dbc.Input(placeholder="extra command for uvfits_extract.py", id="loc_snippet_extra_cmd", value=""), width=4),
        ]),
        dbc.Container(id="loc_exe_info", style={'marginBottom': '1.0em', 'marginTop': '1.0em'})
    ]

def exe_outputs(logs):
    logstyle = {"font-family": "Dejavu", "fontsize": 10}
    rows = [
        dbc.Row(log, style=logstyle) for log in logs
    ]
    return dbc.Container(rows)

### here for the main layout
def layout(**cand_query):
    # print("in the layout...", cand_query)
    return [
        dcc.Store(id="local_cand_query_strings",),
        dcc.Store(id="local_cmd_strings",),
        dbc.Container(
            html.H2("CRACO Automatic Localisation Tool (Beta)"), 
            style={'marginBottom': '1.0em', 'marginTop': '1.0em'}
        ),
        table_inputs(cand_query),
        file_outputs(),
        dbc.Container(id="loc_file_info", style={'marginBottom': '1.0em', 'marginTop': '1.0em'}),
        dbc.Container(id="loc_job_title", style={'marginBottom': '1.0em', 'marginTop': '1.0em'}),
        dbc.Container(id="loc_exe_title", style={'marginBottom': '1.0em', 'marginTop': '1.0em'})

    ]


# test url - http://localhost:8025/localise?sbid=63411&beam=31&scan=00&tstart=20240630170839&runname=results&dm=4.56&boxcwidth=1&lpix=53&mpix=131&totalsample=27973&ra=241.73338281204377&dec=-8.902020511454985