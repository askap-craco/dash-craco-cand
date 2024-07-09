import os
import subprocess

from dash import Dash, html, callback, dcc, Input, Output,State, ctx, dash_table
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash
import plotly.express as px

from apputil import construct_candinfo, get_source_name

from astropy.coordinates import SkyCoord, Angle
from astropy import units

from craft import uvfits
from craco.craco_run.slackpost import SlackPostManager
from craco.datadirs import SchedDir, ScanDir

dash.register_page(__name__, path="/tab", title="Tied-Array Beam Filterbank")

SLACK_CHANNEL = "C07BEEDLX8W"

def get_allsbid_fitsfile(sbid, beam):
    scheddir = SchedDir(sbid=sbid)
    alluvfits = []
    for scan in scheddir.scans:
        scandir = ScanDir(sbid=sbid, scan=scan)
        alluvfits.append(scandir.beam_uvfits_path(beam))
    return alluvfits
        

@callback(
    output=[
        Output("tab_ra_input", "invalid"),
        Output("tab_dec_input", "invalid"),
        Output("tab_file_info", "children"),
        Output("tab_job_title", "children"),
        Output("tab_source_name", "value"),
        Output("tab_cand_query_strings", "data"),
    ],
    inputs=[
        Input("tab_verify_btn", "n_clicks"),
        State("tab_sbid_input", "value"),
        State("tab_scan_input", "value"),
        State("tab_tstart_input", "value"),
        State("tab_beam_input", "value"),
        State("tab_ra_input", "value"),
        State("tab_dec_input", "value"),
        State("tab_norm_switch", "value"),
        State("tab_calib_switch", "value"),
        State("tab_skip_sample", "value"),
        State("tab_process_sample", "value"),
        State("tab_nt", "value"),
        State("tab_allscan_switch", "value"),
        State("tab_source_name", "value"),
    ],
    prevent_initial_call=True,
)
def tab_construct_info(
    nclick, sbid, scan, tstart, beam, ra, dec, 
    norm, calib, skip, process, nt, allscan, srcname
):
    ### first verify the coordinate
    try: ra = float(ra); ra_unit = units.degree
    except: ra_unit = units.hourangle
    
    try: ra_value = Angle(ra, unit=ra_unit).degree; rainvalid = False
    except: rainvalid = True

    try: dec_value = Angle(dec, unit=units.degree).degree; decinvalid = False
    except: decinvalid = True

    if rainvalid or decinvalid:
        return rainvalid, decinvalid, "please provide a correct coordinate...", None, srcname, cand_dict.__str__()

    cand_dict = dict(
        sbid=sbid, beam=beam, scan=scan, tstart=tstart, runname="results",
        ra = ra_value, dec = dec_value
    )

    try:
        cand_dict = construct_candinfo(cand_dict)
    except:
        return rainvalid, decinvalid, "the information you provided is insufficient...", None, srcname, cand_dict.__str__()
        
    # print(cand_dict)
    ### print out information like uvfitspath, calibration path, and wd
    ### get workdir automatically...
    if srcname is None: srcname = get_source_name(ra_value, dec_value)

    defaultworkdir = "/".join(cand_dict["uvfitspath"].split("/")[:-1])
    defaultworkdir += f"/pipetab/{srcname}"
    ###

    ### if allscan... then uvfitspath will be a list...
    try:
        if allscan:
            alluvfits = get_allsbid_fitsfile(sbid, beam)
            cand_dict["uvfitspath"] = alluvfits
        else:
            cand_dict["uvfitspath"] = [cand_dict["uvfitspath"]]
    except:
        return rainvalid, decinvalid, "cannot get all uvfits file for this sbid...", None, cand_dict.__str__()


    fileinfo = [
        dbc.Container([
            dbc.Row([
                dbc.Col(html.B("CALPATH"), width=3, align="center"),
                dbc.Col(cand_dict["calpath"], width=6, align="center"),
            ], style={'marginBottom': '0.5em'}),
            dbc.Row([
                dbc.Col(html.B("UVFITS"), width=3, align="center"),
                dbc.Col(
                    dbc.Textarea(value="\n".join(cand_dict["uvfitspath"]), readonly=True, rows=1)
                    , width=6, align="center"
                ),
            ], style={'marginBottom': '0.5em'}),
            dbc.Row([
                dbc.Col(html.B("WORKING DIR"), width=3, align="center"),
                dbc.Col(dbc.Input(value=defaultworkdir, id="tab_workdir", ), width=6, align="center"),
            ], style={'marginBottom': '0.5em'}),
        ], style={'marginBottom': '2.0em'}),
    ]
    
    pdict = dict(norm=norm, calib=calib, skip=skip, process=process, nt=nt, srcname=srcname)
    cand_dict.update(pdict)
    print(cand_dict)
    return rainvalid, decinvalid, fileinfo, tab_job_layout(), srcname, cand_dict.__str__()

def format_tab_filterbank_cmds(cand_query, extra):
    uvpaths = cand_query.get("uvfitspath")
    calpath = cand_query.get("calpath")
    ra = cand_query.get("ra")
    dec = cand_query.get("dec")
    norm = cand_query.get("norm")
    calib = cand_query.get("calib")
    nt = cand_query.get("nt")
    skip = cand_query.get("skip")
    process = cand_query.get("process")
    srcname = cand_query.get("srcname") 

    cmds = []
    for uvpath in uvpaths:
        filname = default_tabfil_name(uvpath, srcname, norm, calib, skip, process, nt)
        cmd = f"`which tab_filterbank` -uv {uvpath}"
        cmd += f""" -t "{ra}d {dec}d" """
        if calib: cmd += f" -c {calpath}"
        cmd += f" -nt {nt}"
        if norm: cmd += " -norm"
        if skip != 0: cmd += f" --seek_samps {skip}"
        if process != -1: cmd += f" --process_samps {process}"
        if extra is not None: cmd += f" {extra}"
        cmd += f" {filname}"
        cmds.append(f"`which fix_uvfits` {uvpath}")
        cmds.append(cmd)
    return cmds

def default_tabfil_name(uvpath, srcname, norm, cal, skip, process, nt, ):
    # get tstart from uvpath
    tstart = uvpath.split("/")[-2][-6:]
    name_lst = [srcname, tstart,]
    if norm is True:
        name_lst.append(f"nt{nt}")
    if skip != 0:
        name_lst.append(f"s{skip}")
    if process != -1:
        name_lst.append(f"p{process}")
    if cal:
        name_lst.append("cal")
    return ".".join(name_lst) + ".fil"

@callback(
    output=[
        Output("tab_job_info", "children"),
        Output("tab_exe_title", "children"),
    ],
    inputs = [
        Input("tab_job_btn", "n_clicks"),
        State("tab_cand_query_strings", "data"),
        State("tab_workdir", "value"),
        State("tab_extra_cmd", "value"),
    ],
    prevent_initial_call = True,
)
def tab_job_construct_info(
    nclick, cand_query_str, workdir, extra
):
    print("Here is the number of the click...", nclick)
    if nclick is None: raise PreventUpdate
    cand_query = eval(cand_query_str)
    print(cand_query)

    tabcmds = format_tab_filterbank_cmds(cand_query, extra)

    ### form a list of commands...
    cmds = [
        "#!/bin/bash\n",
        "### cd to the correct work directory",
        f"cd {workdir}",
        "\n### activate craco environment",
        "source /home/craftop/.conda/.remove_conda.sh",
        "source /home/craftop/.conda/.activate_conda.sh",
        "conda activate craco", 
        "\n### fix uvfits and run tab filterbank", 
        *tabcmds,
        "\n### send update to slack...",
        f"`which slack_post_tab.py` -workdir {workdir} -channel {SLACK_CHANNEL}"
    ]
    
    return job_outputs(cmds), tab_exe_layout(), 

def exe_outputs(logs):
    logstyle = {"font-family": "Dejavu", "fontsize": 10}
    rows = [
        dbc.Row(log, style=logstyle) for log in logs
    ]
    return dbc.Container(rows)

@callback(
    output=[
        Output("tab_exe_info", "children"),
    ],
    inputs = [
        Input("tab_exe_btn", "n_clicks"),
        State("tab_bash_scripts", "value"),
        State("tab_workdir", "value"),
        State("tab_cand_query_strings", "data")
    ],
    prevent_initial_call = True,
)
def tab_exe_construct_info(nclick, cmds, workdir, cand_query):
    if nclick is None: raise PreventUpdate
    ### write commands to a file...
    logs = []
    if not os.path.exists(workdir):
        os.makedirs(workdir)
        logs.append(f"making directory... - {workdir}")
    cmdpath = f"{workdir}/tab_shell.sh"
    if os.path.exists(cmdpath):
        logs.append(f"command file, {cmdpath}, already exists... aborting...")
        return exe_outputs(logs),
    
    with open(cmdpath, "w") as fp:
        fp.write(cmds)
        logs = [f"writing command to file - {cmdpath}"]
    os.system(f"chmod +x {cmdpath}")
    logs.append("add executing permission to the file...")
    ### post message to slack...
    slackmsg = f'*[TAB]* Submit Tied-Array Beam Filterbank jobs in the Queue\n'
    slackmsg += "\n".join(logs)
    ### tsp the job...
    ecopy = os.environ.copy()
    ecopy.update(dict(TS_SOCKET="/data/craco/craco/tmpdir/queues/tab"))
    p = subprocess.run([f"tsp {cmdpath}"], shell=True, capture_output=True, text=True, env=ecopy)
    slackmsg += f"\n_tsp output_ - {p.stdout.strip()}"
    slackpost = SlackPostManager(test=False, channel=SLACK_CHANNEL)
    slackpost.post_message(slackmsg)
    return exe_outputs(logs),

### layouts
def tab_table_inputs(cand_query):
    sbid = cand_query.get("sbid")
    scan = cand_query.get("scan")
    tstart = cand_query.get("tstart")
    beam = cand_query.get("beam")
    ra = cand_query.get("ra")
    dec = cand_query.get("dec")
    return dbc.Container([
        dbc.Row(
            html.H5("Candidate Information"), align="center",
            style={'marginBottom': '1.0em'}
        ),
        dbc.Row([
            dbc.Col(html.B("SBID"), width=1, align="center"),
            dbc.Col(dbc.Input(id="tab_sbid_input", value=sbid), width=2),
            dbc.Col(html.B("SCAN"), width=1, align="center"),
            dbc.Col(dbc.Input(id="tab_scan_input", value=scan), width=2),
            dbc.Col(html.B("TSTART"), width=1, align="center"),
            dbc.Col(dbc.Input(id="tab_tstart_input", value=tstart), width=2),
            dbc.Col(html.B("BEAM"), width=1, align="center"),
            dbc.Col(dbc.Input(id="tab_beam_input", value=beam), width=2),
        ], style={'marginBottom': '0.5em'}),
        dbc.Row([
            dbc.Col(html.B("RA"), width=2, align="center"),
            dbc.Col(dbc.Input(id="tab_ra_input", value=ra), width=4),
            dbc.Col(html.B("DEC"), width=2, align="center"),
            dbc.Col(dbc.Input(id="tab_dec_input", value=dec), width=4),
        ], style={'marginBottom': '0.5em'}),
        ### skip parameters and normalisation
        dbc.Row([
            dbc.Col(html.B("SKIP"), width=1, align="center"),
            dbc.Col(dbc.Input(id="tab_skip_sample", value=0, type='number'), width=1),
            dbc.Col(html.B("PROCESS"), width=1, align="center"),
            dbc.Col(dbc.Input(id="tab_process_sample", value=-1, type='number'), width=1),
            dbc.Col(html.B("NT"), width=1, align="center"),
            dbc.Col(dbc.Input(id="tab_nt", value=256, type="number"), width=1),
            dbc.Col(html.B("NORM"), width=1, align="center"),
            dbc.Col(dbc.Switch(id="tab_norm_switch", label=None, value=True), width=1, align="center"),
            dbc.Col(html.B("CAL"), width=1, align="center"),
            dbc.Col(dbc.Switch(id="tab_calib_switch", label=None, value=True), width=1, align="center"),
            dbc.Col(html.B("ALLSCAN"), width=1, align="center"),
            dbc.Col(dbc.Switch(id="tab_allscan_switch", label=None, value=True), width=1, align="center"),
        ], style={'marginBottom': '2.0em'}),
    ])

def tab_file_outputs():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H5("File Information"), width=2, align="center",),
            dbc.Col(dbc.Button("Verify Candidate Information", color="success", id="tab_verify_btn"), width=3, align="center", ),
            dbc.Col("Source Name", width=2, align="center"),
            dbc.Col(dbc.Input(id="tab_source_name", placeholder="alternative source name"), width=3, align="center"),
        ], style={'marginBottom': '1.0em', 'marginTop': '1.0em'}),
    ])

def tab_job_layout():
    return [
        dbc.Row([
            dbc.Col(html.H5("Job Information"), width=2, align="center",),
            dbc.Col(dbc.Button("Verify File Information", color="success", id="tab_job_btn"), width=3, align="center", ),
            dbc.Col("extra command", width=2, align="center"),
            dbc.Col(dbc.Input(placeholder="extra command for tab_filterbank, e.g., --flag-ant", id="tab_extra_cmd", value=""), width=4),
        ]),
        dbc.Container(id="tab_job_info", style={'marginBottom': '1.0em', 'marginTop': '1.0em'})
    ]

def tab_exe_layout():
    return [
        dbc.Row([
            dbc.Col(html.H5("Execution Information"), width=2, align="center",),
            dbc.Col(dbc.Button("Confirm and Submit", color="success", id="tab_exe_btn"), width=3, align="center", ),
            # dbc.Col("extra command", width=2, align="center"),
            # dbc.Col(dbc.Input(placeholder="extra command for uvfits_extract.py", id="loc_snippet_extra_cmd", value=""), width=4),
        ]),
        dbc.Container(id="tab_exe_info", style={'marginBottom': '1.0em', 'marginTop': '1.0em'})
    ]

def job_outputs(cmds):
    cmdstyle = {"font-family": "Dejavu", "fontsize": 10}
    rows = [
        dbc.Row("The following commands will be executed...", style={'marginBottom': '0.5em'}),
        dbc.Row(dbc.Textarea(value="\n".join(cmds), style=cmdstyle, id="tab_bash_scripts"), style={"height": "30vh", 'marginBottom': '2.0em'})
    ]
    
    return dbc.Container(rows)


def layout(**cand_query):
    return [
        dcc.Store(id="tab_cand_query_strings",),
        dbc.Container(
            html.H2("CRACO TAB Filterbank Creation Tool (Beta)"), 
            style={'marginBottom': '1.0em', 'marginTop': '1.0em'}
        ),
        tab_table_inputs(cand_query),
        tab_file_outputs(),
        dbc.Container(id="tab_file_info", style={'marginBottom': '1.0em', 'marginTop': '1.0em'}),
        dbc.Container(id="tab_job_title", style={'marginBottom': '1.0em', 'marginTop': '1.0em'}),
        dbc.Container(id="tab_exe_title", style={'marginBottom': '1.0em', 'marginTop': '1.0em'})
    ]


# example of multiple file
# http://localhost:8025/tab?sbid=63380&beam=17&scan=00&tstart=20240629063633&runname=results&dm=534.3820190429688&boxcwidth=1&lpix=153&mpix=165&totalsample=16830&ra=83.54741668701172&dec=-59.67527770996094

# example of one file
# 