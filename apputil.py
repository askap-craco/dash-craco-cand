from dash import html
import dash_bootstrap_components as dbc

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import os
import json
import sqlite3
from io import BytesIO
import base64
import glob

from craft import sigproc

# load password pairs
def load_password(fname):
    with open(fname) as fp:
        pwd = json.load(fp)
    return pwd

# web layout
def _header():
    return dbc.NavbarSimple(
        children = [
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.NavItem(dbc.NavLink("Beamloader", href="/beamfile")),
            # dbc.NavItem(dbc.NavLink("SchedBlock", href="/schedblock")),
            # dbc.NavItem(dbc.NavLink("Beam", href="/beam"))
        ],
        brand="CRACO Candidate Inspection",
        brand_href="/", color="primary", dark=True,
    )

def _footer():
    return dbc.Container(
        [dbc.Row(html.P("ASKAP-CRACO @ 2023"))]
    )

# database related
def _check_table(cursor, table_name):
    """
    check if a table exists in a given database
    """
    check_query = """
    SELECT name FROM sqlite_master WHERE type='table' AND name=?
    """
    cursor.execute(check_query, (table_name, ))
    result = cursor.fetchone()

    return True if result else False

def create_table(cursor, ):
    """
    create tables for craco candidate

    we need to create two tables, scan table, beam table and candidate table
    """
    ### table for scan
    if not _check_table(cursor, "SCAN"):
        scan_query = """
CREATE TABLE SCAN (
    SBID INTEGER NOT NULL,
    SCAN CHAR(2) DEFAULT '00',
    TSTART CHAR(14) NOT NULL
)
"""
        cursor.execute(scan_query)

    ### table for beam
    if not _check_table(cursor, "BEAM"):
        beam_query = """
CREATE TABLE BEAM (
    SCANROWID INTEGER NOT NULL,
    BEAM INTEGER NOT NULL,
    NCAND INTEGER NOT NULL
)
"""
        cursor.execute(beam_query)

    if not _check_table(cursor, "CAND"):
        cand_query = """
CREATE TABLE CAND (
    BEAMROWID INTEGER NOT NULL,
    SNR REAL NOT NULL,
    LPIX INTEGER NOT NULL,
    MPIX INTEGER NOT NULL,
    BOXCWIDTH INTEGER NOT NULL,
    TIME INTEGER NOT NULL,
    DM INTEGER NOT NULL,
    IBLK INTEGER NOT NULL,
    RAWSN INTEGER NOT NULL,
    TOTALSAMPLE INTEGER NOT NULL,
    OBSTIMESEC REAL NOT NULL,
    MJD FLOAT NOT NULL,
    DMPCCM REAL NOT NULL,
    RA FLOAT NOT NULL,
    DEC FLOAT NOT NULL,
    CLASS CHAR(30)
)
"""
        cursor.execute(cand_query)

def init_sqlite():
    """
    initiate a database with tabsql command
    """
    conn = sqlite3.connect("db/craco_cand.sqlite3")
    cursor = conn.cursor()
    create_table(cursor, )
    conn.commit()
    conn.close()

### load candidates handler
def load_candidate(fname):
    dtype = np.dtype(
        [
            ('SNR',np.float32), # physical snr
            ('lpix', np.float32), # l pixel of the detection
            ('mpix', np.float32), # m pixel of the detection
            ('boxcwidth', np.float32), # boxcar width, 0 for 1 to 7 for 8
            ('time', np.float32), # sample of the detection in a given block
            ('dm', np.float32), # dispersion measure in hardware
            ('iblk', np.float32), # index of the block of the detection, by default each block is 256 samples
            ('rawsn', np.float32), # raw snr
            ('totalsample', np.float32), # sample index over the whole observation
            ('obstimesec', np.float32), # time difference between the detection and the start of the observation
            ('mjd', np.float64), # detection time in mjd
            ('dmpccm', np.float32), # physical dispersion measure
            ('ra', np.float64), 
            ('dec', np.float64)
        ]
    )

    cand_np = np.loadtxt(fname, dtype=dtype)
    cand_df = pd.DataFrame(cand_np)

    ### change lpix, mpix, boxcwidth, time, dm, iblk, rawsn, total_sample to integer
    cand_df[
        ["lpix", "mpix", "boxcwidth", "time", "dm", "iblk", "rawsn", "totalsample"]
    ] = cand_df[
        ["lpix", "mpix", "boxcwidth", "time", "dm", "iblk", "rawsn", "totalsample"]
    ].astype(int)

    return cand_df

def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)

### load filterbank
def load_filterbank(filpath, tstart, ntimes):
    if tstart < 0: tstart = 0
    
    # load files
    f = sigproc.SigprocFile(filpath)
    nelements = ntimes*f.nifs*f.nchans
    f.seek_data(f.bytes_per_element*tstart)

    if (f.nbits == 8): dtype = np.uint8
    elif (f.nbits == 32): dtype = np.float32

    v = np.fromfile(f.fin, dtype=dtype, count=nelements )
    v = v.reshape(-1, f.nifs, f.nchans)

    tend = tstart + v.shape[0]

    ### give a list of time
    taxis = np.linspace(tstart, tend, v.shape[0], endpoint=False)
    faxis = np.arange(f.nchans) * f.foff + f.fch1

    ### change 0 value to nan
    v[v == 0.] = np.nan

    return v, taxis, faxis

### load pulsar catalogue, RACS catalogue etc.
def load_external_cat():
    pass

def select_region(df, ra, dec, racol, deccol, radius):
    ra_r = min(radius / np.cos(np.deg2rad(dec)), 2)
    rabool = ((df[racol] - ra) % 360 <= ra_r) | ((ra - df[racol]) % 360 <= ra_r)
    decbool = (np.abs(df[deccol] - dec) <= radius)
    return df[rabool & decbool]

def circular_mean(angles):
    angles = np.array(angles)
    cossum = np.cos(np.deg2rad(angles)).sum()
    sinsum = np.sin(np.deg2rad(angles)).sum()
    return np.rad2deg(np.arctan2(sinsum, cossum)) % 360

### function to find corresponding files
def find_file(
        filetype, query_dict
    ):
    """
    find a certain type of the file based on the information provided

    Parameters
    ----------
    filetype: str, allowed values are
        cand_raw: unclustered candidate file
        cand_cls: clustered candiddate file
        cal: calibration file (.npy file)
        uvfits: uvfits file
        ics: incoherent sum
        cas: coherent sum
    sbid: string 
        must provide to get a result
    beam: string
        must provide to get a result
    scan,tstart,runname or scanpath:
        if scanpath provided, it will ignore all remaining three parameters
        if none is provided, we will search for results scan
    """
    assert filetype in ["cand_raw", "cand_cls", "cal", "uvfits", "ics", "cas"], "not an avialable file type..."

    default_dict = dict(
        sbid=None, beam=None, scan=None, tstart=None,
        runname=None, scanpath=None
    )
    query_dict = _update_default_dict(query_dict, default_dict)
    sbid = query_dict["sbid"]; beam = query_dict["beam"]
    scan = query_dict["scan"]; tstart = query_dict["tstart"]
    runname = query_dict["runname"]; scanpath = query_dict["scanpath"]

    if sbid is None or beam is None:
        return None

    ### reformat sbid
    if isinstance(sbid, int):
        sbid = str(sbid)
    if sbid.startswith("SB"):
        sbid = sbid[2:]
    sbid = "SB{:0>6}".format(sbid)

    ### reformat beam
    beam = "{:0>2}".format(beam)

    ### if you are looking for the calibration file, you don't need to worry about something else
    if filetype == "cal":
        calfile = f"/data/seren-01/big/craco/{sbid}/cal/{beam}/b{beam}.aver.4pol.smooth.npy"
        if _check_file(calfile): return calfile
        return None

    if scanpath is None:
        scanpath = _find_scanpath(sbid, scan, tstart, runname)
        if scanpath is None: return None
    else:
        if not _check_scanpath(sbid, scanpath): return None
    
    # use scanpath from now
    scandir = f"/data/seren-??/big/craco/{sbid}/scans/{scanpath}"
    ### for uvfits file
    if filetype == "uvfits":
        uvfitsfiles = glob.glob(f"{scandir}/b{beam}.uvfits")
        if len(uvfitsfiles) == 0: return None
        return uvfitsfiles[0]

    if filetype == "ics":
        icsfiles = glob.glob(f"{scandir}/ics_b{beam}.fil")
        if len(icsfiles) == 0: return None
        return icsfiles[0]

    if filetype == "cas":
        casfiles = glob.glob(f"{scandir}/cas_b{beam}.fil")
        if len(casfiles) == 0: return None
        return casfiles[0]

    ### for clustered candidates
    if filetype == "cand_raw":
        # there are two different format...
        candfiles = glob.glob(f"{scandir}/candidates.b{beam}.txt")
        if len(candfiles) == 0:
            candfiles = glob.glob(f"{scandir}/candidates.txtb{beam}")
            if len(candfiles) == 0: return None
        return candfiles[0] # assume this is the only file...

    ### for clustered candidates
    if filetype == "cand_cls":
        candfiles = glob.glob(f"{scandir}/clustering_output/candidates.b{beam}.txt.uniq")
        if len(candfiles) == 0:
            candfiles = glob.glob(f"{scandir}/clustering_output/candidates.txtb{beam}.uniq")
            if len(candfiles) == 0: return None
        return candfiles[0] # assume this is the only file...
    

def _find_scanpath(sbid, scan=None, tstart=None, runname=None):
    """
    find scanpath based on sbid
    """
    if scan is None: scan = "*"
    if tstart is None: tstart = "*"
    if runname is None: runname = "results"
    
    beamfiles = glob.glob(f"/data/seren-??/big/craco/{sbid}/scans/{scan}/{tstart}/{runname}/")
    if len(beamfiles) == 0: 
        beamfiles = glob.glob(f"/data/seren-??/big/craco/{sbid}/scans/{scan}/{tstart}/*/")
        if len(beamfiles) == 0: return None
    return "/".join(beamfiles[0].split("/")[7:10])

def _check_scanpath(sbid, scanpath):
    scans = glob.glob(f"/data/seren-??/big/craco/{sbid}/scans/{scanpath}/")
    if len(scans) == 0: return False
    return True

def _check_file(filepath):
    if os.path.exists(filepath): return True
    return False

### update candidate dictionary related
def _update_default_dict(rawdict, defdict):
    for key, value in defdict.items():
        if key not in rawdict:
            rawdict[key] = value
    return rawdict

def _extract_beam_candfile(candfile):
    """
    extract beam information based on the candidate file name
    """
    candfile = candfile.split("/")[-1]
    candfile = candfile.replace(".uniq", "")
    try:
        if "txtb" in candfile: return int(candfile[-2:])
        return int(candfile[-6:-4])
    except:
        return None

def construct_candinfo(query_dict):
    default_dict = dict(
        sbid=None, beam=None, scan=None, tstart=None,
        runname=None, dm=None, lpix=None, mpix=None, 
        boxcwidth=None, totalsample=None, ra=None, dec=None,
    )
    query_dict = _update_default_dict(query_dict, default_dict)

    query_dict["unclustpath"] = find_file("cand_raw", query_dict)
    query_dict["clustpath"] = find_file("cand_cls", query_dict)
    query_dict["calpath"] = find_file("cal", query_dict)
    query_dict["uvfitspath"] = find_file("uvfits", query_dict)
    query_dict["icspath"] = find_file("ics", query_dict)
    query_dict["caspath"] = find_file("cas", query_dict)

    return query_dict

def construct_beaminfo(query_dict):
    default_dict = dict(
        sbid=None, beam=None, scan=None, tstart=None,
        runname=None, fname=None, unique=False, scanpath=None
    )
    query_dict = _update_default_dict(query_dict, default_dict)
    
    if query_dict["fname"] is None:
        query_dict["fname"] = find_file("cand_cls", query_dict) if query_dict["unique"] else find_file("cand_raw", query_dict)

    fname = query_dict["fname"]
    fnamesplit = fname.split("/")
    ### it will update all other information based on the fname
    newdict = dict(
        unique = True if fname.endswith("uniq") else False,
        beam = _extract_beam_candfile(fname),
        sbid = int(fnamesplit[5][2:]),
        scan = fnamesplit[7],
        tstart = fnamesplit[8],
        runname = fnamesplit[9],
        scanpath = f"{fnamesplit[7]}/{fnamesplit[8]}/{fnamesplit[9]}"
    )
    query_dict.update(newdict)

    return query_dict