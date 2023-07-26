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

    return v, taxis, faxis

