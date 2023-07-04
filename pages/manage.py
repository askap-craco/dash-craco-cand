from dash import Dash, html, callback
import dash_bootstrap_components as dbc
import dash

import sqlite3
import glob

import pandas as pd 
import numpy as np

dash.register_page(__name__, path="/manage", title="CRACO Manage")

##### callback related functions

### load and update candidate files...
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

def _extract_fileinfo(candfpath):
    """
    get basic information from the candidate file path

    Returns:
    ----------
    candinfo: dict
    """
    candfpath_split = candfpath.split("/")
    beamid = candfpath_split[-1][-7:-5]
    return {
        "SBID": int(candfpath_split[6]),
        "SCAN": candfpath_split[7],
        "TSTART": candfpath_split[8],
        "BEAM": int(beamid),
    }

def _update_scan(cursor, candinfo):
    ### first check if the data is already there...
    cursor.execute(
        """
        SELECT rowid, * FROM SCAN
        WHERE SBID={} AND SCAN='{}' AND TSTART='{}'
        """.format(candinfo["SBID"], candinfo["SCAN"], candinfo["TSTART"])
    )
    scanresult = cursor.fetchone()

    if scanresult: return scanresult[0], True

    # this means we need to update the database
    scanupdate = """ 
    INSERT INTO SCAN (SBID, SCAN, TSTART)
    VALUES (?, ?, ?)
    """
    cursor.execute(
        scanupdate, (candinfo["SBID"], candinfo["SCAN"], candinfo["TSTART"])
    )

    return cursor.lastrowid, False

def _update_beam(cursor, candinfo, scanrowid, ncand):
    cursor.execute(
        """
        SELECT rowid, * FROM BEAM
        WHERE SCANROWID={} AND BEAM={}
        """.format(scanrowid, candinfo["BEAM"])
    )
    beamresult = cursor.fetchone()

    if beamresult:
        if ncand == beamresult[-1]: return beamresult[0], True
        cursor.execute(
            """
            UPDATE BEAM SET NCAND={}
            WHERE ROWID={}
            """.format(ncand, beamresult[0])
        )

        return beamresult[0], False
    
    # otherwise update new one
    beamupdate = """
    INSERT INTO BEAM (SCANROWID, BEAM, NCAND)
    VALUES (?, ?, ?)
    """
    cursor.execute(
        beamupdate, (scanrowid, candinfo["BEAM"], ncand)
    )

    return cursor.lastrowid, False

def _update_cand(cursor, beamrowid, canddf, beamexist, force=False):
    if (beamexist) and (not force): return
    ### for other cases, remove the candidates related to the beamrowid
    cursor.execute(
        """
        DELETE FROM CAND WHERE BEAMROWID={}
        """.format(beamrowid)
    )
    
    candcol = ["beamrowid"] + [col for col in canddf.columns]
    candupdate = """ 
    INSERT INTO CAND ({})
    VALUES ({})
    """.format(", ".join(candcol), ", ".join(["?"] * len(candcol)))

    for i, row in canddf.iterrows():
        cursor.execute(candupdate, [beamrowid] + row.to_list())

def update_database(candfpath, force=False):
    """
    update database based on the candfpath

    TODO: figure out how clustered data will be arranged with Vivek
    for now, we just use "/data/fast/wan342/work/cracorun/50705/00/20230622173926/results/candidates.txtb31.uniq"
    as an example
    """
    candinfo = _extract_fileinfo(candfpath)
    conn = sqlite3.connect("db/craco_cand.sqlite3")
    cursor = conn.cursor()

    canddf = load_candidate(candfpath)

    ### update beam information
    scanrowid, scanexist = _update_scan(cursor, candinfo)
    conn.commit()
    ### update scan information
    beamrowid, beamexist = _update_beam(cursor, candinfo, scanrowid, len(canddf))
    conn.commit()
    ### update candidates
    _update_cand(cursor, beamrowid, canddf, beamexist, force)
    conn.commit()

    conn.close()

### function to load all possibility
def _list_scans():
    """
    list all possible scan from seren...
    """
    pass
    

### layout related functions

layout = dbc.Container(
    html.P("Not available at the moment...")
)