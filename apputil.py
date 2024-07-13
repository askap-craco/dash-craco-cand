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
from craco.datadirs import DataDirs, SchedDir, ScanDir, RunDir, format_sbid

from astropy.coordinates import SkyCoord
from astropy import units

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
            dbc.NavItem(dbc.NavLink("Monitor", href="/monitor")),
            dbc.NavItem(dbc.NavLink("TAB Filterbank", href="/tab")),
            dbc.NavItem(dbc.NavLink("Localise Source", href="/localise")),
            dbc.NavItem(dbc.NavLink("Multiλ Overlay", href="/multiwave")),
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
def check_cand_ftype(fname):
    ndarr = np.loadtxt(fname, max_rows=5)
    ndarr_shape = ndarr.shape
    if len(ndarr_shape) != 2: return "old"
    if ndarr_shape[-1] == 16: return "new"
    return "old"

def load_candidate(fname, snrcut=None):
    if fname.endswith(".csv"):
        cand_df = pd.read_csv(fname, index_col=0)
        ### rename columns
        cand_df = cand_df.rename(
            columns={
                "boxc_width": "boxcwidth",
                "total_sample": "totalsample",
                "obstime_sec": "obstimesec",
                "dm_pccm3": "dmpccm",
                "ra_deg": "ra",
                "dec_deg": "dec",
            }
        )
        ### change SNR from lower case to upper case...
        if "snr" in cand_df.columns:
            cand_df = cand_df.rename(columns={"snr": "SNR"})
        ### cut
        cand_df = cand_df[cand_df["SNR"] > snrcut]
    else:
        if check_cand_ftype(fname) == "old":
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
        else:
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
                    ('dec', np.float64),
                    ("ibeam", int), # beam index
                    ("latencyms", np.float32) # detection latency
                ]
            )

        cand_np = np.loadtxt(fname, dtype=dtype)
        ### snrcut here - so that we can even load lots of candidates...
        if cand_np is not None:
            cand_np = cand_np[cand_np["SNR"] >= snrcut]
        cand_df = pd.DataFrame(cand_np)

    ### change lpix, mpix, boxcwidth, time, dm, iblk, rawsn, total_sample to integer
    cand_df[
        ["lpix", "mpix", "boxcwidth", "time", "dm", "iblk", "rawsn", "totalsample"]
    ] = cand_df[
        ["lpix", "mpix", "boxcwidth", "time", "dm", "iblk", "rawsn", "totalsample"]
    ].astype(int)

    return cand_df

def fig_to_uri(in_fig, close_all=True, **save_args):
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

def load_figure_from_disk(figpath):
    with open(figpath, "rb") as fp:
        encoded = base64.b64encode(fp.read()).decode()
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
    assert filetype in ["cand_raw", "cand_cls", "cal", "uvfits", "ics", "cas", "meta"], "not an avialable file type..."

    default_dict = dict(
        sbid=None, beam=None, scan=None, tstart=None,
        runname=None, scanpath=None
    )
    query_dict = _update_default_dict(query_dict, default_dict)
    sbid = query_dict["sbid"]; beam = query_dict["beam"]
    scan = query_dict["scan"]; tstart = query_dict["tstart"]
    runname = query_dict["runname"]; scanpath = query_dict["scanpath"]

    ### note scan path from scan/tstart/runname..

    if sbid is None or beam is None:
        return None

    ### reformat beam
    beam = "{:0>2}".format(beam)

    if scanpath is not None:
        if not scanpath.endswith("/"):
            scanpath += "/"
        ddirs = DataDirs()
        scan, tstart = ddirs.path_to_scan(scanpath).split("/")
        runname = ddirs.path_to_runname(scanpath)

    scheddir = SchedDir(sbid)
    ### if you are looking for the calibration file, you don't need to worry about something else
    if filetype == "cal":
        calfile = scheddir.beam_cal_path(beam)
        if _check_file(calfile): return calfile
        return None

    ### get rundir
    if scan is None or tstart is None:
        scantime = None
    else: scantime = f"{scan}/{tstart}"

    try: rundir = RunDir(sbid, scantime, runname)
    except: return None
    ### for the realtime - it won't create run_head_dir...
    # if not _check_file(rundir.run_head_dir): return None
    
    ### for uvfits file
    if filetype == "uvfits":
        uvfitsfile = rundir.scandir.beam_uvfits_path(beam)
        if not _check_file(uvfitsfile): return None
        return uvfitsfile

    if filetype == "ics":
        icsfile = rundir.scandir.beam_ics_path(beam)
        if not _check_file(icsfile): return None
        return icsfile

    if filetype == "cas":
        return None # no available CAS now

    ### for clustered candidates
    if filetype == "cand_raw":
        candfile = rundir.beam_candidate(beam)
        if not _check_file(candfile): return None
        return candfile

    ### for clustered candidates
    if filetype == "cand_cls":
        uniqcandfile = rundir.beam_unique_cand(beam)
        if not _check_file(uniqcandfile): return None
        return uniqcandfile

    if filetype == "meta":
        metafile = rundir.scheddir.metafile
        if not _check_file(metafile): return None
        return metafile

def _check_file(filepath):
    if filepath is None: return False
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

    # print(query_dict)

    query_dict["unclustpath"] = find_file("cand_raw", query_dict)
    query_dict["clustpath"] = find_file("cand_cls", query_dict)
    query_dict["calpath"] = find_file("cal", query_dict)
    query_dict["uvfitspath"] = find_file("uvfits", query_dict)
    query_dict["icspath"] = find_file("ics", query_dict)
    query_dict["caspath"] = find_file("cas", query_dict)
    query_dict["metapath"] = find_file("meta", query_dict)

    return query_dict

def construct_beaminfo(query_dict):
    default_dict = dict(
        sbid=None, beam=None, scan=None, tstart=None,
        runname=None, fname=None, unique=False, scanpath=None
    )
    query_dict = _update_default_dict(query_dict, default_dict)

    if isinstance(query_dict["unique"], str): query_dict["unique"] = eval(query_dict["unique"])
    
    if query_dict["fname"] is None:
        query_dict["fname"] = find_file("cand_cls", query_dict) if query_dict["unique"] else find_file("cand_raw", query_dict)

    # print(query_dict)
    fname = query_dict["fname"].replace(".csv", "")
    # fnamesplit = fname.split("/")
    ### it will update all other information based on the fname
    ddir = DataDirs()
    newdict = dict(
        unique = True if fname.endswith(".uniq") else False,
        beam = _extract_beam_candfile(fname),
        sbid = int(ddir.path_to_sbid(fname)[2:]),
        scan = ddir.path_to_scan(fname).split("/")[0],
        tstart = ddir.path_to_scan(fname).split("/")[1],
        runname = ddir.path_to_runname(fname),
        scanpath = f"{ddir.path_to_scan(fname)}/{ddir.path_to_runname(fname)}"
    )
    query_dict.update(newdict)

    return query_dict

def get_source_name(ra, dec):
    srccoord = SkyCoord(ra, dec, unit=units.degree)
    ra_hms = srccoord.ra.hms
    dec_dms = srccoord.dec.dms

    ra_str = f"{int(ra_hms.h):02d}{int(ra_hms.m):02d}"
    dec_str = f"{int(dec_dms.d):+03d}{int(abs(dec_dms.m)):02d}"

    return f"J{ra_str}{dec_str}"

### for skyview...
skyview_allsurveys = {'Allbands:GOODS/HDF/CDF': ['GOODS: Chandra ACIS HB',
                             'GOODS: Chandra ACIS FB',
                             'GOODS: Chandra ACIS SB',
                             'GOODS: VLT VIMOS U',
                             'GOODS: VLT VIMOS R',
                             'GOODS: HST ACS B',
                             'GOODS: HST ACS V',
                             'GOODS: HST ACS I',
                             'GOODS: HST ACS Z',
                             'Hawaii HDF U',
                             'Hawaii HDF B',
                             'Hawaii HDF V0201',
                             'Hawaii HDF V0401',
                             'Hawaii HDF R',
                             'Hawaii HDF I',
                             'Hawaii HDF z',
                             'Hawaii HDF HK',
                             'GOODS: HST NICMOS',
                             'GOODS: VLT ISAAC J',
                             'GOODS: VLT ISAAC H',
                             'GOODS: VLT ISAAC Ks',
                             'HUDF: VLT ISAAC Ks',
                             'GOODS: Spitzer IRAC 3.6',
                             'GOODS: Spitzer IRAC 4.5',
                             'GOODS: Spitzer IRAC 5.8',
                             'GOODS: Spitzer IRAC 8.0',
                             'GOODS: Spitzer MIPS 24',
                             'GOODS: Herschel 100',
                             'GOODS: Herschel 160',
                             'GOODS: Herschel 250',
                             'GOODS: Herschel 350',
                             'GOODS: Herschel 500',
                             'CDFS: LESS',
                             'GOODS: VLA North'],
  'Allbands:HiPS': ['UltraVista-H',
                    'UltraVista-J',
                    'UltraVista-Ks',
                    'UltraVista-NB118',
                    'UltraVista-Y',
                    'CFHTLS-W-u',
                    'CFHTLS-W-g',
                    'CFHTLS-W-r',
                    'CFHTLS-W-i',
                    'CFHTLS-W-z',
                    'CFHTLS-D-u',
                    'CFHTLS-D-g',
                    'CFHTLS-D-r',
                    'CFHTLS-D-i',
                    'CFHTLS-D-z'],
  'GammaRay': ['Fermi 5',
               'Fermi 4',
               'Fermi 3',
               'Fermi 2',
               'Fermi 1',
               'EGRET (3D)',
               'EGRET <100 MeV',
               'EGRET >100 MeV',
               'COMPTEL'],
  'HardX-ray': ['INT GAL 17-35 Flux',
                'INT GAL 17-60 Flux',
                'INT GAL 35-80 Flux',
                'INTEGRAL/SPI GC',
                'GRANAT/SIGMA',
                'RXTE Allsky 3-8keV Flux',
                'RXTE Allsky 3-20keV Flux',
                'RXTE Allsky 8-20keV Flux'],
  'IR:2MASS': ['2MASS-J', '2MASS-H', '2MASS-K'],
  'IR:AKARI': ['AKARI N60', 'AKARI WIDE-S', 'AKARI WIDE-L', 'AKARI N160'],
  'IR:IRAS': ['IRIS  12',
              'IRIS  25',
              'IRIS  60',
              'IRIS 100',
              'SFD100m',
              'SFD Dust Map',
              'IRAS  12 micron',
              'IRAS  25 micron',
              'IRAS  60 micron',
              'IRAS 100 micron'],
  'IR:Planck': ['Planck 857',
                'Planck 545',
                'Planck 353',
                'Planck 217',
                'Planck 143',
                'Planck 100',
                'Planck 070',
                'Planck 044',
                'Planck 030'],
  'IR:UKIDSS': ['UKIDSS-Y', 'UKIDSS-J', 'UKIDSS-H', 'UKIDSS-K'],
  'IR:WISE': ['WISE 3.4', 'WISE 4.6', 'WISE 12', 'WISE 22'],
  'IR:WMAP&COBE': ['WMAP ILC',
                   'WMAP Ka',
                   'WMAP K',
                   'WMAP Q',
                   'WMAP V',
                   'WMAP W',
                   'COBE DIRBE/AAM',
                   'COBE DIRBE/ZSMA'],
  'Optical:DSS': ['DSS',
                  'DSS1 Blue',
                  'DSS1 Red',
                  'DSS2 Red',
                  'DSS2 Blue',
                  'DSS2 IR'],
  'Optical:SDSS': ['SDSSg',
                   'SDSSi',
                   'SDSSr',
                   'SDSSu',
                   'SDSSz',
                   'SDSSdr7g',
                   'SDSSdr7i',
                   'SDSSdr7r',
                   'SDSSdr7u',
                   'SDSSdr7z'],
  'OtherOptical': ['Mellinger Red',
                   'Mellinger Green',
                   'Mellinger Blue',
                   'NEAT',
                   'H-Alpha Comp',
                   'SHASSA H',
                   'SHASSA CC',
                   'SHASSA C',
                   'SHASSA Sm'],
  'ROSATDiffuse': ['RASS Background 1',
                   'RASS Background 2',
                   'RASS Background 3',
                   'RASS Background 4',
                   'RASS Background 5',
                   'RASS Background 6',
                   'RASS Background 7'],
  'ROSATw/sources': ['RASS-Cnt Soft',
                     'RASS-Cnt Hard',
                     'RASS-Cnt Broad',
                     'PSPC 2.0 Deg-Int',
                     'PSPC 1.0 Deg-Int',
                     'PSPC 0.6 Deg-Int',
                     'HRI'],
  'Radio:GHz': ['CO',
                'GB6 (4850MHz)',
                'VLA FIRST (1.4 GHz)',
                'NVSS',
                'Stripe82VLA',
                '1420MHz (Bonn)',
                'HI4PI',
                'EBHIS',
                'nH'],
  'Radio:GLEAM': ['GLEAM 72-103 MHz',
                  'GLEAM 103-134 MHz',
                  'GLEAM 139-170 MHz',
                  'GLEAM 170-231 MHz'],
  'Radio:MHz': ['SUMSS 843 MHz',
                '0408MHz',
                'WENSS',
                'TGSS ADR1',
                'VLSSr',
                '0035MHz'],
  'SoftX-ray': ['SwiftXRTCnt', 'SwiftXRTExp', 'SwiftXRTInt', 'HEAO 1 A-2'],
  'SwiftUVOT': ['UVOT WHITE Intensity',
                'UVOT V Intensity',
                'UVOT B Intensity',
                'UVOT U Intensity',
                'UVOT UVW1 Intensity',
                'UVOT UVM2 Intensity',
                'UVOT UVW2 Intensity'],
  'UV': ['GALEX Near UV',
         'GALEX Far UV',
         'ROSAT WFC F1',
         'ROSAT WFC F2',
         'EUVE 83 A',
         'EUVE 171 A',
         'EUVE 405 A',
         'EUVE 555 A'],
  'X-ray:SwiftBAT': ['BAT SNR 14-195',
                     'BAT SNR 14-20',
                     'BAT SNR 20-24',
                     'BAT SNR 24-35',
                     'BAT SNR 35-50',
                     'BAT SNR 50-75',
                     'BAT SNR 75-100',
                     'BAT SNR 100-150',
                     'BAT SNR 150-195']}

decam_allsurveys = {
    "ls-dr10": dict(res=0.06, ),
    "decaps2": dict(res=0.06, ),
    "galex": dict(res=0.06, ),
    # "halpha": dict(res=0.06, ) # was not available during the development
    "vlass1.2": dict(res=0.06, ),
}