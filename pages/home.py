from dash import Dash, html, callback, dcc
import dash_bootstrap_components as dbc
import dash

dash.register_page(__name__, path="/", title="CRACO Candidate HOME")

filefinder = """
The candidate file finder and loader can be accessed [here](/beamfile).
In the `Candidate File Input` section, you can specify the sbid of the observation.
It will automatically load all possible scans (pipeline runs) under seren. 
If there is a pipeline run named `results`, then it will be the default selection.
After you select a scan/run, you are able to input the beam number (from 0-35).
If there are some files under the scan/run, you will see some important files listed in the page.
They are 
- `unclustered` - unique candidate file
- `clustered` - candidate file from the pipeline
- `uvfits` - uvfits file path
- `calfile` - calibration file for the given SBID and beam
*Note* some files will be invisible if they do not exist

Also you will see two buttons if two candidate files exist - `clustered (recom)` and `unclustered (no)`.
These two buttons will navigate you to the all candidate plotter tools for further investigation.
We highly recommend you using the `clustered (recom)` button, as the unclustered one will take a while to render.
"""

coordmater = """
This tool is located in the same page as the file finder/loader ([here](/beamfile)).
Input R.A. and Decl. to the input boxes (you can either use `12h34m56s -65d32m10s`, `10.00d -10.2d`, or `10 10 10 -10 10 10`).
If the coordinate is parsed successfully, you will see the button `Go To Coord` turn to green.
Click this button, you will jump into the single candidate plotter. 
In that tool, you can check if there is any match between the target and ATNF pulsar catalog, RACS Low catalog, and SIMBAD.
"""

allcandplotter = """
This tool is located at [here](/beam). If you want to access it directly, you can use `[server_url]/beam`. 
We allow you to input some parameter in the url if you access it directly.
- You can use the file name to load a candidate file directly by passing the path of the file to the `fname` argument.
For example, `[server_url]/beam?fname=/data/seren-02/big/craco/SB050668/scans/00/20230622062301/results/clustering_output/candidates.txtb01.uniq`
- You can also use information of the observation, such as `sbid`, `beam`, and `scanpath`. 
If `scanpath` is not specified, the tool will try to find the `results` run or the first run it finds on `seren`. For example, you can use
`[server_url]/beam?sbid=50668&scanpath=00/20230622062301/results&beam=1&unique=True`. If `unique` is not specified, the default value is `True`, 
in other word, load the clustered file.

The SNR slider can be used to control the lower limit of the candidate to be plotted in the webpage. It is a global setting. You will not see any candidate
below the threshold in any figures or tables in this page.

Three out of four figures are connected to each other - top left, top right, bottom left. You can use selection tools to select regions. If you do 
multiple selections in these three images, we will only display the cross section of the candidates. The `Selection Region Indicator` helps you to identify
which figure(s) has selected region in it. (double click the figure to get ride of the selection).
The detail of all selected sources (i.e., visible sources) will be displayed in the `Filtered Candidate Table`. In the table,
you can sort the value based on one or more columns, filter by condition (the cell under the column name). You can also select the row by clicking the box on the left side.
Once selected, you will see the information of the candidate is copied into the `Selected Candidate` table.
If you want to have a more detailed investigation (e.g., crossmatch with external catalog, produce filterbank images, produce synthesized images), press the `push` button.
*Note*: if more rows are selected, the `push` button will only push the first row to the single candidate plotter

The figure at the bottom right are the candidates within a given window (width) centered at a given totalsample (slider). If you click any points in the three figures,
the value in the slider will automatically change to the totalsample value of the candidate you just clicked on.
and also you can see a more detailed information of the source in the `Clicked Candidate` table. 
Same as `Selected Candidate`, if you want to have a detailed investigation, press the `Push` button
"""

singcandplotter = """
This tool is located at [here](/candidate). To access it directly, using `[server_url]/candidate`.
You can also input some parameters as a url query. It accepts
`sbid`, `beam`, `scan`, `tstart`, `results`, `dm`, `boxcwidth`, `lpix`, `mpix`, `totalsample`, `ra`, `dec`.
Some missing parameters are acceptable, but you cannot use all features in this webpage.
- *candidate related files* You need to provide `sbid`, `beam`, `scan`, `tstart` (not neccessary, can be guessed by the tool), `results` (not neccessary, by default, it is "results")
- *catalogue crossmatching* You need to provide `ra` and `dec` (in degree)
- *craco data generating and plotting* You need to provide all information (except `tstart` and `results`, as mentioned in the first bullet point)

If you see a copy icon, you can click the icon and copy the content on the left to the clipboard (which might be helpful?)

In the candidate external information, you will see the matches in ATNF pulsar catalogue, RACS low DR1 catalogue, and SIMBAD server (within 30 arcsec).
If there is a match(es), the button will turn to be green. and `ATNF PSRCAT` and `SIMBAD` button are clickable (if it turns to green).
Sometimes you will see `NO_INTERNET` tables, if that happens, refresh the page to retrieve the matches again.
There is also a link to RACS HIPS map under the RACS table.

If you think this is an interesting candidate, you can press `Process` button to generate filterbank, make images from CRACO data.
For both filterbank plots and synthesized images, the first row is not interactive, but the second row is.

Figures for filterbank
- filterbank dedispered at DM=0; filterbank dedispered at the DM specified in the url query; butterfly plot (i.e., DM-time plot)
- dedispered filterbank plot (interactive); average/maximum/minimum profile as a function of time (interactive)
you can change the color scale of the filterbank plot but pressing `Mannual Color Scale Disabled` button, and adjust the scale use the slider below.
To disabled the mannual color scale, just click the button again.
Red means disabled, and green means enabled (I forgot to change the label on the button... might be confusing... sorry!)

Figures for images
- maximum image (the image with the hottest pixel around the candidate region); median image; std image
- std image (interactive); zoom-in images (interactive and animation); detection image (interactive), which is the mean image when the detection happens (i.e., corresponding to the boxcwidth)
"""

layout = dbc.Container(
    [
        html.P("This is an interactive tool for inspecting candidates"),
        dbc.Container([
            dbc.Row(html.H5("Candidate File Finder/Loader Interface")),
            dbc.Row(dcc.Markdown(filefinder)),
            dbc.Row(html.H5("Candidate Coordinate Matcher Interface")),
            dbc.Row(dcc.Markdown(coordmater)),
            dbc.Row(html.H5("All Candidate Plotter Interface")),
            dbc.Row(dcc.Markdown(allcandplotter)),
            dbc.Row(html.H5("Single Candidate Plotter Interface")),
            dbc.Row(dcc.Markdown(singcandplotter)),
        ])
    ]
)
