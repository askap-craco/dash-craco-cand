### CRACO Offline Candidate Plotter

This tool can facilitate the inspection of CRACO 110-ms pipeline candidates.

To use this tool, you need to install `dash` related packages.
Create a new environment, and install the packages listed under `requirements.txt`.
In addition, you also need to install [`craco-python`](https://github.com/askap-craco/craco-python) and [`craco`](https://github.com/askap-craco/craft) package.

To deploy the tool, `cd` to the directory, and run 
`gunicorn app:server -b :[port]`
Then you can access the tool by going to `localhost:[port]` in your browser.

I would suggest you running it in the background, 
```
tmux new -s <name_of_your_session>
source <path_to_your_venv>/bin/activate 
cd <path of this dash-craco-cand>
gunicorn app:server -b :[port]
```
`[Control + B]` + `[D]` to exit

For a detailed description of all features in this tool, please navigate to the home page once you set everything up.