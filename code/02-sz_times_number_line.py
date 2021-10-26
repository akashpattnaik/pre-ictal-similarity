'''
This code generates plots with a number line and points where clinical seizures occur during a patient's EMU stay.
The output of this code is a figure with the number line.
'''
# %%
# Imports and environment setup
import numpy as np
import os, sys
import json
import pandas as pd

from os.path import join as ospj

sys.path.append('tools')

from get_iEEG_duration import get_iEEG_duration
from pull_sz_starts import pull_sz_starts

# set parameters for plotting
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
metadata_path = config['metadataPath']
palette = config['lightColors']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figure')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

with open("../credentials.json") as f:
    credentials = json.load(f)
    username = credentials['username']
    password = credentials['password']

# %%
SEC_TO_DAY = 60 * 60 * 24
# These patients had significant temporal correlation ["HUP070", "HUP111", "HUP187"]
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]
    iEEG_filename = row["portal_ID"]
    print(pt)

    pt_figure_path = ospj("../figures", pt)
    
    if not os.path.exists(pt_figure_path):
        os.makedirs(pt_figure_path)

    duration_day = get_iEEG_duration(username, password, iEEG_filename) / 1e6 / SEC_TO_DAY
    sz_starts = pull_sz_starts(pt, metadata)

    # set up the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_yticks([])
    ax.set_xlim([0, duration_day])
    ax.set_ylim([0, 1])

    points = ax.scatter(
        sz_starts / SEC_TO_DAY, 
        np.zeros(sz_starts.shape), 
        c=palette['2'], 
        zorder=10, 
        edgecolors=palette['1'], 
        linewidths = 0.5
        )

    points.set_clip_on(False)

    ax.set_xlabel("Time in EMU (days)")
    plt.tight_layout()

    ax.spines['bottom'].set_color(palette['1'])
    ax.tick_params(axis='x', colors=palette['1'], which='both')

    ax.xaxis.label.set_color(palette['1'])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    plt.savefig(ospj(pt_figure_path, "sz_time_number_line.svg"), transparent=True)
    plt.close()
# %%
