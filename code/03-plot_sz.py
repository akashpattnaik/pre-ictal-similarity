'''
This script creates plots of the seizures.
'''
#%%
# %load_ext autoreload
# %autoreload 2
# Imports and environment setup
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import json
import pandas as pd
import time
from scipy.io import loadmat

from os.path import join as ospj

sys.path.append('tools')

from get_iEEG_data import get_iEEG_data
from plot_iEEG_data import plot_iEEG_data
from pull_sz_starts import pull_sz_starts
from pull_sz_ends import pull_sz_ends


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
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]
    iEEG_filename = row["portal_ID"]

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    # pull and format electrode metadata
    electrodes_mat = loadmat(ospj(pt_data_path, 'target-electrodes-regions.mat'))
    target_electrode_region_inds = electrodes_mat['targetElectrodesRegionInds'][0] - 1

    sz_starts = pull_sz_starts(pt, metadata)
    sz_ends = pull_sz_ends(pt, metadata)

    # Plot and save each seizure
    for ind, sz_start, sz_end in zip(range(len(sz_starts)), sz_starts, sz_ends):
        try:
            clip_data, fs = get_iEEG_data(username, password, row["portal_ID"], sz_start * 1e6, sz_end * 1e6, select_electrodes=list(target_electrode_region_inds))
        except:
            # catches exceptions from iEEG.org and retries
            time.sleep(5)
            clip_data, fs = get_iEEG_data(username, password, row["portal_ID"], sz_start * 1e6, sz_end * 1e6, select_electrodes=list(target_electrode_region_inds))

        # Plot the data
        t_sec = np.linspace(0, sz_end - sz_start, num=clip_data.shape[0])
        fig, ax = plot_iEEG_data(clip_data, t_sec, linecolor=palette['1'])

        ax.spines['bottom'].set_color(palette['1'])
        ax.tick_params(axis='x', colors=palette['1'], which='both')
        ax.tick_params(axis='y', colors=palette['1'], which='both')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.xaxis.label.set_color(palette['1'])

        ax.set_title("Seizure {}".format(ind + 1), color=palette['1'])

        plt.savefig(ospj(pt_figure_path, "sz_{}_plot.svg".format(ind + 1)), transparent=True, bbox_inches='tight')
        plt.savefig(ospj(pt_figure_path, "sz_{}_plot.png".format(ind + 1)), transparent=True, bbox_inches='tight')
        plt.close()
        
# %%
