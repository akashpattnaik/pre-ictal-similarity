'''
This script plots spectrograms for pre-ictal periods.
'''
# %%
# Imports and environment setup
import numpy as np
import sys
import pandas as pd
import json
from scipy.io import loadmat
import matplotlib.pyplot as plt

from os.path import join as ospj

sys.path.append('tools')
from pull_sz_starts import pull_sz_starts
from plot_spectrogram import plot_spectrogram

from time2ind import time2ind

# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
metadata_path = config['metadataPath']
palette = config['lightColors']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

PLOT = True

FIXED_PREICTAL_SEC = 60 * 30
LEAD_SZ_WINDOW_SEC = (FIXED_PREICTAL_SEC + 60 * 15) # 15 min buffer

# %%
n_removed_sz = {}
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]

    print(pt)

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    sz_starts = pull_sz_starts(pt, metadata)

    bandpower_mat_data = loadmat(ospj(pt_data_path, "bandpower-windows-12hr.mat"))
    # log transform bandpower
    bandpower_data = 10*np.log10(bandpower_mat_data['allFeats'])
    t_sec = np.squeeze(bandpower_mat_data['entireT']) / 1e6
    sz_id = np.squeeze(bandpower_mat_data['szID'])

    n_sz = np.size(np.unique(sz_id))
    
    lead_sz = np.diff(np.insert(sz_starts, 0, [0])) > LEAD_SZ_WINDOW_SEC # 15 min buffer
    
    # which seizures are kept and which should be removed
    remaining_sz_ids = np.where(lead_sz)[0] + 1
    remove_sz_ids = np.where(~lead_sz)[0] + 1

    # remove non-lead seizures
    for remv in remove_sz_ids:
        t_sec = np.delete(t_sec, np.where(sz_id == remv))
        bandpower_data = np.delete(bandpower_data, np.where(sz_id == remv), axis=0)
        sz_id = np.delete(sz_id, np.where(sz_id == remv))

    pi_starts = sz_starts[lead_sz] - FIXED_PREICTAL_SEC
    pi_ends = sz_starts[lead_sz]

    pi_start_ind = [time2ind(i, t_sec) for i in pi_starts]
    pi_end_ind = [time2ind(i, t_sec) for i in pi_ends]
    
    for i in range(len(pi_start_ind)):
        ax = plot_spectrogram(bandpower_data[pi_start_ind[i]:pi_end_ind[i], :], start_time=(pi_starts[i] - pi_ends[i]) / 60, end_time=0)
        ax.set_xlabel("Time from seizure onset (min)")
        ax.set_title("Seizure {}".format(remaining_sz_ids[i]))
        plt.savefig(ospj(pt_figure_path, "pi_bandpower_spectrogram_sz_{}.svg".format(remaining_sz_ids[i])))
        plt.savefig(ospj(pt_figure_path, "pi_bandpower_spectrogram_sz_{}.png".format(remaining_sz_ids[i])))
        plt.close()
