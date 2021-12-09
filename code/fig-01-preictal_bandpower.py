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
electrodes_opt = config['electrodes']
band_opt = config['bands']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort_test.xlsx"))

preictal_window_min = config['preictal_window_min']

# %%
n_removed_sz = {}
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]

    print(pt)

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    sz_starts = pull_sz_starts(pt, metadata)

    df = pd.read_pickle(ospj(pt_data_path, "bandpower_elec-{}_period-preictal.pkl".format(electrodes_opt)))

    if band_opt == "all":
        bandpower_data = df.filter(regex=("^((?!broad).)*$"), axis=1)
        bandpower_data  = bandpower_data.drop(['Seizure id'], axis=1)
    elif band_opt == "broad":
        bandpower_data = df.filter(regex=("broad"), axis=1)
    else:
        sys.exit("Band configuration not given properly")
    sz_id = np.squeeze(df['Seizure id'])
    t_sec = np.array(df.index / np.timedelta64(1, 's'))
    n_sz = np.size(np.unique(sz_id))
        
    remaining_sz_ids = np.load(ospj(pt_data_path, "remaining_sz_ids.npy"))

    for i in remaining_sz_ids:
        ax = plot_spectrogram(bandpower_data[sz_id == i], start_time=(t_sec[sz_id == i][0] - t_sec[sz_id == i][-1]) / 60, end_time=0)
        ax.set_xlabel("Time from seizure onset (min)")
        ax.set_title("Seizure {}".format(remaining_sz_ids[i]))
        ax.set_xlim([-2, 0])
        plt.savefig(ospj(pt_figure_path, "spectrogram_band-{}_elec-{}_sz-{}.svg".format(band_opt, electrodes_opt, remaining_sz_ids[i])))
        plt.savefig(ospj(pt_figure_path, "spectrogram_band-{}_elec-{}_sz-{}.png".format(band_opt, electrodes_opt, remaining_sz_ids[i])))
        # plt.close()

# %%
