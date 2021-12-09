'''
This script applies a line length detector to 5 second clips of iEEG to 
determine patient specific thresholds for seizures to all channels.

Line length in 5 second, non-overlapping windows are saved for each channel and 
the corresponding time array is also saved in a compressed numpy data (.npz) 
file.
'''
# %%
# Imports and environment setup
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.io import loadmat
from scipy import signal
from time import sleep
from tqdm import tqdm

from os.path import join as ospj

sys.path.append('tools')
# from line_length import line_length
from get_iEEG_data import get_iEEG_data
from plot_iEEG_data import plot_iEEG_data
from get_iEEG_duration import get_iEEG_duration
from pull_patient_localization import pull_patient_localization

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

with open("../credentials.json") as f:
    credentials = json.load(f)
    username = credentials['username']
    password = credentials['password']

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

PLOT = False
SAVE_PLOT = False

def _line_length(signal):
    return np.sum(np.abs(np.diff(signal, axis=0)), axis=0) / signal.shape[0]

pt = "HUP187"
iEEG_fname = "HUP187_phaseII"

# def detector(pt, iEEG_fname):
pt_data_path = ospj(data_path, pt)
pt_figure_path = ospj(figure_path, pt)

# pull and format electrode metadata
electrodes_mat = loadmat(ospj(pt_data_path, 'target-electrodes-regions.mat'))
target_electrode_region_inds = electrodes_mat['targetElectrodesRegionInds'][0] - 1

patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(ospj(metadata_path, 'patient_localization_final.mat'))
ignore_electrodes = np.squeeze(ignore[patients.index(pt)]) == 1

duration_usec = get_iEEG_duration(username, password, iEEG_fname)
duration_sec = duration_usec / 1e6

LL_WIN_SIZE = 5 # time (seconds) for each clip to calculate line length
PULL_WIN_SIZE = 30 * 60 # time (seconds) for each clip to pull

# %%
n_pull = np.int(np.ceil( duration_sec / PULL_WIN_SIZE ))
n_windows_in_pull = np.int(PULL_WIN_SIZE / LL_WIN_SIZE)
ll_arr = []
t_arr = []
for i_pull in tqdm(range(n_pull)):
    start_time_sec = i_pull * PULL_WIN_SIZE
    end_time_sec = start_time_sec + PULL_WIN_SIZE

    try:
        data, fs = get_iEEG_data(
            username, 
            password, 
            iEEG_fname, 
            start_time_sec*1e6,
            end_time_sec*1e6,
            # select_electrodes=target_electrode_region_inds
            select_electrodes=list(np.where(ignore_electrodes == 0)[0])
            )

    except:
        # tqdm.write("Some type of exception")
        sleep(2)
        continue

    # create time array based on start and end times
    t = np.linspace(start_time_sec, end_time_sec, data.shape[0])

    # filters
    bp_sos = signal.butter(4, [4, 120], 'bandpass', fs=fs, output='sos')
    notch_b, notch_a = signal.iirnotch(60, 30.0, fs)

    data_filt = signal.filtfilt(notch_b, notch_a, data)
    data_filt = signal.sosfilt(bp_sos, data_filt)

    data[:] = data_filt

    for i_win in range(n_windows_in_pull):
        start_ind = np.int(i_win * LL_WIN_SIZE * fs)
        end_ind = np.int(start_ind + LL_WIN_SIZE * fs)

        data_clip = data.iloc[start_ind:end_ind, :]
        ll_arr.append(_line_length(data_clip))
        t_arr.append(t[start_ind])

    if PLOT:
        fig, ax = plot_iEEG_data(data, t)
        if SAVE_PLOT:
            plt.savefig(ospj(pt_figure_path, "post_filter_sig.svg"), bbox_inches='tight', transparent='true')
            plt.savefig(ospj(pt_figure_path, "post_filter_sig.png"), bbox_inches='tight', transparent='true')
            plt.close()
    if PLOT:
        fig, ax = plt.subplots()
        ax.plot(t_arr, ll_arr)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Line Length")
        if SAVE_PLOT:
            plt.savefig(ospj(pt_figure_path, "post_filter_ll.svg"), bbox_inches='tight', transparent='true')
            plt.savefig(ospj(pt_figure_path, "post_filter_ll.png"), bbox_inches='tight', transparent='true')
            plt.close()

ll_arr = np.array(ll_arr)
t_arr = np.array(t_arr)
np.savez(ospj(pt_data_path, "ll_arr.npz"), ll_arr, t_arr)
