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
from scipy.signal import iirnotch, filtfilt, butter
import re
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
figure_path = ospj(repo_path, 'figures')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort_test.xlsx"))

with open("../credentials.json") as f:
    credentials = json.load(f)
    username = credentials['username']
    password = credentials['password']

def _laplacian_reference(data):
    columns = data.columns

    # separate contact names
    electrodes = []
    contacts = []
    for i in columns:
        m = re.match(r"(\D+)(\d+)", i)
        electrodes.append(m.group(1))
        contacts.append(int(m.group(2)))

    # find channel before and after, if it's on the same electrode
    indices_to_average = {}
    for i in range(n_channels):
        electrode = electrodes[i]
        if i == 0:
            electrode_post = electrodes[i + 1]
            if electrode == electrode_post:
                indices_to_average[columns[i]] = [i + 1]
        elif i == n_channels - 1:
            electrode_pre = electrodes[i - 1]
            if electrode == electrode_pre:
                indices_to_average[columns[i]] = [i - 1]
        else:
            electrode_pre = electrodes[i - 1]
            electrode_post = electrodes[i + 1]
            avg_li = []
            if electrode == electrode_pre:
                avg_li.append(i - 1)
            if electrode == electrode_post:
                avg_li.append(i + 1)
            indices_to_average[columns[i]] = avg_li

    # subtract mean of two nearby channels and return
    for electrode, inds in indices_to_average.items():
        data[electrode] = data[electrode] - data.iloc[:, inds].mean(axis=1)
    return data

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

        # extract dims
        n_samples = np.size(clip_data, axis=0)
        n_channels = np.size(clip_data, axis=1)

        # set time array
        t_sec = np.linspace(sz_start, sz_end, n_samples)

        # indices for 5 second non-overlapping windows
        win_size = 1 * fs
        ind_overlap = np.reshape(np.arange(len(t_sec)), (-1, int(win_size)))
        n_windows = np.size(ind_overlap, axis=0)

        # nan check
        nan_mask = np.ones(n_samples, dtype=bool)
        for win_inds in ind_overlap:
            if np.sum(np.isnan(clip_data.iloc[win_inds, :]), axis=0).any():
                nan_mask[win_inds] = False
            if (np.sum(np.abs(clip_data.iloc[win_inds, :]), axis=0) < 1/12).any():
                nan_mask[win_inds] = False
            if (np.sqrt(np.sum(np.diff(clip_data.iloc[win_inds, :]))) > 15000).any():
                nan_mask[win_inds] = False
            
        signal_nan = clip_data[nan_mask]
        t_sec_nan = t_sec[nan_mask]

        if len(t_sec_nan) == 0:
            continue

        # remove 60Hz noise
        f0 = 60.0  # Frequency to be removed from signal (Hz)
        Q = 30.0  # Quality factor
        b, a = iirnotch(f0, Q, fs)
        print(signal_nan.shape)
        signal_filt = filtfilt(b, a, signal_nan, axis=0)
        bandpass_b, bandpass_a = butter(3, [1, 120], btype='bandpass', fs=fs)
        print(signal_filt.shape)
        signal_filt = filtfilt(bandpass_b, bandpass_a, signal_filt, axis=0)

        signal_filt = pd.DataFrame(signal_filt, columns=signal_nan.columns, index=signal_nan.index)

        # set index to time
        signal_filt.index = pd.to_timedelta(t_sec_nan, unit="S")

        signal_ref = _laplacian_reference(signal_filt)

        # Plot the data
        t_sec = np.linspace(0, sz_end - sz_start, num=clip_data.shape[0])
        fig, ax = plot_iEEG_data(signal_ref, t_sec, linecolor=palette['1'])

        ax.spines['bottom'].set_color(palette['1'])
        ax.tick_params(axis='x', colors=palette['1'], which='both')
        ax.tick_params(axis='y', colors=palette['1'], which='both')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.xaxis.label.set_color(palette['1'])

        ax.set_title("Seizure {}".format(ind + 1), color=palette['1'])

        plt.savefig(ospj(pt_figure_path, "sz_{}_plot.svg".format(ind)), transparent=True, bbox_inches='tight')
        plt.savefig(ospj(pt_figure_path, "sz_{}_plot.png".format(ind)), transparent=True, bbox_inches='tight')
        # plt.close()
        
# %%
