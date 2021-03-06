#%%
# %load_ext autoreload
# %autoreload 2

import os, sys
sys.path.append('tools')

from get_iEEG_data import get_iEEG_data
from plot_iEEG_data import plot_iEEG_data
from line_length import line_length
from get_iEEG_duration import get_iEEG_duration
from pull_sz_starts import pull_sz_starts
from pull_sz_ends import pull_sz_ends

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
from scipy.io import loadmat, savemat
import pandas as pd
import re
from tqdm import tqdm
from tqdm.auto import trange


from scipy.signal import iirnotch, filtfilt, butter, coherence
from os.path import join as ospj
from bct import strengths_und

import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
metadata_path = config['metadataPath']
palette = config['lightColors']
DTW_FLAG = config['flags']["DTW_FLAG"]
electrodes_opt = config['electrodes']
band_opt = config['bands']
preictal_window_min = config['preictal_window_min']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

# size of window for each downloaded data chunk
data_pull_min = 5

if len(sys.argv) < 2:
    sys.exit("No argument given")
period = sys.argv[1]
# period = "preictal" # "preictal" or "ictal"
# params_table = pd.read_excel(os.path.join(data_path, 'nnmf_pipeline_params_firsttwo.xlsx'))
patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

f0 = 60.0  # Frequency to be removed from signal (Hz)
Q = 30.0  # Quality factor

delta_band = [1, 4]
theta_band = [4, 8]
alpha_band = [8, 13]
beta_band = [13, 30]
gamma_band = [30, 70]
high_gamma_band = [70, 120]
broad_band = [5, 115]

bands = [
    delta_band,
    theta_band,
    alpha_band,
    beta_band,
    gamma_band,
    high_gamma_band,
    broad_band
]

band_names = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "high_gamma",
    "broad"
]
# %%
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
            if len(avg_li) == 0:
                avg_li.extend([i - 1, i + 1])
            indices_to_average[columns[i]] = avg_li
    # subtract mean of two nearby channels and return
    for electrode, inds in indices_to_average.items():
        data[electrode] = data[electrode] - data.iloc[:, inds].mean(axis=1)
    return data

def _common_average_reference(data):
    data = data.subtract(data.mean(axis=1), axis=0)
    return data

# %%
def _coherence(data_hat, fs, band, return_mode="node_strength"):
    '''
    Adapted from Andy Revell and Ankit Khambhati

    Uses coherence to compute a band-specific functional network from ECoG
    Parameters
    ----------
        data_hat: ndarray, shape (T, N)
            Input signal with T samples over N variates
        fs: int
            Sampling frequency
        band: list
            Frequency range over which to compute coherence [-NW+C, C+NW]

    Returns
    -------
        Node_strength: ndarray, shape (N,)
            node strength of each channel
    '''

    data_hat = np.array(signal_ref.iloc[win_inds, :])
    # Get data_hat attributes
    n_samp, n_chan = data_hat.shape

    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    adj = np.zeros((n_chan, n_chan))

    for n1, n2 in zip(triu_ix, triu_iy):
        out = coherence(x= data_hat[:, n1],
                                y = data_hat[:, n2],
                                fs = fs,
                                window= range(int(fs-fs/3)) #if n_samp = fs, the window has to be less than fs, or else you will get output as all ones. So I modified to be fs - fs/3, and not just fs
                                )

        # Find closest frequency to the desired center frequency
        cf_idx = np.flatnonzero((out[0] >= band[0]) &
                                (out[0] <= band[1]))

        # Store coherence in association matrix
        adj[n1, n2] = np.mean(out[1][cf_idx])

    adj += adj.T
    return strengths_und(adj)
    
# %%
# Get credentials
with open("../credentials.json") as f:
    credentials = json.load(f)
    username = credentials['username']
    password = credentials['password']
    
# %%
pbar = tqdm(patient_cohort.iterrows(), total=len(patient_cohort))
for index, row in pbar:
    if row['Ignore']:
        continue

    pt = row['Patient']
    iEEG_filename = row['portal_ID']

    pbar.set_description("Calculating features for {}".format(pt))

    # tqdm.write("Calculating features for {}".format(pt))
    pt_data_path = ospj(data_path, pt)

    target_electrodes_vars = loadmat(ospj(pt_data_path, "selected_electrodes_elec-{}.mat".format(electrodes_opt)))
    electrodes = list(target_electrodes_vars['targetElectrodesRegionInds'][0])

    duration = get_iEEG_duration(username, password, iEEG_filename)

    sz_starts = pull_sz_starts(pt, metadata)
    sz_ends = pull_sz_ends(pt, metadata)

    # remove short inter-seizure intervals
    lead_sz = np.diff(np.insert(sz_starts, 0, [0])) > ((preictal_window_min + 15) * 60) # 15 min buffer
    remaining_sz_ids = np.where(lead_sz)[0]

    band_powers = {}
    for name in band_names:
        band_powers[name] = None

    all_signal = None
    t_sec_arr = []
    sz_id_arr = []
    for sz_id, sz_start in tqdm(enumerate(sz_starts), total=len(sz_starts), desc='seizure', leave=False):
        if period == "preictal":
            duration_usec = preictal_window_min * 60 * 1e6
            n_iter = int(np.floor(preictal_window_min / data_pull_min))

            data_duration_usec = data_pull_min * 60 * 1e6

            preictal_start_sec = sz_start - preictal_window_min * 60
            preictal_start_usec = preictal_start_sec * 1e6
        elif period == "ictal":
            sz_start_usec = sz_start * 1e6
            sz_end_usec = sz_ends[sz_id] * 1e6

            # extend pull time to the nearest second
            duration_usec = sz_end_usec - sz_start_usec 
            duration_usec = np.ceil(duration_usec / 1e6) * 1e6
            duration_min = duration_usec / (1e6 * 60)
            n_iter = int(np.ceil(duration_min / data_pull_min))
        else:
            sys.exit("invalid period given")

        for i in trange(n_iter, desc='window', leave=False):
            if period == "preictal":
                start_usec = preictal_start_usec + i * data_duration_usec
            elif period == "ictal":
                start_usec = sz_start_usec
                data_duration_usec = duration_usec

            data, fs = get_iEEG_data(username, password, iEEG_filename, start_usec, start_usec + data_duration_usec, select_electrodes=electrodes)
            
            # extract dims
            n_samples = np.size(data, axis=0)
            n_channels = np.size(data, axis=1)

            # set time array
            t_usec = np.linspace(start_usec, start_usec + data_duration_usec, n_samples)
            t_sec = t_usec / 1e6

            # indices for 5 second non-overlapping windows
            win_size = int(1 * fs)
            ind_overlap = np.reshape(np.arange(len(t_sec)), (-1, int(win_size)))
            n_windows = np.size(ind_overlap, axis=0)

            # nan check
            nan_mask = np.ones(n_samples, dtype=bool)
            for win_inds in ind_overlap:
                if np.sum(np.isnan(data.iloc[win_inds, :]), axis=0).any():
                    nan_mask[win_inds] = False
                if (np.sum(np.abs(data.iloc[win_inds, :]), axis=0) < 1/12).any():
                    nan_mask[win_inds] = False
                if (np.sqrt(np.sum(np.diff(data.iloc[win_inds, :]))) > 15000).any():
                    nan_mask[win_inds] = False
                
            signal_nan = data[nan_mask]
            t_sec_nan = t_sec[nan_mask]

            # if artifact rejection finds that the entire data pull is artifact, skip this iteration
            if len(t_sec_nan) == 0:
                continue
            
            # remove 60Hz noise
            f0 = 60.0  # Frequency to be removed from signal (Hz)
            Q = 30.0  # Quality factor
            b, a = iirnotch(f0, Q, fs)
            signal_filt = filtfilt(b, a, signal_nan, axis=0)

            # bandpass between 1 and 120Hz
            bandpass_b, bandpass_a = butter(3, [1, 120], btype='bandpass', fs=fs)
            signal_filt = filtfilt(bandpass_b, bandpass_a, signal_filt, axis=0)

            # format resulting data into pandas DataFrame
            signal_filt = pd.DataFrame(signal_filt, columns=signal_nan.columns)
            signal_filt.index = pd.to_timedelta(t_sec_nan, unit="S")
            
            # re-reference the signals using laplacian referencing
            # signal_ref = _laplacian_reference(signal_filt)
            signal_ref = _common_average_reference(signal_filt)

            if all_signal is None:
                all_signal = signal_ref

            for name, band in tqdm(zip(band_names, bands), total=len(bands), desc='band', leave=False):
                # indices for 1 second non-overlapping windows
                win_size = 1 * fs
                ind_overlap = np.reshape(np.arange(signal_ref.shape[0]), (-1, int(win_size)))
                n_windows = np.size(ind_overlap, axis=0)

                # calculate bandpower in one second windows and set time array for this data pull
                t_sec_bandpower = np.zeros((n_windows))
                power_mat = np.zeros((n_windows, n_channels))
                for ind, win_inds in enumerate(ind_overlap):
                    window_data = signal_ref.iloc[win_inds, :]
                    power_mat[ind, :] = _coherence(window_data, fs, band)
                    t_sec_bandpower[ind] = t_sec_nan[win_inds[-1]]

                # append results to large arrays with all data pull
                if band_powers[name] is None:
                    band_powers[name] = power_mat
                else:
                    band_powers[name] = np.vstack((band_powers[name], power_mat))

            t_sec_arr.extend(t_sec_bandpower)
            sz_id_arr.extend([sz_id] * len(t_sec_bandpower))

    # set column names to contain both electrode and band
    col_names = [[ "{} {}".format(elec, band) for band in band_names for elec in data.columns ]]

    # format and save dataframe with all features by time
    all_band_powers = np.hstack(([band_powers[name] for name in band_names]))
    df = pd.DataFrame(all_band_powers, index=pd.to_timedelta(t_sec_arr, unit='S'), columns=col_names)         
    df['Seizure id'] = sz_id_arr

    df.to_pickle(ospj(pt_data_path, "coherence_elec-{}_period-{}.pkl".format(electrodes_opt, period)))
# %%
