# %%
# %load_ext autoreload
# %autoreload 2
# Imports and environment setup
import numpy as np
import sys
import os
import pandas as pd
import json
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join as ospj
from scipy.stats import zscore
import time
from kneed import KneeLocator

sys.path.append('tools')

from plot_spectrogram import plot_spectrogram
from movmean import movmean
from pull_sz_starts import pull_sz_starts
from pull_patient_localization import pull_patient_localization
from mpl_toolkits.axes_grid1 import make_axes_locatable

from time2ind import time2ind

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.decomposition import NMF
from sklearn.metrics.cluster import adjusted_rand_score

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
metadata_path = config['metadataPath']
palette = config['lightColors']
DTW_FLAG = config['flags']["DTW_FLAG"]
electrodes = config['electrodes']
bands = config['bands']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

# flags
SAVE_PLOT = True

NMF_FLAG = True
FIXED_PREICTAL_SEC = 60 * 30
LEAD_SZ_WINDOW_SEC = (FIXED_PREICTAL_SEC + 60 * 15) # 15 min buffer

def soz_state(H, soz_electrodes, metric="max_all", is_zscore=False):
    '''
    soz_mask: soz electrodes are true and non_soz electrodes are false

    metric: determines how to find soz state. max_all takes the state where soz 
    channels have higher bandpower in all frequency bands

    '''
    n_components = H.shape[0]
    n_electrodes = soz_electrodes.shape[0] 

    # reshape to (component, frequency band, electrode)
    component_arr = np.reshape(H, (n_components, -1, n_electrodes))
    if is_zscore:
        component_z = np.zeros(component_arr.shape)
        for i_comp in range(n_components):
            component_z[i_comp, :, :] = zscore(component_arr[i_comp, :, :], axis=1)
        component_arr = component_z
    # sort to put non-soz first
    sort_soz_inds = np.argsort(soz_electrodes)
    n_soz = np.sum(soz_electrodes)
    n_non_soz = n_electrodes - n_soz


    n_iter = 10000
    null_z = np.zeros(n_components)

    for i_comp in range(n_components):
        # randomly resample electrodes and take the mean bandpower of sample
        means = np.zeros(n_iter)
        for iter in range(n_iter):
            means[iter] = np.mean(component_arr[i_comp, :, np.random.choice(n_electrodes, n_soz)])
        # append true soz
        means = np.append(means, np.mean(component_arr[i_comp, :, soz_electrodes]))
        # calculate z_score of true soz and save
        null_z[i_comp] = zscore(means)[-1]

    return np.argmax(np.abs(null_z))

patient_localization_mat = loadmat(ospj(metadata_path, 'patient_localization_final.mat'))['patient_localization']
patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(ospj(metadata_path, 'patient_localization_final.mat'))

# %%
# Plot the NMF subgraphs and expression
for index, row in patient_cohort.iterrows():
    if row['Ignore']:
        continue

    pt = row["Patient"]
    print("Calculating for {}".format(pt))

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    remaining_sz_ids = np.load(ospj(pt_data_path, "remaining_sz_ids.npy"))
    t_sec = np.load(ospj(pt_data_path, "lead_sz_t_sec_band-{}_elec-{}.npy".format(bands, electrodes)))
    sz_id = np.load(ospj(pt_data_path, "lead_sz_sz_id_band-{}_elec-{}.npy".format(bands, electrodes)))
    W = np.load(ospj(pt_data_path, "nmf_expression_band-{}_elec-{}.npy".format(bands, electrodes)))
    H = np.load(ospj(pt_data_path, "nmf_coefficients_band-{}_elec-{}.npy".format(bands, electrodes)))
    n_components = H.shape[0]

    
    # pull and format electrode metadata
    electrodes_mat = loadmat(ospj(pt_data_path, "selected_electrodes_elec-{}.mat".format(electrodes)))
    target_electrode_region_inds = electrodes_mat['targetElectrodesRegionInds'][0]
    pt_index = patients.index(pt)
    sz_starts = pull_sz_starts(pt, metadata)


    # find seizure onset zone and state with most seizure onset zone
    soz_electrodes = np.array(np.squeeze(soz[pt_index][target_electrode_region_inds, :]), dtype=bool)
    pt_soz_state = soz_state(H, soz_electrodes)
    
    patient_cohort.at[index, 'SOZ Sensitive State'] = pt_soz_state

    np.save(ospj(pt_data_path, "soz_electrodes_band-{}_elec-{}.npy".format(bands, electrodes)), soz_electrodes)
    np.save(ospj(pt_data_path, "pt_soz_state_band-{}_elec-{}.npy".format(bands, electrodes)), pt_soz_state)

patient_cohort.to_excel(ospj(data_path, "patient_cohort_test.xlsx"))
# %%
