'''
Find subgraphs with seizure onset zone

'''
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
from scipy.stats import zscore
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.ticker as ticker

from tqdm import tqdm
from os.path import join as ospj

sys.path.append('tools')

from pull_sz_starts import pull_sz_starts
from pull_patient_localization import pull_patient_localization
from time2ind import time2ind
from plot_spectrogram import plot_spectrogram
from movmean import movmean

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

# flags
PLOT = True
SAVE_PLOT = True
FIXED_PREICTAL_SEC = 60 * 30


# %% define soz state
def soz_state(H, soz_electrodes, metric="max_all"):
    '''
    soz_mask: soz electrodes are true and non_soz electrodes are false

    metric: determines how to find soz state. max_all takes the state where soz 
    channels have higher bandpower in all frequency bands

    '''
    n_components = H.shape[0]
    n_electrodes = soz_electrodes.shape[0] 

    # reshape to (component, frequency band, electrode)
    component_arr = np.reshape(H, (n_components, -1, n_electrodes))
    component_z = np.zeros(component_arr.shape)
    for i_comp in range(n_components):
        component_z[i_comp, :, :] = zscore(component_arr[i_comp, :, :], axis=1)

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
            means[iter] = np.mean(component_z[i_comp, :, np.random.choice(n_electrodes, n_soz)])
        # append true soz
        means = np.append(means, np.mean(component_z[i_comp, :, soz_electrodes]))
        # calculate z_score of true soz and save
        null_z[i_comp] = zscore(means)[-1]

    return np.argmax(np.abs(null_z))

# %%
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]
    print(pt)
    # temporary while pre-ictal features are being calculated
    if pt == "HUP073":
        continue

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    # pull and format electrode metadata
    electrodes_mat = loadmat(ospj(pt_data_path, 'target-electrodes-regions.mat'))
    target_electrode_region_inds = electrodes_mat['targetElectrodesRegionInds'][0] - 1
    patient_localization_mat = loadmat(ospj(metadata_path, 'patient_localization_final.mat'))['patient_localization']
    patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(ospj(metadata_path, 'patient_localization_final.mat'))

    pt_index = patients.index(pt)
    # get soz electrodes
    soz_electrodes = np.array(np.squeeze(soz[pt_index][target_electrode_region_inds, :]), dtype=bool)

    n_electrodes = np.size(target_electrode_region_inds)
    # # get bandpower in soz electrodes
    H = np.load(ospj(pt_data_path, "nmf_coefficients.npy"))
    n_components = H.shape[0]

    pt_soz_state = soz_state(H, soz_electrodes)
    patient_cohort.at[index, 'SOZ Sensitive State (0-index)'] = pt_soz_state

    # reshape to (component, frequency band, electrode)
    component_arr = np.reshape(H, (n_components, -1, n_electrodes))
    component_z = np.zeros(component_arr.shape)
    for i_comp in range(n_components):
        component_z[i_comp, :, :] = zscore(component_arr[i_comp, :, :], axis=1)
    # %% Plot bandpower and highlight soz channels
    sort_soz_inds = np.argsort(soz_electrodes)
    
    n_soz = np.sum(soz_electrodes)
    n_non_soz = n_electrodes - n_soz
    # for i_comp in range(n_components):
    fig, ax = plt.subplots()
    im = ax.imshow(component_z[pt_soz_state, :, sort_soz_inds].T)

    ax.axvline(n_non_soz - 0.5, c='r', lw=2)
    ax.set_title("Subgraph {}, {}".format(pt_soz_state, pt))
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels([r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'low-$\gamma$', r'high-$\gamma$'])

    ax.set_xticks(np.arange(n_electrodes))
    ax.set_xlabel("Electrode Number")
    ax.set_ylabel("Frequency band")

    ax_divider = make_axes_locatable(ax)
    # Add an axes to the right of the main axes.
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(im, cax=cax)

    if SAVE_PLOT:
        plt.savefig(ospj(pt_figure_path, "soz_subgraph_{}_heatmap.svg".format(pt_soz_state)))
        plt.savefig(ospj(pt_figure_path, "soz_subgraph_{}_heatmap.png".format(pt_soz_state)))
        plt.close()

patient_cohort.to_csv(ospj(data_path, "patient_cohort_with_soz_states.csv"))

# %% Analysis
patient_cohort = pd.read_csv(ospj(data_path, "patient_cohort_with_soz_states.csv"))
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    if pt == "HUP073":
        continue

    soz_state = int(row['SOZ Sensitive State (0-index)'])

    # # get bandpower in soz electrodes
    W = np.load(ospj(pt_data_path, "nmf_expression.npy"))

    sz_starts = pull_sz_starts(pt, metadata)

    # remove short inter-seizure intervals
    lead_sz = np.diff(np.insert(sz_starts, 0, [0])) > (FIXED_PREICTAL_SEC + 60 * 15) # 15 min buffer
    remaining_sz_ids = np.where(lead_sz)[0] + 1
    remove_sz_ids = np.where(~lead_sz)[0] + 1
    for remv in remove_sz_ids:
        t_sec = np.delete(t_sec, np.where(sz_id == remv))
        bandpower_data = np.delete(bandpower_data, np.where(sz_id == remv), axis=0)
        sz_id = np.delete(sz_id, np.where(sz_id == remv))

    # get bandpower from pre-ictal period and log transform
    bandpower_mat_data = loadmat(ospj(pt_data_path, "bandpower-windows-12hr.mat"))
    bandpower_data = 10*np.log10(bandpower_mat_data['allFeats'])
    t_sec = np.squeeze(bandpower_mat_data['entireT']) / 1e6
    sz_id = np.squeeze(bandpower_mat_data['szID'])

    # for i_sz in remaining_sz_ids:
    #     fig, ax = plt.subplots()
    #     ax.plot(movmean(W[sz_id==i_sz, soz_state], k=100))
    #     ax.set_title("Seizure {}, {}".format(i_sz, pt))

# %%
