# %%
# Imports and environment setup
import numpy as np
import sys
import os
import pandas as pd
import json
import matplotlib.pyplot as plt

from os.path import join as ospj

sys.path.append('tools')

from pull_sz_starts import pull_sz_starts

from scipy.stats import pearsonr
import seaborn as sns
# %%
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

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

PLOT = False
# %%
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]

    print("Comparing dissimilarity matrices for {}".format(pt))

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    sz_dissim_mat = np.load(ospj(pt_data_path, "sz_dissim_mat_dtw_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    time_dissim_mat = np.load(ospj(pt_data_path, "time_dissim_mat.npy"))
    circadian_dissim_mat = np.load(ospj(pt_data_path, "circadian_dissim_mat.npy"))

    assert sz_dissim_mat.shape[0] == time_dissim_mat.shape[0], "{}: sz {}, time {}".format(pt, sz_dissim_mat.shape[0], time_dissim_mat.shape[0])
    n_sz = sz_dissim_mat.shape[0]

    if n_sz <= 2:
        continue
    tri_inds = np.triu_indices(n_sz, k=1)

    sz_dissim = sz_dissim_mat[tri_inds]
    time_dissim = time_dissim_mat[tri_inds]
    circadian_dissim = circadian_dissim_mat[tri_inds]

    corr, sig = pearsonr(sz_dissim, time_dissim)
    patient_cohort.at[index, 'Seizure-Time Correlation'] = corr

    corr, sig = pearsonr(sz_dissim, circadian_dissim)
    patient_cohort.at[index, 'Seizure-Circadian Correlation'] = corr

    # if os.path.exists(ospj(pt_data_path, "pi_dissim_mats_dtw.npy")):
    #     pi_dissim_mats_dtw = np.load(ospj(pt_data_path, "pi_dissim_mats_dtw.npy"))
    #     n_components = 6

    #     for i_comp in range(n_components):
    #         pi_dtw_dissim = pi_dissim_mats_dtw[i_comp, :, :][tri_inds]
    #         corr, sig = pearsonr(sz_dissim, pi_dtw_dissim)
    #         patient_cohort.at[index, 'Seizure-Preictal DTW Correlation (Subgraph {})'.format(i_comp)] = corr

    if os.path.exists(ospj(pt_data_path, "states_dissim_mat.npy")):
        states_dissim_mat = np.load(ospj(pt_data_path, "states_dissim_mat.npy"))
        remaining_sz_ids = np.load(ospj(pt_data_path, "remaining_sz_ids.npy"))

        if np.sum(states_dissim_mat) == 0:
            continue
        n_remaining_inds = np.size(remaining_sz_ids)
        # get upper triangular indices of square matrices where n is the number of lead seizures
        remaining_tri_inds = np.triu_indices(n_remaining_inds, k=1)
        remaining_sz_dissim_mat = sz_dissim_mat[remaining_sz_ids[:, None] - 1, remaining_sz_ids - 1]

        states_dissim = states_dissim_mat[remaining_tri_inds]
        remaining_sz_dissim = remaining_sz_dissim_mat[remaining_tri_inds]
        
        corr, sig = pearsonr(remaining_sz_dissim, states_dissim)
        patient_cohort.at[index, 'Seizure-States Correlation'] = corr


    if os.path.exists(ospj(pt_data_path, "soz_subgraph_dissim_mat_band-{}_elec-{}.npy".format(band_opt, electrodes_opt))):
        soz_subgraph_dissim_mat = np.load(ospj(pt_data_path, "soz_subgraph_dissim_mat_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
        remaining_sz_ids = np.load(ospj(pt_data_path, "remaining_sz_ids.npy"))

        n_remaining_inds = np.size(remaining_sz_ids)
        # get upper triangular indices of square matrices where n is the number of lead seizures
        remaining_tri_inds = np.triu_indices(n_remaining_inds, k=1)
        remaining_sz_dissim_mat = sz_dissim_mat[remaining_sz_ids[:, None] - 1, remaining_sz_ids - 1]

        soz_subgraph_dissim = soz_subgraph_dissim_mat[remaining_tri_inds]
        remaining_sz_dissim = remaining_sz_dissim_mat[remaining_tri_inds]

        if np.size(remaining_sz_dissim) < 2:
            patient_cohort.at[index, 'Seizure-SOZ Subgraph Correlation'] = None
            continue
        
        corr, sig = pearsonr(remaining_sz_dissim, soz_subgraph_dissim)
        patient_cohort.at[index, 'Seizure-SOZ Subgraph Correlation'] = corr


patient_cohort.to_csv(ospj(data_path, "patient_cohort_with_corr.csv"))

# %%
ax = sns.barplot(x='Patient', y='Seizure-States Correlation', data=patient_cohort)
ax.set_xticklabels(patient_cohort["Patient"], rotation=90)

if PLOT:
    plt.savefig(ospj(figure_path, "sz_state_corr.svg"), transparent=True, bbox_inches='tight')
    plt.savefig(ospj(figure_path, "sz_state_corr.png"), transparent=True, bbox_inches='tight')
    plt.close()
# %%
ax = sns.barplot(x='Patient', y='Seizure-Time Correlation', data=patient_cohort)
ax.set_xticklabels(patient_cohort["Patient"], rotation=90)

if PLOT:
    plt.savefig(ospj(figure_path, "sz_time.svg"), transparent=True, bbox_inches='tight')
    plt.savefig(ospj(figure_path, "sz_time.png"), transparent=True, bbox_inches='tight')
    plt.close()

# %%
ax = sns.barplot(x='Patient', y='Seizure-Circadian Correlation', data=patient_cohort)
ax.set_xticklabels(patient_cohort["Patient"], rotation=90)

if PLOT:
    plt.savefig(ospj(figure_path, "sz_circadian.svg"), transparent=True, bbox_inches='tight')
    plt.savefig(ospj(figure_path, "sz_circadian.png"), transparent=True, bbox_inches='tight')
    plt.close()

# %% Why does HUP 144 have the same correlation coefficient?

# pt = "HUP144"
# pt_data_path = ospj("../data", pt)

# sz_dissim_mat = np.load(ospj(pt_data_path, "sz_dissim_mat.npy"))
# time_dissim_mat = np.load(ospj(pt_data_path, "time_dissim_mat.npy"))
# circadian_dissim_mat = np.load(ospj(pt_data_path, "circadian_dissim_mat.npy"))

# fig, ax = plt.subplots()
# ax.imshow(sz_dissim_mat)

# fig, ax = plt.subplots()
# ax.imshow(time_dissim_mat)

# fig, ax = plt.subplots()
# ax.imshow(circadian_dissim_mat)

# sz_starts = pull_sz_starts(pt, metadata)
# # It's because all of the seizures happened on the same day -- time dissim mat is the same is circadian dissim mat

# %%

# sig_pre_ictal = ["HUP070", "HUP111", "HUP187"]
# for pt in sig_pre_ictal:
#     pt_data_path = ospj("../data", pt)

#     sz_dissim_mat = np.load(ospj(pt_data_path, "sz_dissim_mat.npy"))
#     time_dissim_mat = np.load(ospj(pt_data_path, "time_dissim_mat.npy"))
#     circadian_dissim_mat = np.load(ospj(pt_data_path, "circadian_dissim_mat.npy"))
#     pi_dissim_mat = np.load(ospj(pt_data_path, "pi_dissim_mat_60_sec.npy"))

#     fig, axes = plt.subplots(nrows=2, ncols=2)

#     axes.flat[0].imshow(sz_dissim_mat)
#     axes.flat[1].imshow(pi_dissim_mat)
#     axes.flat[2].imshow(time_dissim_mat)
#     axes.flat[3].imshow(circadian_dissim_mat)

#     axes.flat[0].set_title("Seizures")
#     axes.flat[1].set_title("Pre-ictal")
#     axes.flat[2].set_title("Time")
#     axes.flat[3].set_title("Circadian")

# %%
