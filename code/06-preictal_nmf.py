'''
This script plots spectrograms for pre-ictal periods. 
Then, it uses NMF to find subgraphs and expressions for pre-ictal periods.
Finally, it calculates states as the subgraph with maximal expression at each time point
and calculates the dissimilarity between states.
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
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join as ospj
from scipy.stats import zscore

sys.path.append('tools')

from plot_spectrogram import plot_spectrogram
from movmean import movmean
from pull_sz_starts import pull_sz_starts

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

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

# flags
PLOT = True
SAVE_PLOT = True
DTW_FLAG = False

FIXED_PREICTAL_SEC = 60 * 30
LEAD_SZ_WINDOW_SEC = (FIXED_PREICTAL_SEC + 60 * 15) # 15 min buffer

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
for index, row in tqdm(patient_cohort.iterrows()):
    pt = row["Patient"]

    # temporary while pre-ictal features are being calculated
    if pt == "HUP073":
        continue
    if pt == "HUP150":
        continue
    
    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)
    if not os.path.exists(pt_figure_path):
        os.makedirs(pt_figure_path)

    sz_starts = pull_sz_starts(pt, metadata)

    # get bandpower from pre-ictal period and log transform
    bandpower_mat_data = loadmat(ospj(pt_data_path, "bandpower-windows-12hr.mat"))
    bandpower_data = 10*np.log10(bandpower_mat_data['allFeats'])
    t_sec = np.squeeze(bandpower_mat_data['entireT']) / 1e6
    sz_id = np.squeeze(bandpower_mat_data['szID'])

    n_sz = np.size(np.unique(sz_id))
    
    # remove short inter-seizure intervals
    lead_sz = np.diff(np.insert(sz_starts, 0, [0])) > (FIXED_PREICTAL_SEC + 60 * 15) # 15 min buffer
    remaining_sz_ids = np.where(lead_sz)[0] + 1
    remove_sz_ids = np.where(~lead_sz)[0] + 1
    for remv in remove_sz_ids:
        t_sec = np.delete(t_sec, np.where(sz_id == remv))
        bandpower_data = np.delete(bandpower_data, np.where(sz_id == remv), axis=0)
        sz_id = np.delete(sz_id, np.where(sz_id == remv))

    
    # Plot spectrograms of pre-ictal period
    if PLOT:
        pi_starts = sz_starts[lead_sz] - FIXED_PREICTAL_SEC
        pi_ends = sz_starts[lead_sz]
        pi_start_ind = [time2ind(i, t_sec) for i in pi_starts]
        pi_end_ind = [time2ind(i, t_sec) for i in pi_ends]
        for i in range(len(pi_start_ind)):
            ax = plot_spectrogram(bandpower_data[pi_start_ind[i]:pi_end_ind[i], :], start_time=(pi_starts[i] - pi_ends[i]) / 60, end_time=0)
            ax.set_xlabel("Time from seizure onset (min)")
            ax.set_title("Seizure {}".format(remaining_sz_ids[i]))

            if SAVE_PLOT:
                plt.savefig(ospj(pt_figure_path, "pi_bandpower_spectrogram_sz_{}.svg".format(remaining_sz_ids[i])))
                plt.savefig(ospj(pt_figure_path, "pi_bandpower_spectrogram_sz_{}.png".format(remaining_sz_ids[i])))
                plt.close()

    # Apply NMF to pre-ictal period to find components (H) and expression (W)
    n_remaining_sz = np.size(remaining_sz_ids)
    n_components = 6
    if os.path.exists(ospj(pt_data_path, "nmf_expression.npy")):
        W = np.load(ospj(pt_data_path, "nmf_expression.npy"))
        H = np.load(ospj(pt_data_path, "nmf_coefficients.npy"))
    else:
        model = NMF(n_components=n_components, init='nndsvd', random_state=0, max_iter=1000)
        W = model.fit_transform(bandpower_data - np.min(bandpower_data))
        H = model.components_
        np.save(ospj(pt_data_path, "nmf_expression.npy"), W)
        np.save(ospj(pt_data_path, "nmf_coefficients.npy"), H)

    # States are defined as the max expressed component at each time point
    states = np.argmax(movmean(W[:, 1:].T, k=100).T, axis=-1) + 1

    # take the dissimilarity in states, optionally using fast dynamic time warping
    if DTW_FLAG:
        states_dissim_mat = np.zeros((n_remaining_sz, n_remaining_sz))
        for ind1, i in enumerate(remaining_sz_ids):
            for ind2, j in enumerate(remaining_sz_ids):
                distance, path = fastdtw(states[sz_id == i], states[sz_id == j], dist=euclidean)
                states_dissim_mat[ind1, ind2] = distance

    else:
        pre_ictal_lengths = np.zeros(remaining_sz_ids.shape, dtype=int)
        for ind, i_sz in enumerate(remaining_sz_ids):
            pre_ictal_lengths[ind] = np.size(states[sz_id == i_sz])
        pre_ictal_length = np.min(pre_ictal_lengths)

        states_dissim_mat = np.zeros((n_remaining_sz, n_remaining_sz))
        for ind1, i in enumerate(remaining_sz_ids):
            for ind2, j in enumerate(remaining_sz_ids):
                rand = adjusted_rand_score(states[sz_id == i][-pre_ictal_length:], states[sz_id == j][-pre_ictal_length:])
                states_dissim_mat[ind1, ind2] = 1 - rand

    np.save(ospj(pt_data_path, "states_dissim_mat.npy"), states_dissim_mat)
    np.save(ospj(pt_data_path, "remaining_sz_ids.npy"), remaining_sz_ids)

    # Plot the NMF subgraphs and expression
    if PLOT:
        for i in remaining_sz_ids:
            fig, ax = plt.subplots()
            ax.plot(t_sec[sz_id == i] - t_sec[sz_id == i][0], movmean(W[sz_id == i, 1:].T, k=100, mode='same').T)

            ax.set_xlabel("Time from seizure onset (min)")
            ax.set_ylabel("Subgraph coefficient")
            ax.set_title("Seizure {}".format(i))
            ax.legend(np.arange(n_components - 1) + 2, title="Component")

            if SAVE_PLOT:
                plt.savefig(ospj(pt_figure_path, "subgraph_expression_sz_{}.svg".format(i)), bbox_inches='tight', transparent='true')
                plt.savefig(ospj(pt_figure_path, "subgraph_expression_sz_{}.png".format(i)), bbox_inches='tight', transparent='true')
                plt.close()

        ax = plot_spectrogram(H, start_time=0, end_time=n_components)
        ax.set_title("{}".format(pt))
        ax.set_xlabel("Component")
        if SAVE_PLOT:
            plt.savefig(ospj(pt_figure_path, "subgraphs.svg"), bbox_inches='tight', transparent='true')
            plt.savefig(ospj(pt_figure_path, "subgraphs.png"), bbox_inches='tight', transparent='true')
            plt.close()



# %% Run once NMF has been saved

# for index, row in patient_cohort.iterrows():
#     pt = row["Patient"]    
#     pt_data_path = ospj(data_path, pt)
#     pt_figure_path = ospj(figure_path, pt)

#     bandpower_mat_data = loadmat(ospj(pt_data_path, "bandpower-windows-12hr.mat"))
#     t_sec = np.squeeze(bandpower_mat_data['entireT']) / 1e6
#     sz_id = np.squeeze(bandpower_mat_data['szID'])

#     W = np.load(ospj(pt_data_path, "nmf_expression.npy"))
#     H = np.load(ospj(pt_data_path, "nmf_coefficients.npy"))

#     n_components = W.shape[-1]
#     sz_starts = pull_sz_starts(pt, metadata)
#     # remove short inter-seizure intervals
#     lead_sz = np.diff(np.insert(sz_starts, 0, [0])) > (FIXED_PREICTAL_SEC + 60 * 15) # 15 min buffer
#     remaining_sz_ids = np.where(lead_sz)[0] + 1
#     remove_sz_ids = np.where(~lead_sz)[0] + 1
#     for remv in remove_sz_ids:
#         t_sec = np.delete(t_sec, np.where(sz_id == remv))
#         sz_id = np.delete(sz_id, np.where(sz_id == remv))

#     # States are defined as the max expressed component at each time point
#     states = np.argmax(movmean(W[:, 1:].T, k=100).T, axis=-1) + 1

#     # take the dissimilarity in states using fast dynamic time warping
#     # if DTW_FLAG:
#     n_remaining_sz = np.size(remaining_sz_ids)
#     pre_ictal_lengths = np.zeros(remaining_sz_ids.shape, dtype=int)
#     for ind, i_sz in enumerate(remaining_sz_ids):
#         pre_ictal_lengths[ind] = np.size(states[sz_id == i_sz])

#     pre_ictal_length = np.min(pre_ictal_lengths)

#     # state_corr_mat = np.zeros((n_remaining_sz, pre_ictal_length))
#     # for ind, i_sz in enumerate(remaining_sz_ids):
#     #     state_corr_mat[ind, :] = states[sz_id == i_sz][-pre_ictal_length:]

#     states_dissim_mat = np.zeros((n_remaining_sz, n_remaining_sz))
#     for ind1, i in enumerate(remaining_sz_ids):
#         for ind2, j in enumerate(remaining_sz_ids):
#             rand = adjusted_rand_score(states[sz_id == i][-pre_ictal_length:], states[sz_id == j][-pre_ictal_length:])
#             states_dissim_mat[ind1, ind2] = 1 - rand

#     np.save(ospj(pt_data_path, "states_dissim_mat.npy"), states_dissim_mat)

#     fig, ax = plt.subplots()
#     ax.imshow(states_dissim_mat)
#     ax.set_title(pt)
#     continue

    # states_dissim_mat = np.zeros((n_remaining_sz, n_remaining_sz))
    # for ind1, i in enumerate(remaining_sz_ids):
    #     for ind2, j in enumerate(remaining_sz_ids):
    #         distance, path = fastdtw(states[sz_id == i], states[sz_id == j], dist=euclidean)
    #         states_dissim_mat[ind1, ind2] = distance
    # continue
    # if PLOT:
    #     ax = plot_spectrogram(H, start_time=0, end_time=n_components)
    #     ax.set_title("{}".format(pt))
    #     ax.set_xlabel("Component")
    #     ax.set_xticks(np.arange(n_components) + 0.5)
    #     ax.set_xticklabels(np.arange(1, n_components + 1))

    #     if SAVE_PLOT:
    #         plt.savefig(ospj(pt_figure_path, "subgraphs.svg"), bbox_inches='tight', transparent='true')
    #         plt.savefig(ospj(pt_figure_path, "subgraphs.png"), bbox_inches='tight', transparent='true')
    #         plt.close()

    # for i in remaining_sz_ids:
    #     fig, ax = plt.subplots()
    #     xaxis = (t_sec[sz_id == i] - t_sec[sz_id == i][-1]) / 60
    #     ax.plot(xaxis, movmean(W[sz_id == i, 1:].T, k=100).T)

    #     ax.set_xlabel("Time from seizure onset (min)")
    #     ax.set_ylabel("Subgraph coefficient")
    #     ax.set_title("Seizure {}".format(i))
    #     ax.legend(np.arange(n_components - 1) + 2, title="Component")

    #     if SAVE_PLOT:
    #         plt.savefig(ospj(pt_figure_path, "subgraph_expression_sz_{}.svg".format(i)), bbox_inches='tight', transparent='true')
    #         plt.savefig(ospj(pt_figure_path, "subgraph_expression_sz_{}.png".format(i)), bbox_inches='tight', transparent='true')
    #         plt.close()


# %%
