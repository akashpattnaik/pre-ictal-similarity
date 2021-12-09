'''
This script plots spectrograms for pre-ictal periods. 
Then, it uses NMF to find subgraphs and expressions for pre-ictal periods.
Finally, it calculates states as the subgraph with maximal expression at each time point
and calculates the dissimilarity between states.

Inputs:
target-electrodes-{mode}.mat
bandpower-windows-pre-sz-{mode}.mat

Outputs:

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
electrodes_opt = config['electrodes']
band_opt = config['bands']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

# flags
SAVE_PLOT = True

NMF_OPT_FLAG = True
FIXED_PREICTAL_SEC = 60 * 30
LEAD_SZ_WINDOW_SEC = (FIXED_PREICTAL_SEC + 60 * 15) # 15 min buffer

# %%
patient_localization_mat = loadmat(ospj(metadata_path, 'patient_localization_final.mat'))['patient_localization']
patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(ospj(metadata_path, 'patient_localization_final.mat'))

for index, row in patient_cohort.iterrows():
    if row['Ignore']:
        continue

    pt = row["Patient"]
    print("Calculating pre-ictal NMF for {}".format(pt))

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)
    if not os.path.exists(pt_figure_path):
        os.makedirs(pt_figure_path)

    # pull and format electrode metadata
    electrodes_mat = loadmat(ospj(pt_data_path, "selected_electrodes_elec-{}.mat".format(electrodes_opt)))
    target_electrode_region_inds = electrodes_mat['targetElectrodesRegionInds'][0]
    pt_index = patients.index(pt)
    sz_starts = pull_sz_starts(pt, metadata)

    # get bandpower from pre-ictal period and log transform
    # bandpower_mat_data = loadmat(ospj(pt_data_path, "bandpower-windows-pre-sz-{}.mat".format(electrodes_opt)))
    # bandpower_data = 10*np.log10(bandpower_mat_data['allFeats'])
    # t_sec = np.squeeze(bandpower_mat_data['entireT']) / 1e6
    # sz_id = np.squeeze(bandpower_mat_data['szID'])

    df = pd.read_pickle(ospj(pt_data_path, "bandpower_elec-{}_period-preictal.pkl".format(electrodes_opt)))

    if band_opt == "all":
        bandpower_data = df.filter(regex=("^((?!broad).)*$"), axis=1)
        bandpower_data  = bandpower_data.drop(['Seizure id'], axis=1)
    elif band_opt == "broad":
        bandpower_data = df.filter(regex=("broad"), axis=1)
    else:
        print("Band configuration not given properly")
        exit
    sz_id = np.squeeze(df['Seizure id'])
    t_sec = np.array(df.index / np.timedelta64(1, 's'))
    n_sz = np.size(np.unique(sz_id))
    
    # remove short inter-seizure intervals
    lead_sz = np.diff(np.insert(sz_starts, 0, [0])) > (FIXED_PREICTAL_SEC + 60 * 15) # 15 min buffer
    remaining_sz_ids = np.where(lead_sz)[0]
    remove_sz_ids = np.where(~lead_sz)[0]

    print("\tremoving seizures {}".format(remove_sz_ids))
    print(type(sz_id))
    for remv in remove_sz_ids:
        t_sec = np.delete(t_sec, np.where(sz_id == remv))
        bandpower_data.drop(bandpower_data.index[np.where(sz_id == remv)[0]], inplace=True)
        sz_id.drop(sz_id.index[np.where(sz_id == remv)[0]], inplace=True)
    np.save(ospj(pt_data_path, "remaining_sz_ids.npy"), remaining_sz_ids)

    # Apply NMF to pre-ictal period to find components (H) and expression (W)
    n_remaining_sz = np.size(remaining_sz_ids)
    n_components = range(2, 20)

    if NMF_OPT_FLAG:
        print("\trunning NMF")
        start_time = time.time()
        reconstruction_err = np.zeros(np.size(n_components))
        for ind, i_components in enumerate(n_components):
            print("\t\tTesting NMF with {} components".format(i_components))
            model = NMF(n_components=i_components, init='nndsvd', random_state=0, max_iter=1000)
            W = model.fit_transform(bandpower_data - np.min(bandpower_data))
            reconstruction_err[ind] = model.reconstruction_err_
        end_time = time.time()
        print("\tNMF took {} seconds".format(end_time - start_time))

        kneedle = KneeLocator(n_components, reconstruction_err, curve="convex", direction="decreasing")
        n_opt_components = kneedle.knee

        print("\t{} components was found as optimal, rerunning for final iteration".format(n_opt_components))
    else:
        print("\tusing predefined number of components for NMF")
        n_opt_components = 6

    model = NMF(n_components=n_opt_components, init='nndsvd', random_state=0, max_iter=1000)
    W = model.fit_transform(bandpower_data - np.min(bandpower_data))
    H = model.components_

    np.save(ospj(pt_data_path, "nmf_expression_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)), W)
    np.save(ospj(pt_data_path, "nmf_coefficients_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)), H)
    np.save(ospj(pt_data_path, "lead_sz_t_sec_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)), t_sec)
    np.save(ospj(pt_data_path, "lead_sz_sz_id_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)), sz_id)

##############################################
    # %%

#     # States are defined as the max expressed component at each time point
#     states = np.argmax(movmean(W[:, 1:].T, k=100).T, axis=-1) + 1

#     # take the dissimilarity in states, optionally using fast dynamic time warping
#     if DTW_FLAG:
#         states_dissim_mat = np.zeros((n_remaining_sz, n_remaining_sz))
#         for ind1, i in enumerate(remaining_sz_ids):
#             for ind2, j in enumerate(remaining_sz_ids):
#                 distance, path = fastdtw(states[sz_id == i], states[sz_id == j], dist=euclidean)
#                 states_dissim_mat[ind1, ind2] = distance
#     else:
#         # find how long pre-ictal segments are for each sz and take shortest one
#         pre_ictal_lengths = np.zeros(remaining_sz_ids.shape, dtype=int)
#         for ind, i_sz in enumerate(remaining_sz_ids):
#             pre_ictal_lengths[ind] = np.size(states[sz_id == i_sz])
#         pre_ictal_length = np.min(pre_ictal_lengths)

#         # matrix of adjusted rand score for similar state occurences
#         states_dissim_mat = np.zeros((n_remaining_sz, n_remaining_sz))
#         for ind1, i in enumerate(remaining_sz_ids):
#             for ind2, j in enumerate(remaining_sz_ids):
#                 rand = adjusted_rand_score(states[sz_id == i][-pre_ictal_length:], states[sz_id == j][-pre_ictal_length:])
#                 states_dissim_mat[ind1, ind2] = 1 - rand

#     np.save(ospj(pt_data_path, "states_dissim_mat_{}.npy".format(mode)), states_dissim_mat)
#     np.save(ospj(pt_data_path, "remaining_sz_ids.npy"), remaining_sz_ids)

#     # Plot the NMF subgraphs and expression
#     if PLOT:
#         for i in remaining_sz_ids:
#             fig, ax = plt.subplots()
#             t_arr_min = (t_sec[sz_id == i] - t_sec[sz_id == i][-1]) / 60

#             ax.plot(t_arr_min, movmean(W[sz_id == i, 1:].T, k=100, mode='same').T)

#             ax.set_xlabel("Time from seizure onset (min)")
#             ax.set_ylabel("Subgraph coefficient")
#             ax.set_title("Seizure {}".format(i))
#             ax.legend(np.arange(n_components - 1) + 2, title="Component")

#             if SAVE_PLOT:
#                 plt.savefig(ospj(pt_figure_path, "subgraph_expression_sz_{}_{}.svg".format(i, mode)), bbox_inches='tight', transparent='true')
#                 plt.savefig(ospj(pt_figure_path, "subgraph_expression_sz_{}_{}.png".format(i, mode)), bbox_inches='tight', transparent='true')
#                 plt.close()

#         ax = plot_spectrogram(H, start_time=0, end_time=n_components)
#         ax.set_title("{}".format(pt))
#         ax.set_xlabel("Component")
#         if SAVE_PLOT:
#             plt.savefig(ospj(pt_figure_path, "subgraphs_{}.svg".format(mode)), bbox_inches='tight', transparent='true')
#             plt.savefig(ospj(pt_figure_path, "subgraphs_{}.png".format(mode)), bbox_inches='tight', transparent='true')
#             plt.close()


#     if PLOT:
#         n_electrodes = soz_electrodes.shape[0] 

#         # plot all states
#         component_arr = np.reshape(H, (n_components, -1, n_electrodes))
#         # component_z = np.zeros(component_arr.shape)
#         # for i_comp in range(n_components):
#         #     component_z[i_comp, :, :] = zscore(component_arr[i_comp, :, :], axis=1)

#         # sort to put non-soz first
#         sort_soz_inds = np.argsort(soz_electrodes)
#         n_soz = np.sum(soz_electrodes)
#         n_non_soz = n_electrodes - n_soz

#         for i_comp in range(n_components):
#             fig, ax = plt.subplots()
#             divider = make_axes_locatable(ax)
#             cax = divider.append_axes('right', size='5%', pad=0.05)

#             im = ax.imshow(component_arr[i_comp, :, sort_soz_inds].T)

#             ax.axvline(n_non_soz - 0.5, c='r', lw=2)
#             ax.set_title("Subgraph {}, {}".format(i_comp, pt))
#             ax.set_yticks(np.arange(6))
#             ax.set_yticklabels([r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'low-$\gamma$', r'high-$\gamma$'])

#             ax.set_xticks(np.arange(n_electrodes))
#             ax.set_xticks([n_non_soz / 2, n_non_soz + n_soz / 2])
#             ax.set_xticklabels(["Non SOZ", "SOZ"])
            
#             ax.set_xlabel("Electrodes")
#             ax.set_ylabel("Frequency band")

#             cbar = fig.colorbar(im, cax=cax, orientation='vertical')
#             cbar.ax.set_ylabel('Power (dB)', rotation=90)

#             if SAVE_PLOT:
#                 plt.savefig(ospj(pt_figure_path, "soz_subgraph_{}_heatmap_{}.svg".format(i_comp, mode)), bbox_inches='tight', transparent='true')
#                 plt.savefig(ospj(pt_figure_path, "soz_subgraph_{}_heatmap_{}.png".format(i_comp, mode)), bbox_inches='tight', transparent='true')
#                 plt.close()

#         # plot soz state expression for all seizures
#         for i in remaining_sz_ids:
#             fig, ax = plt.subplots()
#             t_arr_min = (t_sec[sz_id == i] - t_sec[sz_id == i][-1]) / 60
#             ax.plot(t_arr_min, movmean(W[sz_id == i,pt_soz_state].T, k=100).T)
#             ax.set_xlabel("Time from seizure onset (min)")
#             ax.set_ylabel("SOZ subgraph coefficient")
#             ax.set_title("Seizure {}".format(i))

#             if SAVE_PLOT:
#                 plt.savefig(ospj(pt_figure_path, "soz_expression_sz_{}_{}.svg".format(i, mode)), bbox_inches='tight', transparent='true')
#                 plt.savefig(ospj(pt_figure_path, "soz_expression_sz_{}_{}.png".format(i, mode)), bbox_inches='tight', transparent='true')
#                 plt.close()
    
#     break
# # %%
# min_pre_ictal_size = min([W[sz_id == i,pt_soz_state].shape[0] for i in remaining_sz_ids])

# pre_ictal_soz_state = np.zeros((np.size(remaining_sz_ids), min_pre_ictal_size))

# for ind, i_sz in enumerate(remaining_sz_ids):
#     pre_ictal_soz_state[ind, :] = W[sz_id == i_sz,pt_soz_state][-min_pre_ictal_size:]

# # %%

# # %%
