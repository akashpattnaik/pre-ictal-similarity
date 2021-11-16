'''
This script uses dynamic time warping and euclidean distance to crease dissimilarity matrices between pairs of seizures.
In addition, this script creates temporal and circadian dissimilarity matrices.
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
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import time
from os.path import join as ospj

sys.path.append('tools')
from pull_sz_starts import pull_sz_starts

# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
metadata_path = config['metadataPath']
DTW_FLAG = config['flags']["DTW_FLAG"]
mode = config['mode']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

# Make patient directories
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    if not os.path.exists(pt_data_path):
        os.makedirs(pt_data_path)
    if not os.path.exists(pt_figure_path):
        os.makedirs(pt_figure_path)

# %% Calcualate seizure dissimilarities
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]

    pt = "HUP130"
    print("Calculating dissimilarity matrices for {}".format(pt))

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    bandpower_mat_data = loadmat(ospj(pt_data_path, "bandpower-windows-sz-{}.mat".format(mode)))
    bandpower_data = 10*np.log10(bandpower_mat_data['allFeats'])
    t_sec = np.squeeze(bandpower_mat_data['entireT']) / 1e6
    sz_id = np.squeeze(bandpower_mat_data['szID'])

    n_sz = np.size(np.unique(sz_id))
    # Seizure dissimilarity
    sz_dissim_mat = np.zeros((n_sz, n_sz))

    # apply dynamic time warping to each seizure
    if DTW_FLAG:
        start_time = time.time()
        for i_sz in range(1, n_sz + 1):
            for j_sz in range(1, n_sz + 1):
                if i_sz != j_sz:
                    distance, path = fastdtw(bandpower_data[sz_id == i_sz, :], bandpower_data[sz_id == j_sz, :], dist=euclidean)
                    sz_dissim_mat[i_sz - 1, j_sz - 1] = distance
        np.save(ospj(pt_data_path, "sz_dissim_mat_dtw_{}.npy".format(mode)), sz_dissim_mat)
        end_time = time.time()
        print("\tDynamic time warping took {} seconds".format(end_time - start_time))

    else:
        start_time = time.time()
        # find number of windows for each seizure
        sz_breaks = np.array([np.size(sz_id[sz_id == i]) for i in range(1, np.max(sz_id) + 1)])
        # take min number of windows to compare seizures
        n_sz_windows = np.min(sz_breaks)

        # collect bandpowers to perform correlation
        corr_mat = np.zeros((n_sz, n_sz_windows*bandpower_data.shape[-1]))
        for i_sz in range(1, n_sz + 1):
            corr_mat[i_sz - 1, :] = np.ravel(bandpower_data[sz_id == i_sz, :][:n_sz_windows, :])
        sz_dissim_mat = 1 - np.corrcoef(corr_mat)
        np.save(ospj(pt_data_path, "sz_dissim_mat_{}.npy".format(mode)), sz_dissim_mat)
        end_time = time.time()
        print("\tCorrelating beginning of seizure clips took {} seconds".format(end_time - start_time))

    sz_starts = pull_sz_starts(pt, metadata)
    # time dissimilarity
    print("\tCalculating time dissimilarity matrix")
    time_dissim_mat = np.abs(sz_starts[:, None] - sz_starts[None, :]) / 60 / 60
    np.save(ospj(pt_data_path, "time_dissim_mat.npy"), time_dissim_mat)

    # circadian dissimilarity
    print("\tCalculating circadian dissimilarity matrix")
    circadian_dissim_mat = np.abs(sz_starts[:, None] % (60 * 60 * 24) - sz_starts[None, :] % (60 * 60 * 24)) / 60 / 60
    np.save(ospj(pt_data_path, "circadian_dissim_mat.npy"), circadian_dissim_mat)

    print("\tDissimilarity matrices calculated for {}".format(pt))

    break