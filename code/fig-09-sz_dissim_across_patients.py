# %%
from datetime import time
import numpy as np
import os
import sys
from os.path import join as ospj
import matplotlib.pyplot as plt
sys.path.append('tools')
from pull_sz_starts import pull_sz_starts
import pandas as pd
import json
from scipy.stats import pearsonr


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

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort_test.xlsx"))

metadata_path = "../../ieeg-metadata/"
metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

# %%
xcoords = []
ycoords = []
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]

    print(pt)
    pt_data_path = ospj("../data", pt)

    if DTW_FLAG:
        sz_dissim_mat = np.load(ospj(pt_data_path, "sz_dissim_mat_dtw_{}_{}.npy".format(electrodes_opt, band_opt)))
    else:
        sz_dissim_mat = np.load(ospj(pt_data_path, "sz_dissim_mat_{}_{}.npy".format(electrodes_opt, band_opt)))
    time_dissim_mat = np.load(ospj(pt_data_path, "time_dissim_mat.npy"))
    circadian_dissim_mat = np.load(ospj(pt_data_path, "circadian_dissim_mat.npy"))
    pi_dissim_mats = {}

    print("sz", sz_dissim_mat.shape)

    tri_inds = np.triu_indices(sz_dissim_mat.shape[0], k=1)
    xcoords.extend(sz_dissim_mat[tri_inds])
    ycoords.extend(np.repeat(index, np.size(tri_inds[0])))

    if pt == "HUP097":
        fig, ax = plt.subplots()
        ax.imshow(sz_dissim_mat)
        ax.set_title(pt)
    if pt == "HUP082":
        fig, ax = plt.subplots()
        ax.imshow(sz_dissim_mat)
        ax.set_title(pt)


fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(xcoords, ycoords)
ax.set_yticks(range(len(patient_cohort)))
ax.set_yticklabels(patient_cohort["Patient"])
ax.set_xlabel("Seizure dissimilarity")
plt.savefig(ospj(figure_path, "group_sz_dissimilarity.svg"), bbox_inches='tight')
# %%
