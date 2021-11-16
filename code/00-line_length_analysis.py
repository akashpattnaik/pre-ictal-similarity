'''
This script applies a line length detector to 30 second clips of iEEG to 
determine patient specific thresholds for seizures.
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
from scipy.stats import zscore
import numpy.ma as ma
from os.path import join as ospj

sys.path.append('tools')
# from line_length import line_length
from get_iEEG_data import get_iEEG_data
from pull_sz_starts import pull_sz_starts
from pull_sz_ends import pull_sz_ends
from time2ind import time2ind
from movmean import movmean
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

PLOT = True

def consecutive(data, stepsize=1):
    data = np.where(data)[0]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

# %%
pt = "HUP187"
iEEG_fname = "HUP187_phaseII"

pt_data_path = ospj(data_path, pt)
pt_figure_path = ospj(figure_path, pt)

# pull and format electrode metadata
electrodes_mat = loadmat(ospj(pt_data_path, 'target-electrodes-regions.mat'))
target_electrode_region_inds = electrodes_mat['targetElectrodesRegionInds'][0] - 1
patient_localization_mat = loadmat(ospj(metadata_path, 'patient_localization_final.mat'))['patient_localization']
patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(ospj(metadata_path, 'patient_localization_final.mat'))

# %%
ll_data = np.load(ospj(pt_data_path, "ll_arr.npz"))
ll = ll_data["arr_0"]
t = ll_data['arr_1']
del ll_data
# %%


# %% where to cut off line length
threshold_range = np.arange(0, 500, 10)
n_meet_threshold = np.array([t[np.where(ll > i)[0]].shape[0] for i in threshold_range])
percent_above_threshold = n_meet_threshold / ll.shape[0]

fig, ax = plt.subplots()
ax.plot(threshold_range, percent_above_threshold)
ax.set_xlabel("Threshold Line Length")
ax.set_ylabel("# windows above threshold")



# %%
threshold = 200
artifact_mask = np.zeros(t.shape, dtype=bool)
artifact_mask[np.where(ll > threshold)[0]] = True

consecutive_artifacts = consecutive(artifact_mask)

for seg in consecutive_artifacts:
    if len(seg) < 5:
        artifact_mask[seg] = False

ll_clean = np.delete(ll, artifact_mask, axis=0)
t_clean = np.delete(t, artifact_mask, axis=0)


fig, ax = plt.subplots()
ax.plot(t_clean / (60 * 60 * 24), movmean(ll_clean.T, k=100).T)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Line length")

# %%
sz_starts = pull_sz_starts(pt, metadata)
sz_ends = pull_sz_ends(pt, metadata)
n_sz = np.size(sz_starts)
# %%
sz_start_inds = np.array([time2ind(i, t) for i in sz_starts])
sz_end_inds = np.array([time2ind(i, t) for i in sz_ends])
sz_duration_inds = sz_end_inds - sz_start_inds

sz_mask = np.zeros((t_clean.shape), dtype=bool)
for i_sz in range(n_sz):
    sz_mask[sz_start_inds[i_sz]:sz_end_inds[i_sz]] = True

# %% What is the line length during seizures?
ll_sz = np.delete(ll_clean, sz_mask)
ll_non_sz = ll_clean[~sz_mask]

t_sz = t_clean[sz_mask]
t_non_sz = t_clean[~sz_mask]

# %% pick 100 iterations of control
n_iter = 1000
controls = np.zeros((n_iter, np.sum(sz_duration_inds)))

i_iter = 0
while i_iter < n_iter:
    print(i_iter)
    rand_start_inds = np.random.randint(np.size(ll_non_sz), size=(n_sz))
    rand_end_inds = rand_start_inds + sz_duration_inds

    # redraw if indices go out of bounds
    if any(rand_end_inds > np.size(ll_non_sz)):
        continue

    # create random indices mask
    rand_mask = np.zeros((ll.shape), dtype=bool)

    for i_sz in range(n_sz):
        rand_mask[rand_start_inds[i_sz]:rand_end_inds[i_sz]] = True

    if np.sum(rand_mask) != np.sum(sz_duration_inds):
        continue

    controls[i_iter, :] = ll[rand_mask]
    i_iter = i_iter + 1

fig, ax = plt.subplots()
ax.hist(np.mean(controls, axis=-1))
ax.axvline(np.mean(ll_sz), c='r', ls='--')
ax.set_xlabel("Mean line length")

# # %%
# potential_sz_mask = np.zeros(ll.shape, dtype=bool)
# potential_sz_mask[ll > 20] = True
# fig, ax = plt.subplots()
# ax.plot(t, potential_sz_mask)
# for sz_start in sz_starts:
#     ax.axvline(sz_start, ls='--', c='r', alpha=1, lw=0.5)

# np.diff(np.where(np.concatenate(([potential_sz_mask[0]],
#                                      potential_sz_mask[:-1] != potential_sz_mask[1:],
#                                      [True])))[0])[::2]
# # %%
# min_sz_length_inds = 100
# candidate_sz = []
# for i in range(len(potential_sz_mask)):
#     if all(potential_sz_mask[i:(i+min_sz_length_inds)]):
#         candidate_sz.append(i)
# candidate_sz = np.array(candidate_sz)

# # %%

# %%
