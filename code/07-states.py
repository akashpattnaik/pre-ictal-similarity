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
from sklearn.preprocessing import normalize

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

# %%

# Plot the NMF subgraphs and expression
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]
    print("Calculating states for {}".format(pt))

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    remaining_sz_ids = np.load(ospj(pt_data_path, "remaining_sz_ids.npy"))
    t_sec = np.load(ospj(pt_data_path, "lead_sz_t_sec_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    sz_id = np.load(ospj(pt_data_path, "lead_sz_sz_id_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    W = np.load(ospj(pt_data_path, "nmf_expression_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))

    sz_id = np.squeeze(sz_id)
    n_components = W.shape[-1]

    W_norm = normalize(W, axis=1, norm='l1')

    # States are defined as the max expressed component at each time point
    states = np.argmax(W_norm[:, :], axis=-1)
    np.save(ospj(pt_data_path, "states_{}_{}.npy".format(electrodes_opt, band_opt)), states)

    # for i in remaining_sz_ids:
    #     fig, ax = plt.subplots()
    #     t_arr_min = (t_sec[sz_id == i] - t_sec[sz_id == i][-1]) / 60

    #     ax.plot(t_arr_min, states[sz_id == i])

    #     ax.set_xlabel("Time from seizure onset (min)")
    #     ax.set_ylabel("Subgraph number")
    #     ax.set_yticks(np.arange(n_components, dtype=int))
    #     ax.set_title("Seizure {}".format(i))
        
    #     ax.set_xlim([-10, 0])

    #     fig, ax = plt.subplots()
    #     ax.imshow(np.expand_dims(states[sz_id == i], axis=0), aspect='auto', interpolation='none')
        
# %%
