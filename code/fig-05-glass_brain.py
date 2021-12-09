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
from nilearn import plotting
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

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort_test.xlsx"))

patient_localization_mat = loadmat(ospj(metadata_path, 'patient_localization_final.mat'))['patient_localization']
patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(ospj(metadata_path, 'patient_localization_final.mat'))


# %%
for index, row in patient_cohort.iterrows():
    pt = row['Patient']
    iEEG_filename = row['portal_ID']

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    # pull and format electrode metadata
    target_electrodes_vars = loadmat(ospj(pt_data_path, "selected_electrodes_elec-{}.mat".format(electrodes_opt)))
    target_electrode_region_inds = target_electrodes_vars['targetElectrodesRegionInds'][0] - 1
    pt_index = patients.index(pt)
    sz_starts = pull_sz_starts(pt, metadata)

    # find seizure onset zone and state with most seizure onset zone
    soz_electrodes = np.array(np.squeeze(soz[pt_index][target_electrode_region_inds, :]), dtype=bool)

    # %%
    pt_coords = coords[pt_index][target_electrode_region_inds, :]
    # %%
    n_electrodes = pt_coords.shape[0]

    fig = plt.figure()
    plotting.plot_markers(
        np.ones((n_electrodes)), 
        pt_coords,
        node_size=np.ones((n_electrodes))*50,
        display_mode='x',
        figure=fig,
        colorbar=False
        )

    plt.savefig(ospj(pt_figure_path, "glass_brain_elec-{}.svg".format(electrodes_opt)), bbox_inches='tight', transparent='true')
    plt.savefig(ospj(pt_figure_path, "glass_brain_elec-{}.png".format(electrodes_opt)), bbox_inches='tight', transparent='true')

    # %%

    H = np.load(ospj(pt_data_path, "nmf_coefficients_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    n_components = H.shape[0]

    component_arr = np.reshape(H, (n_components, -1, n_electrodes))
    # %%
    for i_component in range(n_components):
        fig = plt.figure()
        plotting.plot_markers(
            np.mean(component_arr[i_component, :, :], axis=0), 
            pt_coords,
            node_size=np.ones((n_electrodes))*50,
            display_mode='x',
            figure=fig,
            title="Subgraph {}".format(i_component),
            colorbar=False
            )

        plt.savefig(ospj(pt_figure_path, "glass_brain_subgraph_{}_{}.png".format(i_component, mode)), bbox_inches='tight', transparent='true')
        plt.savefig(ospj(pt_figure_path, "glass_brain_subgraph_{}_{}.svg".format(i_component, mode)), bbox_inches='tight', transparent='true')

# %%
