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

sz_metadata = pd.read_excel(ospj(data_path, "seizure_metadata.xlsx"))
# %%
focal_focal_dissims = []
fbtcs_focal_dissims = []
fbtcs_fbtcs_dissims = []

for index, row in patient_cohort.iterrows():
    pt = row["Patient"]
    pt = "HUP187"
    print("Collecting dissimilarity matrices for {}".format(pt))

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)


    soz_subgraph_dissim_mat = np.load(ospj(pt_data_path, "soz_subgraph_dissim_mat_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    remaining_sz_ids = np.load(ospj(pt_data_path, "remaining_sz_ids.npy"))

    n_remaining_inds = np.size(remaining_sz_ids)
    # get upper triangular indices of square matrices where n is the number of lead seizures
    remaining_tri_inds = np.triu_indices(n_remaining_inds, k=1)
    remaining_sz_dissim_mat = soz_subgraph_dissim_mat[remaining_sz_ids[:, None] - 1, remaining_sz_ids - 1]

    states_dissim = soz_subgraph_dissim_mat[remaining_tri_inds]
    remaining_sz_dissim = remaining_sz_dissim_mat[remaining_tri_inds]
    
    corr, sig = pearsonr(remaining_sz_dissim, states_dissim)
    patient_cohort.at[index, 'Seizure-SOZ Subgraph Correlation'] = corr

    
    pt_sz_metadata = sz_metadata[sz_metadata['Patient'] == pt]
    pt_sz_metadata = pt_sz_metadata[pt_sz_metadata["Seizure number"].isin(remaining_sz_ids)]
    
    assert(len(pt_sz_metadata) == soz_subgraph_dissim_mat.shape[0])

    focal_inds = np.array(pt_sz_metadata.index[pt_sz_metadata['Seizure category'] == "Focal"].tolist())
    focal_inds = focal_inds - pt_sz_metadata.index[0]

    fbtcs_inds = np.array(pt_sz_metadata.index[pt_sz_metadata['Seizure category'] == "FBTCS"].tolist())
    fbtcs_inds = fbtcs_inds - pt_sz_metadata.index[0]

    
    break
# %%
