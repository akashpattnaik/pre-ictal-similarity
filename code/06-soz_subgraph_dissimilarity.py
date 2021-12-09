# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join as ospj
import pandas as pd
import json
from sklearn.preprocessing import normalize
import sys
sys.path.append('tools')
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from movmean import movmean

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
for index, row in patient_cohort.iterrows():
    if row['Ignore']:
        continue

    pt = row["Patient"]
    print("Calculating dissimilarity for {}".format(pt))
    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    remaining_sz_ids = np.load(ospj(pt_data_path, "remaining_sz_ids.npy"))
    t_sec = np.load(ospj(pt_data_path, "lead_sz_t_sec_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    sz_id = np.load(ospj(pt_data_path, "lead_sz_sz_id_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    W = np.load(ospj(pt_data_path, "nmf_expression_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    soz_electrodes = np.load(ospj(pt_data_path, "soz_electrodes_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    pt_soz_state = np.load(ospj(pt_data_path, "pt_soz_state_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))

    n_electrodes = soz_electrodes.shape[0] 
    n_remaining_sz = np.size(remaining_sz_ids)
    soz_subgraph_dissim_mat = np.zeros((n_remaining_sz, n_remaining_sz))

    k = 60
    W_norm = normalize(W, norm='l1')

    for ind_i, i in enumerate(remaining_sz_ids):
        for ind_j, j in enumerate(remaining_sz_ids):
            if i != j:
                dist, path = fastdtw(W_norm[sz_id == i, pt_soz_state], W_norm[sz_id == j, pt_soz_state], dist=euclidean)
                soz_subgraph_dissim_mat[ind_i, ind_j] = dist
    np.save(ospj(pt_data_path, "soz_subgraph_dissim_mat_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)), soz_subgraph_dissim_mat)

# %%
