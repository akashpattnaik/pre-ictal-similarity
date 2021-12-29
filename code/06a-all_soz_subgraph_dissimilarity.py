# %%
# %load_ext autoreload
# %autoreload 2
# Imports and environment setup
import numpy as np
import sys
import pandas as pd
import json
from tqdm import tqdm
from os.path import join as ospj

sys.path.append('tools')

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
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
electrodes = config['electrodes']
bands = config['bands']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

seizure_metadata = pd.read_excel(ospj(data_path, "seizure_metadata_with_soz_subgraph.xlsx"))

# remove "other" rows
seizure_metadata = seizure_metadata[seizure_metadata['Seizure category'] != "Other"]
seizure_metadata = seizure_metadata.dropna().reset_index(drop=True)
# flags
SAVE_PLOT = True

NMF_FLAG = True
FIXED_PREICTAL_SEC = 60 * 30
LEAD_SZ_WINDOW_SEC = (FIXED_PREICTAL_SEC + 60 * 15) # 15 min buffer

# %%
all_soz_component_expressions = []
n_all_sz = len(seizure_metadata)
for index, row in seizure_metadata.iterrows():
    pt = row['Patient']
    sz_num = row['Seizure number']
    pt_soz_component = row['SOZ Sensitive State (mann-whitney)']
    pt_soz_component = int(pt_soz_component)

    pt_data_path = ospj(data_path, pt)

    W = np.load(ospj(pt_data_path, "nmf_expression_band-{}_elec-{}_sz-{}.npy".format(bands, electrodes, sz_num)))
    soz_component_expression = W[:, pt_soz_component]

    all_soz_component_expressions.append(soz_component_expression)
# %%
all_soz_dissim_mat = np.zeros((n_all_sz, n_all_sz))
for i_sz in tqdm(range(n_all_sz)):
    for j_sz in range(n_all_sz):
        if j_sz > i_sz:
            dist, path = fastdtw(all_soz_component_expressions[i_sz], all_soz_component_expressions[j_sz], dist=euclidean)
            all_soz_dissim_mat[i_sz, j_sz] = dist
all_soz_dissim_mat = all_soz_dissim_mat + all_soz_dissim_mat.T
np.save(ospj(data_path, "all_soz_dissimilarity_band-{}_elec-{}.npy".format(bands, electrodes)), all_soz_dissim_mat)
# %%
