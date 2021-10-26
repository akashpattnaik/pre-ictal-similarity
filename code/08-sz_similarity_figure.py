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


root_path = os.path.dirname(os.path.realpath(__file__))
root_path = "/".join(root_path.split("/")[:-1])
data_path = ospj(root_path, 'data')

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

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

    sz_dissim_mat = np.load(ospj(pt_data_path, "sz_dissim_mat.npy"))
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
plt.savefig(ospj(root_path, "figures", "group_sz_dissimilarity.svg"), bbox_inches='tight')
# %%
