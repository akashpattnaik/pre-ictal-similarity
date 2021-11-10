'''
Criteria for this project is as follows:
    1) Good outcome patients
    2) MTL target
    3) At least x recorded seizures

'''
#%%
# Imports and environment setup
import numpy as np
import os, sys
import json
import pandas as pd

from os.path import join as ospj

sys.path.append('tools')

from pull_patient_localization import pull_patient_localization
from pull_sz_starts import pull_sz_starts

# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
metadata_path = config['metadataPath']

data_path = ospj(repo_path, 'data')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

#### Ultimately use this cohort
# %% Pull and format metadata from patient_localization_mat
patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(ospj(data_path, 'patient_localization_final.mat'))
metadata_table = pd.read_excel(ospj(data_path, "atlas_metadata_simplified.xlsx"))
metadata_table = metadata_table[metadata_table.filter(regex='^(?!Unnamed)').columns]

# Good outcome
criteria1 = np.floor(metadata_table['Engel_6_mo']) <= 2
# MTL target
criteria2 = metadata_table['Target'].str.contains(r'MTL', na=True)
# Get and sort by number of seizures
for i_pt, pt in enumerate(metadata_table["Patient"]):
    metadata_table.at[i_pt, ["Num Seizures"]] = len(pull_sz_starts(pt, metadata))
metadata_table.sort_values(by='Num Seizures', axis=0, ascending=False, inplace=True)

cohort = metadata_table[criteria1 & criteria2]
cohort.to_excel(ospj(data_path, "patient_cohort.xlsx"), index=False)
# %%
# This is the cohort with the most seizures
patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(ospj(data_path, 'patient_localization_final.mat'))
metadata_table = pd.read_excel(ospj(data_path, "atlas_metadata_simplified.xlsx"))

metadata_table = metadata_table[metadata_table.filter(regex='^(?!Unnamed)').columns]
# Good outcome
criteria1 = np.floor(metadata_table['Engel_12_mo']) == 1
# Get and sort by number of seizures
for i_pt, pt in enumerate(metadata_table["Patient"]):
    metadata_table.at[i_pt, ["Num Seizures"]] = len(pull_sz_starts(pt, metadata))

criteria2 = metadata_table["Num Seizures"] > 4
cohort = metadata_table[criteria1 & criteria2]
cohort.to_excel(ospj(data_path, "patient_cohort.xlsx"), index=False)

# %%
