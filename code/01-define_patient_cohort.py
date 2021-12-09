'''
Criteria for this project is as follows:
    1) Good outcome patients
    2) MTL target
    3) At least x recorded seizures

Inputs:
None

Outputs:
patient_cohort.xlsx
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
criteria1 = np.floor(metadata_table['Engel_12_mo']) <= 1
# MTL target
criteria2 = metadata_table['Target'].str.contains(r'MTL', na=True)
criteria3 = metadata_table['Target'].str.contains(r'Temporal', na=True)
# Get and sort by number of seizures
for i_pt, pt in enumerate(metadata_table["Patient"]):
    metadata_table.at[i_pt, ["Num Seizures"]] = len(pull_sz_starts(pt, metadata))
metadata_table.sort_values(by='Num Seizures', axis=0, ascending=False, inplace=True)

cohort = metadata_table[criteria1 & (criteria2 |  criteria3)]

# skip HUP99 because the electrodes file is for a different dataset than the 
# seizure times
cohort = cohort[cohort["Patient"] != "HUP099"]
cohort = cohort[cohort["Patient"] != "HUP117"]
cohort = cohort[cohort["Patient"] != "HUP165"]
cohort['Ignore'] = False
cohort.to_excel(ospj(data_path, "patient_cohort.xlsx"), index=False)

#
# 
# #  %%
# # This is the cohort with the most seizures
# patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(ospj(data_path, 'patient_localization_final.mat'))
# metadata_table = pd.read_excel(ospj(data_path, "atlas_metadata_simplified.xlsx"))

# metadata_table = metadata_table[metadata_table.filter(regex='^(?!Unnamed)').columns]
# # Good outcome
# criteria1 = np.floor(metadata_table['Engel_12_mo']) == 1
# # Get and sort by number of seizures
# for i_pt, pt in enumerate(metadata_table["Patient"]):
#     metadata_table.at[i_pt, ["Num Seizures"]] = len(pull_sz_starts(pt, metadata))

# criteria2 = metadata_table["Num Seizures"] > 2
# cohort = metadata_table[criteria1 & criteria2]
# cohort.to_excel(ospj(data_path, "patient_cohort.xlsx"), index=False)

# %% Summary statistics
therapy_val_counts = cohort["Therapy"].value_counts()
implant_val_counts = cohort["Implant"].value_counts()
laterality_val_counts = cohort["Laterality"].value_counts()
lesion_val_counts = cohort["Lesion_status"].value_counts()


mean_age = cohort["Age_surgery"].mean()
std_age = cohort["Age_surgery"].std()

n_sz = int(cohort["Num Seizures"].sum())
min_sz = int(cohort["Num Seizures"].min())
max_sz = int(cohort["Num Seizures"].max())

# %%
print(
    '''
    {}
    {}
    {}
    {}

    Mean age: {:.2f}\u00B1{:.2f}
    Number of Seizures: {}
    Max/min number of seizures: {}/{}
    '''.format(
            therapy_val_counts, 
            implant_val_counts, 
            laterality_val_counts, 
            lesion_val_counts, 
            mean_age, 
            std_age, 
            n_sz,
            min_sz,
            max_sz
        )

)
# %%
