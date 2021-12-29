# %%
import numpy as np
import json
# from nilearn import plotting
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from ieeg.auth import Session
import scipy
import csv
import pandas as pd
import os
from scipy.io import savemat, loadmat
from os.path import join as ospj
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append('tools')
from pull_patient_localization import pull_patient_localization
# Read credentials and start iEEG session
with open('../credentials.json') as f:
    credentials = json.load(f)
username = credentials['username']
password = credentials['password']
s = Session(username, password)

# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
metadata_path = config['metadataPath']

data_path = ospj(repo_path, 'data')
# %%
patients, labels, ignore, resect, gm_wm, coords, region, soz = pull_patient_localization(ospj(data_path, 'patient_localization_final.mat'))

# %% Load metadata file
metadata = pd.read_excel(os.path.join(metadata_path, 'atlas_metadata_simplified.xlsx'))

electrodes = config['electrodes']
# def identify_channels(mode='all'):
# for patient in metadata['Patient']:
for patient in patients:
    localization_ind = patients.index(patient)

    iEEG_filename = metadata[metadata['Patient'] == patient]['portal_ID'].item()

    pt_labels = labels[localization_ind]
    pt_ignore = ignore[localization_ind].T[0]
    pt_resect = resect[localization_ind].T[0]
    pt_gm_wm = gm_wm[localization_ind].T[0]

    # Set up region list
    pt_region = []
    for i in region[localization_ind]:
        if len(i[0]) == 0:
            pt_region.append('')
        else:
            pt_region.append(i[0][0])


    pt_soz = soz[localization_ind].T[0]

    df_data = {
        'labels': pt_labels,
        'ignore': pt_ignore,
        'resect': pt_resect,
        'gm_wm': pt_gm_wm,
        'region': pt_region,
        'soz': pt_soz
        }

    print("Starting pipeline for {0}, iEEG filename is {1}".format(patient, iEEG_filename))        

    df = pd.DataFrame(df_data).reset_index()

    df_filtered = df[df['ignore'] != 1]

    if electrodes == "regions":
        # Take first electrode in each region
        df_filtered = df_filtered.groupby("region").first()
    # Remove white matter and non-localized electrodes
    df_filtered = df_filtered[df_filtered['gm_wm'] != -1]
    # Sort rows in alphabetical order by electrode name, easier to read with iEEG.org
    df_filtered.sort_values(by=['labels'], inplace=True)

    if electrodes == "regions":
        mdic = {
            "iEEGFilename": iEEG_filename,
            "targetElectrodesRegionInds": np.array(df_filtered['index']), # +1 for one-indexing in MATLAB
            "Regions": list(df_filtered.index),
            "electrodeNames": list(df_filtered['labels'])
        }
    else:
        mdic = {
            "iEEGFilename": iEEG_filename,
            "targetElectrodesRegionInds": np.array(df_filtered['index']), # +1 for one-indexing in MATLAB
            "Regions": list(df_filtered['region']),
            "electrodeNames": list(df_filtered['labels'])
        }

    patient_data_path = os.path.join(data_path, patient)
    if not os.path.exists(patient_data_path):
        os.makedirs(patient_data_path)

    save_path = os.path.join(patient_data_path, "selected_electrodes_elec-{}.mat".format(electrodes))
    savemat(save_path, mdic)

    print("\t{} has {} channels after filtering".format(patient, len(mdic['Regions'])))
    print("\tResults are saved in {}".format(save_path))

# %%
