# %%
# Imports and environment setup
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from os.path import join as ospj
pd.set_option("display.max_rows", None, "display.max_columns", None)

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
# %% First with ictal
focal_focal_dissims = []
fbtcs_focal_dissims = []
fbtcs_fbtcs_dissims = []

for index, row in patient_cohort.iterrows():
    pt = row["Patient"]
    # why do these patients have such high dissimilarity
    # if pt == "HUP082":
    #     continue
    # if pt == "HUP088":
    #     continue

    # print("Collecting dissimilarity matrices for {}".format(pt))

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    sz_dissim_fname = "sz_dissim_mat_dtw_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)
    sz_dissim_mat = np.load(ospj(pt_data_path, sz_dissim_fname))
    
    pt_sz_metadata = sz_metadata[sz_metadata['Patient'] == pt]
    
    assert(len(pt_sz_metadata) == sz_dissim_mat.shape[0])

    focal_inds = np.array(pt_sz_metadata.index[pt_sz_metadata['Seizure category'] == "Focal"].tolist())
    focal_inds = focal_inds - pt_sz_metadata.index[0]

    fbtcs_inds = np.array(pt_sz_metadata.index[pt_sz_metadata['Seizure category'] == "FBTCS"].tolist())
    fbtcs_inds = fbtcs_inds - pt_sz_metadata.index[0]

    sz_dissim_mat_utri = np.triu(sz_dissim_mat)

    if len(focal_inds) > 0:
        focal_focal_vals = sz_dissim_mat_utri[np.ix_(focal_inds, focal_inds)]
        focal_focal_dissims.extend(focal_focal_vals[focal_focal_vals != 0])

        if (focal_focal_vals > 2000).any():
            print(pt)

    if len(focal_inds) > 0 and len(fbtcs_inds) > 0:
        fbtcs_focal_vals = sz_dissim_mat_utri[np.ix_(focal_inds, fbtcs_inds)]
        fbtcs_focal_dissims.extend(fbtcs_focal_vals[fbtcs_focal_vals != 0])
        fbtcs_focal_vals = sz_dissim_mat_utri[np.ix_(fbtcs_inds, focal_inds)]
        fbtcs_focal_dissims.extend(fbtcs_focal_vals[fbtcs_focal_vals != 0])

    if len(fbtcs_inds) > 0:
        fbtcs_fbtcs_vals = sz_dissim_mat_utri[np.ix_(fbtcs_inds, fbtcs_inds)]
        fbtcs_fbtcs_dissims.extend(fbtcs_fbtcs_vals[fbtcs_fbtcs_vals != 0])


fig, ax = plt.subplots()
plt.boxplot(
    (focal_focal_dissims, fbtcs_focal_dissims, fbtcs_fbtcs_dissims),
    bootstrap=1000,
    labels=['Focal-Focal', 'Focal-FBTCS', 'FBTCS-FBTCS']
    )
plt.ylabel("Seizure dissimilarity (a.u.)")
plt.savefig(ospj(figure_path, "seizure_dissim_boxplot_band-{}_elec-{}.svg".format(band_opt, electrodes_opt)), bbox_inches='tight', transparent='true')
plt.savefig(ospj(figure_path, "seizure_dissim_boxplot_band-{}_elec-{}.svg".format(band_opt, electrodes_opt)), bbox_inches='tight', transparent='true')
# plt.close()

print(ttest_ind(fbtcs_focal_dissims, focal_focal_dissims))
print(ttest_ind(fbtcs_focal_dissims, fbtcs_fbtcs_dissims))
print(ttest_ind(focal_focal_dissims, fbtcs_fbtcs_dissims))

# %% Pre-ictal
focal_focal_dissims = []
fbtcs_focal_dissims = []
fbtcs_fbtcs_dissims = []

for index, row in patient_cohort.iterrows():
    pt = row["Patient"]
    if pt == "HUP111":
        continue
    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    soz_dissim_fname = "soz_subgraph_dissim_mat_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)
    soz_dissim_mat = np.load(ospj(pt_data_path, soz_dissim_fname))
    

    pt_sz_metadata = sz_metadata[sz_metadata['Patient'] == pt]

    remaining_sz_ids = np.load(ospj(pt_data_path, "remaining_sz_ids.npy"))
    pt_sz_metadata = pt_sz_metadata[pt_sz_metadata['Seizure number'].isin(remaining_sz_ids)]

    assert(len(pt_sz_metadata) == soz_dissim_mat.shape[0])

    focal_inds = np.array(pt_sz_metadata.index[pt_sz_metadata['Seizure category'] == "Focal"].tolist())
    focal_inds = focal_inds - pt_sz_metadata.index[0]
    focal_inds = np.array([np.where(remaining_sz_ids == i)[0] for i in focal_inds]).T
    if np.size(focal_inds) > 0:
        focal_inds = focal_inds[0]

    fbtcs_inds = np.array(pt_sz_metadata.index[pt_sz_metadata['Seizure category'] == "FBTCS"].tolist())
    fbtcs_inds = fbtcs_inds - pt_sz_metadata.index[0]
    fbtcs_inds = np.array([np.where(remaining_sz_ids == i)[0] for i in fbtcs_inds]).T
    if np.size(fbtcs_inds) > 0:
        fbtcs_inds = fbtcs_inds[0]

    soz_dissim_mat_utri = np.triu(soz_dissim_mat)

    if len(focal_inds) > 0:
        focal_focal_vals = soz_dissim_mat_utri[np.ix_(focal_inds, focal_inds)]
        focal_focal_dissims.extend(focal_focal_vals[focal_focal_vals != 0])

        if (focal_focal_vals > 2000).any():
            print(pt)

    if len(focal_inds) > 0 and len(fbtcs_inds) > 0:
        fbtcs_focal_vals = soz_dissim_mat_utri[np.ix_(fbtcs_inds, focal_inds)]
        fbtcs_focal_dissims.extend(fbtcs_focal_vals[fbtcs_focal_vals != 0])
        # fbtcs_focal_vals = soz_dissim_mat_utri[np.ix_(fbtcs_inds,focal_inds)]
        # fbtcs_focal_dissims.extend(fbtcs_focal_vals[fbtcs_focal_vals != 0])

    if len(fbtcs_inds) > 0:
        fbtcs_fbtcs_vals = soz_dissim_mat_utri[np.ix_(fbtcs_inds, fbtcs_inds)]
        fbtcs_fbtcs_dissims.extend(fbtcs_fbtcs_vals[fbtcs_fbtcs_vals != 0])


fig, ax = plt.subplots()
plt.boxplot(
    (focal_focal_dissims, fbtcs_focal_dissims, fbtcs_fbtcs_dissims),
    bootstrap=1000,
    labels=['Focal-Focal', 'Focal-FBTCS', 'FBTCS-FBTCS']
    )
plt.ylabel("Pre-seizure dissimilarity (a.u.)")
plt.savefig(ospj(figure_path, "pre_seizure_dissim_boxplot_band-{}_elec-{}.svg".format(band_opt, electrodes_opt)), bbox_inches='tight', transparent='true')
plt.savefig(ospj(figure_path, "pre_seizure_dissim_boxplot_band-{}_elec-{}.svg".format(band_opt, electrodes_opt)), bbox_inches='tight', transparent='true')
# plt.close()

print(len(fbtcs_focal_dissims))
print(len(focal_focal_dissims))
print(len(fbtcs_fbtcs_dissims))

print(ttest_ind(fbtcs_focal_dissims, focal_focal_dissims))
print(ttest_ind(fbtcs_focal_dissims, fbtcs_fbtcs_dissims))
print(ttest_ind(focal_focal_dissims, fbtcs_fbtcs_dissims))

# %%

seizure_metadata = pd.read_excel(ospj(data_path, "seizure_metadata_with_soz_subgraph.xlsx"))

# remove "other" rows
seizure_metadata = seizure_metadata[seizure_metadata['Seizure category'] != "Other"]
seizure_metadata = seizure_metadata.dropna().reset_index(drop=True)

all_soz_dissim_mat = np.load(ospj(data_path, "all_soz_dissimilarity_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))

focal_inds = seizure_metadata['Seizure category'] == 'Focal'
fbtcs_inds = seizure_metadata['Seizure category'] == 'FBTCS'
focal_dissim = all_soz_dissim_mat[focal_inds, :][:, focal_inds]
fbtcs_dissim = all_soz_dissim_mat[fbtcs_inds, :][:, fbtcs_inds]

focal_dissim_vals = focal_dissim[np.triu_indices(sum(focal_inds), k=1)]
fbtcs_dissim_vals = fbtcs_dissim[np.triu_indices(sum(fbtcs_inds), k=1)]


focal_fbtcs_dissim = np.triu(all_soz_dissim_mat)[focal_inds, :][:, fbtcs_inds]
focal_fbtcs_dissim_vals = focal_fbtcs_dissim[focal_fbtcs_dissim != 0]


fig, ax = plt.subplots()
plt.boxplot(
    (focal_dissim_vals, fbtcs_dissim_vals, focal_fbtcs_dissim_vals),
    bootstrap=1000,
    labels=['Focal-Focal', 'FBTCS-FBTCS', 'Focal-FBTCS']
    )
plt.ylabel("Pre-seizure dissimilarity (a.u.)")
plt.savefig(ospj(figure_path, "pre_seizure_all_dissim_boxplot_band-{}_elec-{}.svg".format(band_opt, electrodes_opt)), bbox_inches='tight', transparent='true')
plt.savefig(ospj(figure_path, "pre_seizure_all_dissim_boxplot_band-{}_elec-{}.svg".format(band_opt, electrodes_opt)), bbox_inches='tight', transparent='true')
# plt.close()

print(len(focal_dissim_vals))
print(len(fbtcs_dissim_vals))
print(len(focal_fbtcs_dissim_vals))

print(ttest_ind(focal_dissim_vals, fbtcs_dissim_vals))
print(ttest_ind(focal_dissim_vals, focal_fbtcs_dissim_vals))
print(ttest_ind(fbtcs_dissim_vals, focal_fbtcs_dissim_vals))

# %%
