#%%
from PIL.Image import ROTATE_180
import numpy as np
import json
from os.path import join as ospj
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

pd.set_option("display.max_rows", None, "display.max_columns", None)

# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
electrodes = config['electrodes']
bands = config['bands']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')


seizure_metadata = pd.read_excel(ospj(data_path, "seizure_metadata_with_soz_subgraph.xlsx"))

# remove "other" rows
seizure_metadata = seizure_metadata[seizure_metadata['Seizure category'] != "Other"]
seizure_metadata = seizure_metadata.dropna().reset_index(drop=True)

all_soz_dissim_mat = np.load(ospj(data_path, "all_soz_dissimilarity_band-{}_elec-{}.npy".format(bands, electrodes)))
#%%
seizure_metadata["isStatusChanged"] = seizure_metadata["Patient"].shift(1, fill_value=seizure_metadata["Patient"].head(1)) != seizure_metadata["Patient"]

xticklocs = seizure_metadata.index[seizure_metadata['isStatusChanged']] + 0.5
xlabelocs = np.insert(seizure_metadata.index[seizure_metadata['isStatusChanged']], 0, 0)
xlabelocs = seizure_metadata.index[seizure_metadata['isStatusChanged']] - np.diff(xlabelocs) / 2
#%%
fig, ax = plt.subplots(figsize=(10, 10))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.10)

im = ax.imshow(all_soz_dissim_mat, cmap='BuPu')
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.set_ylabel('SOZ Subgraph Dissimilarity', rotation=270)
ax.set_xlabel("Seizure Index")
ax.set_ylabel("Seizure Index")

plt.savefig(ospj(figure_path, "all_soz_dissim_band-{}_elec-{}.svg".format(bands, electrodes)), transparent=True)
plt.savefig(ospj(figure_path, "all_soz_dissim_band-{}_elec-{}.png".format(bands, electrodes)), transparent=True)
plt.close()

fig, ax = plt.subplots(figsize=(10, 10))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.10)

im = ax.imshow(all_soz_dissim_mat, cmap='BuPu')
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.set_ylabel('SOZ Subgraph Dissimilarity', rotation=270)
ax.set_xlabel("Seizure Index")
ax.set_ylabel("Seizure Index")

for val in xticklocs:
    ax.axvline(val, c='w', lw=2)
    ax.axhline(val, c='w', lw=2)

plt.savefig(ospj(figure_path, "all_soz_dissim_ptoverlay_band-{}_elec-{}.svg".format(bands, electrodes)), transparent=True)
plt.savefig(ospj(figure_path, "all_soz_dissim_ptoverlay_band-{}_elec-{}.png".format(bands, electrodes)), transparent=True)
plt.close()

sorted_idx = seizure_metadata['Seizure category'].argsort()
fig, ax = plt.subplots(figsize=(10, 10))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.10)

im = ax.imshow(all_soz_dissim_mat[:, sorted_idx][sorted_idx, :], cmap='BuPu')
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.set_ylabel('SOZ Subgraph Dissimilarity', rotation=270)
ax.set_xlabel("Seizure Index")
ax.set_ylabel("Seizure Index")

divder = (seizure_metadata['Seizure category'] == "FBTCS").sum() - 0.5
ax.axvline(divder, c='w', lw=2)
ax.axhline(divder, c='w', lw=2)

plt.savefig(ospj(figure_path, "all_soz_dissim_sztypeoverlay_band-{}_elec-{}.svg".format(bands, electrodes)), transparent=True)
plt.savefig(ospj(figure_path, "all_soz_dissim_sztypeoverlay_band-{}_elec-{}.png".format(bands, electrodes)), transparent=True)
plt.close()



# %%
fig, ax = plt.subplots(figsize=(10, 10))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.10)

im = ax.imshow(all_soz_dissim_mat, cmap='BuPu')
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.set_ylabel('SOZ Subgraph Dissimilarity', rotation=270)
ax.set_xlabel("Seizure Index")
ax.set_ylabel("Seizure Index")

sample1 = 58
sample2 = 21
ax.add_patch(
     patches.Rectangle(
         (sample1-0.5, sample2-0.5),
         1.0,
         1.0,
         edgecolor='red',
         fill=False,
         lw=2
     ) )

plt.savefig(ospj(figure_path, "all_soz_dissim_sampleoverlay_band-{}_elec-{}.svg".format(bands, electrodes)), transparent=True)
plt.savefig(ospj(figure_path, "all_soz_dissim_sampleoverlay_band-{}_elec-{}.png".format(bands, electrodes)), transparent=True)
plt.close()

display(seizure_metadata.iloc[[sample1, sample2]])
# %%
