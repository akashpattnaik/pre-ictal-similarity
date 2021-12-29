# %%
import numpy as np
import pandas as pd
import json
from os.path import join as ospj
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import sys, os

code_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(ospj(code_path, 'tools'))

from line_length import line_length
# See all columns
# pd.set_option('display.max_rows', 10)


with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
data_path = ospj(repo_path, 'data')
electrodes_opt = config['electrodes']

# %%
sz_metadata = pd.read_excel(ospj(data_path, "seizure_metadata.xlsx"))
sz_metadata = sz_metadata[sz_metadata['Seizure category'] != 'Other']

sz_metadata = sz_metadata[sz_metadata['Patient'] != "HUP082"]

# display(sz_metadata.sort_values(by='Seizure duration'))

# drop status epilepticus (>5 mins)
display(sz_metadata[sz_metadata['Seizure duration'] > 300])
sz_metadata = sz_metadata[sz_metadata['Seizure duration'] < 300]
# These seizures are more than double the length of the next largest seizures
# sz_metadata = sz_metadata.drop([137, 125, 36, 134])
# %%
sz_metadata.boxplot(column='Seizure duration', by='Seizure category')

fbtcs_df = sz_metadata[sz_metadata['Seizure category']=='FBTCS']
focal_df = sz_metadata[sz_metadata['Seizure category']=='Focal']

print(ttest_ind(fbtcs_df['Seizure duration'], focal_df['Seizure duration']))

# %%
for index, row in sz_metadata.iterrows():
    pt = row['Patient']
    sz_num = row['Seizure number']

    fname = "raw_signal_elec-{}_period-ictal_sz-{}.pkl".format(electrodes_opt, sz_num)

    sz_df = pd.read_pickle(ospj(data_path, pt, fname))

    time = sz_df.index.total_seconds()

    plt.figure(figsize=(25,25))
    plt.plot(sz_df + np.arange(72)*1000, color='k')

    break
# %%
