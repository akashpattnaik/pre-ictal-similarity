# %%
import json
from os.path import join as ospj
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)

# %%
# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
metadata_path = config['metadataPath']

data_path = ospj(repo_path, 'data')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

# %%
df_list_pt = []
df_list_sz_num = []
df_list_sz_type = []
df_list_sz_category = []

df_list_sz_EEC = []
df_list_sz_UEO = []
df_list_sz_end = []
df_list_sz_duration = []


for pt in cohort['Patient']:
    # get all seizure types
    pt_sz_EEC = []
    pt_sz_UEO = []
    pt_sz_end = []

    ictal_events = metadata[pt]['Events']['Ictal']
    sz_id = 0
    for sz_num, item in ictal_events.items():
        if pt == "HUP111":
            if 'D01' in item['iEEG_record']:
                continue
        
        pt_sz_type = item['SeizureType']

        df_list_pt.append(pt)
        df_list_sz_num.append(sz_id)
        df_list_sz_type.append(pt_sz_type)

        if pt_sz_type in ["FIAS", "FAS"]:
            pt_sz_category = "Focal"
        elif pt_sz_type in ["FBTC", "FBT"]:
            pt_sz_category = "FBTCS"
        else:
            pt_sz_category = "Other"
        df_list_sz_category.append(pt_sz_category)

        df_list_sz_EEC.append(item['SeizureEEC'])
        df_list_sz_UEO.append(item['SeizureUEO'])
        df_list_sz_end.append(item['SeizureEnd'])

        pt_sz_EEC.append(item['SeizureEEC'])
        pt_sz_UEO.append(item['SeizureUEO'])
        pt_sz_end.append(item['SeizureEnd'])

        sz_id = sz_id + 1

    pt_sz_EEC = np.array(pt_sz_EEC)
    pt_sz_UEO = np.array(pt_sz_UEO)
    pt_sz_end = np.array(pt_sz_end)

    if not np.all(np.diff(pt_sz_EEC) >= 0):
        print("{} EEC is not in order".format(pt))
    if not np.all(np.diff(pt_sz_UEO) >= 0):
        print("{} UEO is not in order".format(pt))
    if not np.all(np.diff(pt_sz_end) >= 0):
        print("{} end is not in order".format(pt))
        print()

# %% calculate totals
df = pd.DataFrame(
    {
        "Patient": df_list_pt,
        "Seizure number": df_list_sz_num,
        "Seizure type": df_list_sz_type,
        "Seizure category": df_list_sz_category,
        "Seizure EEC": df_list_sz_EEC,
        "Seizure UEO": df_list_sz_UEO,
        "Seizure end": df_list_sz_end
    }
)
df['Seizure duration'] = df['Seizure end'] - df['Seizure UEO']
# display(df)
print(df['Seizure category'].value_counts())

# %% Check outlier durations
df.boxplot(column='Seizure duration', by='Seizure category')
# %% How many of each category per patient?
df_patient_types = df.groupby(['Patient', 'Seizure category']).size().unstack(fill_value=0)
df_patient_types
# %%
df_patient_types[(df_patient_types["FBTCS"] > 1) & (df_patient_types["Focal"] > 1)]
n_fbtcs_focal = (df_patient_types["FBTCS"] * df_patient_types["Focal"]).sum()
n_focal_focal = (df_patient_types["Focal"] * (df_patient_types["Focal"] - 1)).sum() / 2
n_ftbcs_fbtcs = (df_patient_types["FBTCS"] * (df_patient_types["FBTCS"] - 1)).sum() / 2
# %%
print(
    '''
    Focal-FBTCS: {}
    Focal-Focal: {}
    FBTCS-FBTCS: {}
    '''.format(n_fbtcs_focal, n_focal_focal, n_ftbcs_fbtcs)
)

# %%
df.to_excel(ospj(data_path, "seizure_metadata.xlsx"))
# %%
