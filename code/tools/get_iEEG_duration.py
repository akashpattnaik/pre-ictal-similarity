from ieeg.auth import Session

def get_iEEG_duration(username, password, iEEG_filename):

    s = Session(username, password)
    ds = s.open_dataset(iEEG_filename)
    return ds.get_time_series_details(ds.ch_labels[0]).duration


