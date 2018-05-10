import numpy as np
import pandas as pd
import os.path as op
from scipy.io import loadmat
from glob import glob


def load_cinfo_from_mat_files(save_to_disk=False):
    """ Loads cinfo and converts it to a pandas dataframe. """

    data_path = op.join(op.dirname(__file__), 'data')
    mat_files = glob(op.join(data_path, '*.mat'))

    to_save = dict()
    for matf in mat_files:
        fname = op.basename(matf).split('.mat')[0]
        this_field = loadmat(matf)['this_field']

        if isinstance(this_field[0][0], np.ndarray):
            # Must be string
            clean = np.array([f[0][0] for f in this_field])
        else:
            clean = np.array([f[0] for f in this_field])

        to_save[fname] = clean

    df = pd.DataFrame(to_save)

    if save_to_disk:
        df.to_csv(op.join(data_path, 'cinfo.tsv'), sep="\t", index=False)

    return df


def get_all_au_labels():
    """ Finds all possible AU-labels. """
    all_au_labels = loadmat(op.join(op.dirname(__file__), 'au_labels.mat'))
    all_au_labels = [str(aul[0]).split('AU')[1]
                     for aul in all_au_labels['au_labels'][0]]
    return all_au_labels
