import numpy as np
import pandas as pd
import os.path as op
from scipy.io import loadmat


def load_cinfo(version='v1'):
    """ Loads cinfo given a version. """
    df_path = op.join(op.dirname(__file__), 'data', 'cinfo.tsv')
    df = pd.read_csv(df_path, sep='\t')

    if version == 'v1':
        df = df[df.proc == 1]
    elif version == 'v2':
        df = df[df.proc_v2 == 1]
    else:
        raise ValueError("Version not known. Pick v1 or v2.")

    return df


def get_all_au_labels():
    """ Finds all possible AU-labels. """
    all_au_labels = loadmat(op.join(op.dirname(__file__), 'au_labels.mat'))
    all_au_labels = [str(aul[0]).split('AU')[1]
                     for aul in all_au_labels['au_labels'][0]]
    return all_au_labels


def sample_AUs(au_labels, n=5, p=0.6, remove_prefix=True):
    """ Draws a random sample of AUs.

    Parameters
    ----------
    au_labels : numpy array
        Array with AU-labels
    n : int
        Parameter n of binomial distribution
    p : float
        Parameter p of binomial distribution

    Returns
    -------
    these_AUs : numpy array
        Array with selected AUs.
    """
    this_n = np.random.binomial(n=5, p=0.6)
    while this_n == 0:  # Resample if number of AUs to draw is 0
        this_n = np.random.binomial(n=5, p=0.6)

    these_AUs = np.random.choice(au_labels, size=this_n, replace=False)

    if remove_prefix:
        these_AUs = [au.split('AU')[1] for au in these_AUs]

    return these_AUs
