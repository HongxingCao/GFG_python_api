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
    elif version == 'v2' or version == 'v2dense':
        df = df[df.proc_v2 == 1]
    else:
        raise ValueError("Version not known. Pick v1, v2, v2dense.")

    return df


def get_scodes_given_criteria(gender, age, age_range, ethn, version='v1'):
    """ Returns all scodes conforming to the age/ethn/gender criteria. """
    cinfo = load_cinfo(version=version)
    age_up, age_down = age + age_range, age - age_range
    query = 'gender == @gender & @age_down <= age <= @age_up'
    query += ' & %s == 1' % ethn

    if 'v2' in version:
        query += ' & proc_v2 == 1'
    else:
        query += ' & proc == 1'

    filtered = cinfo.query(query)
    face_ids = filtered.scode.values
    return face_ids


def get_info_given_scode(scode, version='v1'):
    """ Returns info (gender, age, ethnicity) from a face with a given
    scode. """
    cinfo = load_cinfo(version='v1')
    this_face = cinfo.loc[cinfo['scode'] == scode]
    gender, age = this_face['gender'].iloc[0], int(this_face['age'].iloc[0])
    tmp = this_face[['BA', 'WC', 'EA']]
    ethn = tmp.columns[(tmp == 1).iloc[0]][0]
    return dict(gender=gender, age=age, ethn=ethn)


def get_all_au_labels(remove_prefix=True):
    """ Finds all possible AU-labels. """
    data_dir = op.join(op.dirname(__file__), 'data')
    all_au_labels = loadmat(op.join(data_dir, 'au_labels.mat'))['au_labels'][0]
    all_au_labels = [l[0] for l in all_au_labels]

    if 'AU' in all_au_labels[0] and remove_prefix:
        # Remove prefix
        all_au_labels = [str(aul).split('AU')[1] for aul in all_au_labels]

    return all_au_labels


def sample_AUs(au_labels, n=5, p=0.6):
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
    return these_AUs
