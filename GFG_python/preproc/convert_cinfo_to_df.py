import numpy as np
import pandas as pd
import os.path as op
from glob import glob
from scipy.io import loadmat

data_path = op.join(op.dirname(op.dirname(__file__)), 'data')
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
df.to_csv(op.join(data_path, 'cinfo.tsv'), sep="\t", index=False)
