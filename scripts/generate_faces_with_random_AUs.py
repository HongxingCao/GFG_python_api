import GFG_python
import os.path as op
from GFG_python.main import run
from GFG_python.utils import sample_AUs, get_all_au_labels
import numpy as np
from matlab import engine as mlab_eng

# Parameters
N_samples = 2
genders = np.repeat(['M', 'F'], repeats=int(N_samples/2))
age = 27
age_range = 10
ethnicity = 'WC'
save_path = '/Users/lukas/Desktop/random_AU_dataset'
sample_temporal_parameters = True
all_au_labels = get_all_au_labels()

# Start Matlab
engine = mlab_eng.start_matlab()
engine.addpath(op.dirname(GFG_python.__file__), nargout=0)

already_chosen_AU_sets = []

i = 0
while i < N_samples:
    # Write parameters to json or something
    these_AUs = sample_AUs(all_au_labels, n=5, p=0.6).tolist()

    if these_AUs in already_chosen_AU_sets:
        continue
    else:
        already_chosen_AU_sets.append(these_AUs)

    N_aus = len(these_AUs)

    if sample_temporal_parameters:
        temp_params = np.random.uniform(0, 1, (N_aus, 6))
    else:
        temp_params = None

    run(face_id=None, nfdata=None, save_path=save_path, au_labels=these_AUs,
        temp_params=temp_params, eye_params=None, head_params=None,
        gender=genders[i], ethnicity=ethnicity, age=age, age_range=age_range,
        version='v1', engine=engine)

    i += 1
