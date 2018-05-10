from __future__ import print_function
import numpy as np
from main import run
from utils import get_all_au_labels
import matlab.engine as mlab_eng

FACE_ID = 1377
ITERS = 100
N_AUs = 3

all_aus = get_all_au_labels()
save_path = '/Users/lukas/desktop/testGFG'
combinations = []

print("Starting Matlab ...", end="")
engine = mlab_eng.start_matlab()
print(" done.")

# To do: pick number of b

for i in range(ITERS):

    these_AUs = np.random.choice(all_aus, size=N_AUs, replace=False)
    these_AUs = these_AUs.tolist()
    while these_AUs in combinations:
        these_AUs = np.random.choice(all_aus, size=N_AUs, replace=False)

    run(face_id=FACE_ID, save_path=save_path, au_labels=these_AUs,
        temp_params=None, eye_params=None, head_params=None, engine=engine)

    combinations.append(these_AUs)
