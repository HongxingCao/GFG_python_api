import GFG_python
import os.path as op
from GFG_python.main import run
from GFG_python.glm import FaceGenerator
from matlab import engine as mlab_eng

engine = mlab_eng.start_matlab()
engine.addpath(op.dirname(GFG_python.__file__), nargout=0)
fg = FaceGenerator(version='v1', save_dir='/Users/lukas/software/GFG_glm')

out = fg.generate_new_face(N=1, age=20, gender='F', ethn='WC')

for o in out:
    run(face_id=2, nfdata=o, save_path='/Users/lukas/Desktop/testGFG',
        au_labels=None, temp_params=None, eye_params=None,
        head_params=None, version='v1', engine=engine)
