import GFG_python
import os.path as op
from GFG_python.main import run
from GFG_python.glm import FaceGenerator
from matlab import engine as mlab_eng

engine = mlab_eng.start_matlab()
engine.addpath(op.dirname(GFG_python.__file__), nargout=0)
fg = FaceGenerator(version='v1', save_dir='/Users/lukas/software/GFG_glm')

for g in ['M', 'F']:
    for e in ['EA', 'WC', 'BA']:
        for a in [20, 40, 60]:
            out = fg.change_property_face(2, age=a, gender=g, ethn=e)

            run(face_id=2, nfdata=out, save_path='/Users/lukas/Desktop/testGFG',
                au_labels=None, temp_params=None, eye_params=None,
                head_params=None, version='v1', engine=engine)
