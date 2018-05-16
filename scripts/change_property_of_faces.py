from GFG_python.main import run
from GFG_python import FaceGenerator
from matlab import engine as mlab_eng

engine = mlab_eng.start_matlab()

fg = FaceGenerator(version='v1')

for g in ['M', 'F']:
    for e in ['EA', 'WC', 'BA']:
        for a in [20, 40, 60]:
            out = fg.change_property_face(2, age=a, gender=g, ethn=e)

            run(face_id=2, nfdata=out, save_path='/Users/lukas/Desktop/testGFG',
                au_labels=None, temp_params=None, eye_params=None,
                head_params=None, version='v1')
