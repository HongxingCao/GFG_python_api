from __future__ import print_function
import os
import matlab as mlab
import matlab.engine as mlab_eng
from utils import load_cinfo, get_all_au_labels


def run(face_id=None, save_path=None, au_labels=None, temp_params=None,
        eye_params=None, head_params=None, gender='M', ethnicity='WC',
        age=None, age_range=0, v2=True, dense=True, engine=None):

    if save_path is None:
        save_path = os.getcwd()

    if au_labels is None:
        au_labels = ['25-12', '9', '4']

    all_au_labels = get_all_au_labels()
    for lab in au_labels:
        if lab not in all_au_labels:
            raise ValueError("AU-label '%s' not in set!" % lab)

    if temp_params is None:
        temp_params = mlab.double([[1, 0.5, 0.5, 0.5, 0.5, 0.5],
                                   [1, 0.5, 0.5, 0.5, 0.5, 0.5],
                                   [1, 0.5, 0.5, 0.5, 0.5, 0.5]])

    if eye_params is None:
        eye_params = mlab.double([0.5, 0.5, 0.5, 0])

    if head_params is None:
        head_params = 0

    cinfo = load_cinfo()
    if face_id is None:
        age_up, age_down = age + age_range, age - age_range
        query = 'gender == @gender & @age_down <= age <= @age_up'
        query += ' & %s == 1' % ethnicity

        if v2:
            query += ' & proc_v2 == 1'
        else:
            query += ' & proc == 1'

        filtered = cinfo.query(query)
        face_id = filtered.scode.values

        if len(face_id) == 0:
            raise ValueError("""No faces available with parameters:
            gender: %s
            age: %i (+/- %i)
            ethn: %s
            v2: %s""" % (gender, age, age_range, ethnicity, v2))
        else:
            face_id = face_id.tolist()

    if not isinstance(face_id, list):
        face_id = [face_id]

    # Just a check if the face-ids are in the scodes
    for fid in face_id:

        if fid not in cinfo.scode.values:
            raise ValueError("Face-id '%s' not in scodes!" % str(fid))

    if engine is None:
        print("Starting Matlab ...", end="")
        engine = mlab_eng.start_matlab()
        print(" done.")

    face_id_mlab = mlab.double(face_id)
    print("Running face-ids: %s" % (face_id,))
    engine.run_GFG(face_id_mlab, save_path, au_labels, temp_params, eye_params,
                   head_params, v2, dense)


if __name__ == '__main__':
    run(face_id=None, age=20)
