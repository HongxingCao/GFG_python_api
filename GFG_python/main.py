from __future__ import print_function
import os
import os.path as op
import matlab as mlab
import matlab.engine as mlab_eng
from utils import load_cinfo, get_all_au_labels


def run(face_id=None, nfdata=None, save_path=None, au_labels=None, temp_params=None,
        eye_params=None, head_params=None, gender='M', ethnicity='WC',
        age=None, age_range=0, version='v1', engine=None):
    """ Main API to run face-animation with the GFG-toolbox.

    Parameters
    ----------
    face_id : int or list[int]
        Integer (or list of integers) referring to the face-ids
    save_path : str
        Path to directory to which the results should be saved
    au_labels : list[str]
        List with AUs
    temp_params : numpy array
        An N (AUs) x 6 array with temporal parameters
    eye_params : numpy array
        Array with parameters for eye-movements
    head_params : numpy array
        Array with head movements (roll, pitch, yaw)
    gender : str
        Gender of faces ('M' or 'F')
    ethnicity : str
        Ethnicity of faces ('WC', 'BA', 'EA')
    age : int
        Age of faces
    age_range : int
        Range of ages (age +/- age-range)
    version : str
        Version of data ('v1', 'v2', 'v2dense')
    """

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

    if nfdata is None:
        nfdata = 0

    cinfo = load_cinfo()
    if face_id is None:
        age_up, age_down = age + age_range, age - age_range
        query = 'gender == @gender & @age_down <= age <= @age_up'
        query += ' & %s == 1' % ethnicity

        if 'v2' in version:
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
            version: %s""" % (gender, age, age_range, ethnicity, version))
        else:
            face_id = face_id.tolist()

    if face_id is None:
        face_id = 0

    if not isinstance(face_id, list):
        face_id = [face_id]

    # Just a check if the face-ids are in the scodes
    for fid in face_id:

        if fid not in cinfo.scode.values:
            raise ValueError("Face-id '%s' not in scodes!" % str(fid))

    if engine is None:
        print("Starting Matlab ...", end="")
        engine = mlab_eng.start_matlab()
        mlab_path = op.join(op.dirname(__file__))
        engine.addpath(mlab_path, nargout=0)
        print(" done.")

    face_id_mlab = mlab.double(face_id)
    print("Running face-ids: %s" % (face_id,))
    engine.run_GFG(face_id_mlab, nfdata, save_path, au_labels, temp_params, eye_params,
                   head_params, version)
