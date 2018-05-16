from __future__ import print_function, absolute_import
import os
import os.path as op
import numpy as np
import matlab as mlab
import matlab.engine as mlab_eng
from .utils import load_cinfo, get_all_au_labels, get_scodes_given_criteria


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
        face_id = get_scodes_given_criteria(gender, age, age_range, ethnicity,
                                            version='v1')
        if len(face_id) == 0:
            raise ValueError("""No faces available with parameters:
            gender: %s
            age: %i (+/- %i)
            ethn: %s
            version: %s""" % (gender, age, age_range, ethnicity, version))
        else:
            # draw random
            face_id = np.random.choice(face_id.tolist(), 1)
    else:
        if face_id not in cinfo.scode.values:
            raise ValueError("Face-id '%s' not in scodes!" % str(face_id))

    if engine is None:
        print("Starting Matlab ...", end="")
        engine = mlab_eng.start_matlab()
        mlab_path = op.join(op.dirname(__file__))
        engine.addpath(mlab_path, nargout=0)
        print(" done.")

    if nfdata is not None:
        if 'id-g' in nfdata:
            print("Animating random face using %s ... " % op.basename(nfdata))
        else:
            print("Animating face-id (%s) using %s ... " % (str(face_id), op.basename(nfdata)))
    else:
        print("Animating face-id (%s) ... " % str(face_id))

    face_id_mlab = mlab.double([face_id])

    engine.run_GFG(face_id_mlab, nfdata, save_path, au_labels, temp_params, eye_params,
                   head_params, version)
