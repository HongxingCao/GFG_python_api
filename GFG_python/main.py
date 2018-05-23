from __future__ import print_function, absolute_import
import os
import yaml
import os.path as op
import numpy as np
import matlab as mlab
import matlab.engine as mlab_eng
from .utils import (load_cinfo, get_all_au_labels, get_scodes_given_criteria,
                    get_info_given_scode)


def run(face_id=None, nfdata=None, save_path=None, au_labels=None,
        temp_params=None, eye_params=None, head_params=None, gender='M',
        ethnicity='WC', age=None, age_range=0, version='v1', engine=None):
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

    info_dict = dict()

    if save_path is None:
        save_path = os.getcwd()

    if au_labels is None:
        au_labels = ['25-12', '9', '4']
    else:
        if isinstance(au_labels, np.ndarray):
            au_labels = au_labels.tolist()

    all_au_labels = get_all_au_labels()
    for lab in au_labels:
        if lab not in all_au_labels:
            raise ValueError("AU-label '%s' not in set!" % lab)

    info_dict['AUs'] = au_labels

    if temp_params is None:
        N_aus = len(au_labels)
        default_params = np.array([1, 0.5, 0.5, 0.5, 0.5, 0.5])
        temp_params = np.tile(default_params, N_aus).reshape((N_aus, 6))

    info_dict['temp_params'] = _listify(temp_params, make_mlab=False)
    temp_params = _listify(temp_params, make_mlab=True)

    if eye_params is None:
        eye_params = [0.5, 0.5, 0.5, 0]

    info_dict['eye_params'] = _listify(eye_params, make_mlab=False)
    eye_params = _listify(eye_params, make_mlab=True)

    if head_params is None:
        head_params = 0

    info_dict['head_params'] = head_params

    if nfdata is None:
        nfdata = 0

    info_dict['nfdata'] = nfdata

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
            face_id = np.random.choice(face_id.tolist(), 1)[0]
            face_info = get_info_given_scode(face_id)
            info_dict.update(face_info)
    else:
        if face_id not in cinfo.scode.values:
            raise ValueError("Face-id '%s' not in scodes!" % str(face_id))

    info_dict['face_id'] = _listify(face_id, make_mlab=False, convert2int=True)

    if engine is None:
        print("Starting Matlab ...", end="")
        engine = mlab_eng.start_matlab()
        mlab_path = op.join(op.dirname(__file__))
        engine.addpath(mlab_path, nargout=0)
        print(" done.")

    if nfdata != 0:
        if 'id-g' in nfdata:
            print("Animating random face using %s ... " % op.basename(nfdata))
        else:
            print("Animating face-id (%s) using %s ... " % (str(face_id), op.basename(nfdata)))
        save_path = op.join(save_path, op.basename(nfdata).split('.')[0])
    else:
        print("Animating face-id (%s) ... " % str(face_id))
        save_path = op.join(save_path, 'id-' + str(face_id) + '_AUs-' + '-'.join(au_labels))

    if not op.isdir(save_path):
        os.makedirs(save_path)

    info_dict['save_path'] = save_path
    with open(op.join(save_path, 'params.yml'), 'w') as f:
        yaml.dump(info_dict, f)

    face_id = _listify(face_id, make_mlab=True, convert2int=True)

    engine.run_GFG(face_id, nfdata, save_path, au_labels, temp_params,
                   eye_params, head_params, version)


def _listify(arr, make_mlab=False, convert2int=False):
    """ Makes sure array is ready to be passed to matlab. """
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()

    if not isinstance(arr, list):
        arr = [arr]

    if convert2int:
        arr = [int(a) if type(a).__module__ == 'numpy' else a for a in arr]

    if make_mlab:
        arr = mlab.double(arr)

    if len(arr) == 1:
        return arr[0]
    else:
        return arr
