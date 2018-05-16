from __future__ import absolute_import, print_function
import h5py
import os
import numpy as np
import os.path as op
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from tqdm import tqdm
from glob import glob
from scipy.io import savemat
from .utils import load_cinfo


data_path = op.join(op.dirname(__file__), 'data')
DATA_LOCS = dict(
    v1=dict(vertices=op.join(data_path, 'vertices_v1.mat'),
            textures=op.join(data_path, 'textures_v1.mat')),
    v2=dict(vertices=op.join(data_path, 'vertices_v2.mat')),
    v2dense=dict(vertices=op.join(data_path, 'vertices_v2dense.mat'))
)

DATA_SHAPES = dict(
    v1=dict(vertices=(4735, 3), textures=(800, 600, 4)),
    v2=dict(vertices=(13780, 3)),
    v2dense=dict(vertices=(32493, 3))
)


class FaceGenerator:
    """ Class to generate new faces.

    Parameters
    ----------
    version : str
        Version of the face-database (possibel: 'v1', 'v2', 'v2dense')
    save_dir : str
        Path to directory to save (intermediate) results to.
    """

    def __init__(self, version='v1', save_dir=None):
        """ Initializes FaceGenerator object. """
        self.version = version
        self.mods = list(DATA_LOCS[version].keys())
        if save_dir is None:
            self.save_dir = op.join(os.getcwd(), 'glm_data')
        else:
            self.save_dir = save_dir

        self.cinfo = None
        self.scodes = None
        self.fdata = dict(data=dict(),
                          nz_mask=dict())
        self.iv_names = None

    def _load_hdf5(self, h5_file, scode, version='v1', mod='vertices'):
        """ Loads a dataset from hdf5 file. """

        f = h5py.File(h5_file)
        data = np.array(f.get('%s/%i/%s' % (version, scode, mod)))

        if data.ndim == 3:  # assume textures
            data = np.rollaxis(np.rollaxis(data, axis=0, start=3), axis=0, start=2)
        else:
            data = data.T

        f.close()
        return data

    def load(self, h5_file=None):
        """ Loads all necessary data (cinfo, shapes, textures). """

        # Load demographic data (and clean it)
        cinfo = load_cinfo(version=self.version)
        cinfo = cinfo[['fm', 'age', 'WC', 'BA', 'EA', 'scode', 'gender']]
        cinfo = cinfo[(cinfo.WC + cinfo.BA + cinfo.EA) == 1]
        cinfo = cinfo[cinfo.gender.isin(['M', 'F'])]
        cinfo = cinfo[cinfo.age.between(0, 100)]
        cinfo = cinfo.dropna(how='any', axis=0)
        self.cinfo = cinfo
        self.scodes = cinfo.scode
        n_codes = len(self.scodes)

        if not op.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        np.save(op.join(self.save_dir, 'scodes.npy'), self.scodes)

        # Load face-data (vertices/shapes)
        if h5_file is None:
            h5_file = op.join(op.dirname(__file__), 'data', 'all_data.h5')
        else:
            if not op.isfile(h5_file):
                raise ValueError("Could not find file %s!" % h5_file)

        # Load in data (vertices/shapes)
        print("\nLoading data ...")
        for mod in self.mods:
            tmp = np.zeros(DATA_SHAPES.get(self.version)[mod] + (n_codes,))
            for i, scode in tqdm(enumerate(self.scodes), desc=mod):
                data = self._load_hdf5(h5_file, scode, self.version, mod=mod)
                tmp[..., i] = data

            # Remove all-zero stuff
            nz_mask = tmp.sum(axis=-1) != 0
            self.fdata['data'][mod] = tmp[nz_mask].T
            self.fdata['nz_mask'][mod] = nz_mask

    def fit_GLM(self, chunks=5):
        """ Fits a GLM to the shape/texture data.

        Parameters
        ----------
        chunks : int
            For large arrays (like the texture data), the GLM cannot be
            appropriately vectorized; `chunks` refers to the number of "splits"
            applied to the data. Only relevant when array > 1GB.
        """

        if self.scodes is None:
            raise ValueError("You didn't load the data yet!")

        # Only select subjects we got the data from
        cinfo = self.cinfo
        # One-hot encode gender
        ohe = OneHotEncoder(sparse=False)
        gender = ohe.fit_transform(cinfo.fm.values[:, np.newaxis])
        ethn = cinfo[['WC', 'BA', 'EA']].values
        age = cinfo.age.values[:, np.newaxis]
        icept = np.ones((age.size, 1))
        X = np.hstack((icept, gender, ethn, age))

        # TO DO: GENERATE INTERACTIONS!

        np.save(op.join(self.save_dir, 'IVs.npy'), X)
        self.iv_names = ['intercept', 'male', 'female', 'WC', 'BA', 'EA', 'age']

        # Now, define the data
        names2write = ['betas', 'residuals', 'DVs']
        print("\nStart GLM fitting ...")
        for mod in self.mods:
            y = self.fdata['data'][mod]

            # We need this for later
            np.save(op.join(self.save_dir, '%s_nzmask.npy' % mod), self.fdata['nz_mask'][mod])

            if y.nbytes > 1e+9:  # If the data is very large,split up in chunks
                if chunks == 1:
                    print("Warning: attempting to fit millions of models at "
                          "once ... Consider increasing `chunks`.")

                iC = 1
                # Loop across chunks
                for tmp_y in tqdm(np.array_split(y, chunks, axis=1), 'chunk'):
                    betas = np.linalg.lstsq(X, tmp_y, rcond=None)[0]
                    yhat = X.dot(betas)
                    residuals = tmp_y - yhat

                    for name, dat in zip(names2write, [betas, residuals, tmp_y]):
                        this_i = str(iC).rjust(3, '0')
                        np.save(op.join(self.save_dir, '%s_%s_chunk%s.npy' % (mod, name, this_i)), dat)
                    iC += 1
            else:
                # Fit the models in one go
                betas = np.linalg.lstsq(X, y, rcond=None)[0]
                yhat = X.dot(betas)
                residuals = y - yhat

                # Write to disk
                for name, dat in zip(names2write, [betas, residuals, y]):
                    np.save(op.join(self.save_dir, '%s_%s.npy' % (mod, name)), dat)

    def run_PCA(self, save_dir=None):
        """ Runs PCA on shape/texture residuals.

        Parameters
        ----------
        save_dir : str
            Path to directory with (intermediate) results. If None, path is
            inferred from self.
        """
        print("")
        if save_dir is None:
            save_dir = self.save_dir

        for mod in self.mods:
            res_files = sorted(glob(op.join(save_dir, '%s_residuals*.npy' % mod)))
            if len(res_files) == 1:
                residuals = np.load(res_files[0])
            elif len(res_files) > 1:
                residuals = np.hstack(([np.load(f) for f in res_files]))
            else:
                raise ValueError("GLM was not yet fit!")

            print("Running PCA on %s ..." % mod)
            pca = PCA(copy=False)
            pca.fit(residuals)  # MemoryError on textures ... downsample?
            resids_pca = pca.transform(residuals)
            np.save(op.join(save_dir, '%s_pca_min.npy' % mod), resids_pca.min(axis=0))
            np.save(op.join(save_dir, '%s_pca_max.npy' % mod), resids_pca.max(axis=0))
            np.save(op.join(save_dir, '%s_pca_mean.npy' % mod), pca.mean_)
            np.save(op.join(save_dir, '%s_pca_comps.npy' % mod), pca.components_)

    def change_property_face(self, scode, age=None, gender=None, ethn=None,
                             save_dir=None):
        """ Changes the property of a given face (scode).

        Parameters
        ----------
        scode : int
            Face ID (code) of face to change
        age : int
            Desired age of face
        gender : str
            Desired gender of face ('M' or 'F')
        ethn : str
            Desired ethnicity of face ('WC', 'BA', or 'EA')
        save_dir : str
            Directory with (intermediate) results
        """

        print("")
        if save_dir is None:
            save_dir = self.save_dir

        idx = self._get_idx_of_scode(scode, save_dir)
        results = dict()
        for mod in self.mods:
            print("Changing property of face (%s) ..." % mod)
            nz_mask = np.load(op.join(save_dir, '%s_nzmask.npy' % mod))
            betas = self._load_chunks(mod, save_dir, 'betas')
            resids = self._load_chunks(mod, save_dir, 'residuals')[idx, :]
            norm_vec = self._generate_design_vector(gender, age, ethn)
            tmp_result = norm_vec[np.newaxis, :].dot(betas) + resids
            tmp = np.zeros(DATA_SHAPES[self.version][mod])
            tmp[nz_mask] = np.squeeze(tmp_result)
            tmp = tmp.reshape(DATA_SHAPES[self.version][mod])
            results[mod] = tmp

        results = {'scode_' + str(scode): results}
        name = 'id-%i_gen-%s_age-%i_eth-%s.mat' % (scode, gender, age, ethn)
        out_path = op.join(save_dir, name)
        savemat(out_path, results)

        return out_path

    def generate_new_face(self, N, age, gender, ethn, save_dir=None):
        """ Generates a new face by randomly synthesizing PCA components,
        applying the inverse PCA transform, and adding the norm.

        Parameters
        ----------
        N : int
            How many new faces should be generated
        age : int
            Desired age of new face
        gender : str
            Desired gender of new face ('M' or 'F')
        ethn : str
            Desired ethnicity of new face ('WC', 'BA', 'EA')
        save_dir : str
            Path to directory with (intermediate) results.
        """

        print("")
        if save_dir is None:
            save_dir = self.save_dir

        to_write = {'face%i' % i: dict() for i in range(N)}
        for mod in self.mods:
            print("Generating new faces (%s) ..." % mod)
            pca_comps = np.load(op.join(save_dir, '%s_pca_comps.npy' % mod))
            pca_means = np.load(op.join(save_dir, '%s_pca_mean.npy' % mod))
            nz_mask = np.load(op.join(save_dir, '%s_nzmask.npy' % mod))
            betas = self._load_chunks(mod, save_dir, 'betas')
            mins = np.load(op.join(save_dir, '%s_pca_min.npy' % mod))
            maxs = np.load(op.join(save_dir, '%s_pca_min.npy' % mod))

            random_data = np.zeros((N, pca_comps.shape[0]))
            for i in range(N):  # this can probably be implemented faster ...
                random_data[i, :] = np.random.uniform(mins, maxs)

            inverted_resids = random_data.dot(pca_comps) + pca_means
            norm_vec = self._generate_design_vector(gender, age, ethn)
            norm = norm_vec[np.newaxis, :].dot(betas)
            final_face_data = norm + inverted_resids

            out_dir = op.join(save_dir, 'new_faces')
            if not op.isdir(out_dir):
                os.makedirs(out_dir)

            for i in range(N):
                tmp = np.zeros(DATA_SHAPES[self.version][mod])
                tmp[nz_mask] = final_face_data[i, :]
                tmp = tmp.reshape(DATA_SHAPES[self.version][mod])
                to_write['genface_%i' % i][mod] = tmp

        to_return = []
        for key, value in to_write.items():
            outname = op.join(out_dir, key + '.mat')
            savemat(outname, value)
            to_return.append(outname)

        return to_return

    def _load_chunks(self, mod, save_dir, idf):
        """ Loads data which may be in chunks.

        Parameters
        ----------
        mod : str
            Modality of requested data ('textures' or 'vertices')
        save_dir : str
            Path to directory with (intermediate) results.
        idf : str
            Identifier for files (e.g., 'betas' or 'residuals')

        Returns
        -------
        out : numpy array
            Array with data (chunked data is stacked)
        """

        files = sorted(glob(op.join(save_dir, '%s_%s*.npy' % (mod, idf))))

        if len(files) == 1:
            out = np.load(files[0])
        elif len(files) > 1:
            out = np.hstack(([np.load(f) for f in files]))
        else:
            raise ValueError("Could not find files with identifier '%s'!" % idf)

        return out

    def _get_idx_of_scode(self, scode, save_dir=None):
        """ Returns index of scode. """

        if save_dir is None:
            save_dir = self.save_dir

        scodes = np.load(op.join(save_dir, 'scodes.npy'))
        return scode == scodes

    def _generate_design_vector(self, gender, age, ethn):
        """ Generates a 'design vector' (for lack of a better word). """
        mapping = dict(WC=[1, 0, 0], BA=[0, 1, 0], EA=[0, 0, 1])
        gender = [0, 1] if gender == 'F' else [1, 0]
        des_vec = np.array([1] + gender + mapping[ethn] + [age])
        des_vec = self._add_interactions(des_vec)

        return des_vec

    def _add_interactions(self, X):
        """ Adds interaction terms to X. """
        pnf = PolynomialFeatures(interaction_only=True)
        return pnf.fit_transform(X)
