from __future__ import absolute_import, print_function
import h5py
import os
import os.path as op
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.discriminant_analysis import _cov
from scipy.io import savemat
from .utils import load_cinfo, get_scodes_given_criteria

# Some global variables
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
        """ Loads all necessary data (cinfo, shapes, textures).

        Parameters
        ----------
        h5_file : str
            Path to hdf5-file with all data."""

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
        X = self._add_interactions(X)

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

    def run_decomposition(self, algorithm='pca', whiten=False, save_dir=None,
                          **kwargs):
        """ Runs PCA on shape/texture residuals.

        Parameters
        ----------
        algorithm : str
            Decomposition algorithm ('pca', 'nmf', 'ica')
        whiten : bool
            Whether to whiten the data before decomposition (only relevant
            for 'pca' and 'ica')
        save_dir : str
            Path to directory with (intermediate) results. If None, path is
            inferred from self.
        kwargs : dict
            Extra arguments for decomposition algorithm
        """
        print("")
        if save_dir is None:
            save_dir = self.save_dir

        for mod in self.mods:

            residuals = self._load_chunks(mod, save_dir, 'residuals')

            if algorithm == 'ica':
                scaler = StandardScaler()
                residuals = scaler.fit_transform(residuals)
                np.save(op.join(save_dir, '%s_residual_means.npy' % mod), scaler.mean_)
                np.save(op.join(save_dir, '%s_residual_stds.npy' % mod), scaler.scale_)
            if algorithm == 'nmf':
                scaler = MinMaxScaler()
                residuals = scaler.fit_transform(residuals)
                np.save(op.join(save_dir, '%s_residual_mins.npy' % mod), scaler.min_)
                np.save(op.join(save_dir, '%s_residual_scale.npy' % mod), scaler.scale_)
            print("Running decomposition (%s) on %s ..." % (algorithm, mod))

            if algorithm == 'pca':
                decomp = PCA(copy=False, whiten=whiten, **kwargs)
            elif algorithm == 'nmf':
                # n_components set to keep comparable to PCA
                decomp = NMF(n_components=residuals.shape[0], **kwargs)
            elif algorithm == 'ica':
                decomp = FastICA(n_components=residuals.shape[0], **kwargs)
            else:
                raise ValueError("Please choose from 'pca', 'nmf', 'ica'.")

            decomp.fit(residuals)  # MemoryError on textures ... downsample?
            resids_decomp = decomp.transform(residuals)
            np.save(op.join(save_dir, '%s_residuals_decomp.npy' % mod), resids_decomp)

            if algorithm == 'ica':
                # note: this is the mixing matrix (not components!)
                np.save(op.join(save_dir, '%s_decomp_comps.npy' % mod), decomp.mixing_)
            elif algorithm == 'pca':
                np.save(op.join(save_dir, '%s_decomp_means.npy' % mod), decomp.mean_)
                np.save(op.join(save_dir, '%s_decomp_comps.npy' % mod), decomp.components_)
                if whiten:
                    np.save(op.join(save_dir, '%s_decomp_explvar.npy' % mod), decomp.explained_variance_)

            else:  # must be NMF
                np.save(op.join(save_dir, '%s_decomp_comps.npy' % mod), decomp.components_)

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

        if save_dir is None:
            save_dir = self.save_dir

        idx = self._get_idx_of_scode(scode, save_dir)
        results = dict()

        print("")
        for mod in self.mods:
            print("Changing property of face (%s) ..." % mod)
            nz_mask = np.load(op.join(save_dir, '%s_nzmask.npy' % mod))
            betas = self._load_chunks(mod, save_dir, 'betas')
            resids = self._load_chunks(mod, save_dir, 'residuals')[idx, :]
            norm_vec = self._generate_design_vector(gender, age, ethn)
            tmp_result = norm_vec.dot(betas) + resids
            tmp = np.zeros(DATA_SHAPES[self.version][mod])
            tmp[nz_mask] = np.squeeze(tmp_result)
            tmp = tmp.reshape(DATA_SHAPES[self.version][mod])
            results[mod] = tmp

        name = 'id-%i_gen-%s_age-%i_eth-%s.mat' % (scode, gender, age, ethn)
        out_path = op.join(save_dir, name)
        savemat(out_path, results)

        return out_path

    def generate_new_face(self, N, age, gender, ethn, age_range=20, algorithm='pca',
                          dist='normal', whitened=False, shrinkage=False,
                          save_dir=None):
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
        dist : str
            Distribution used to sample new values ('uniform', 'norm', 'mnorm')
        whitened : bool
            Was the data whitened before decomposition?
        shrinkage : bool
            Whether to apply shrinkage to covariance estimation of residuals.
            Only relevant when dist='mnorm'.
        save_dir : str
            Path to directory with (intermediate) results.
        """

        if save_dir is None:
            save_dir = self.save_dir

        to_write = {i: dict() for i in range(N)}
        print("")
        for mod in self.mods:
            print("Generating new faces (%s) ..." % mod)
            decomp_comps = np.load(op.join(save_dir, '%s_decomp_comps.npy' % mod))

            nz_mask = np.load(op.join(save_dir, '%s_nzmask.npy' % mod))
            betas = self._load_chunks(mod, save_dir, 'betas')
            resids_decomp = self._load_chunks(mod, save_dir, 'residuals_decomp')
            relev_scodes = get_scodes_given_criteria(gender, age, age_range, ethn, 'v1')
            idx = self._get_idx_of_scode(relev_scodes)
            relev_resids = resids_decomp[idx, :]
            random_data = np.zeros((N, decomp_comps.shape[0]))
            for i in range(N):  # this can probably be implemented faster ...
                if dist == 'uniform':
                    mins, maxs = relev_resids.min(axis=0), relev_resids.max(axis=0)
                    random_data[i, :] = np.random.uniform(mins, maxs)
                elif dist == 'norm':
                    means, stds = relev_resids.mean(axis=0), relev_resids.std(axis=0)
                    random_data[i, :] = np.random.normal(means, stds)
                elif dist == 'mnorm':
                    means = relev_resids.mean(axis=0)

                    if shrinkage:
                        cov = _cov(relev_resids, shrinkage='auto')
                    else:
                        cov = np.cov(relev_resids.T)

                    random_data[i, :] = np.random.multivariate_normal(means, cov)
                else:
                    raise ValueError("Please choose `dist` from ('uniform', "
                                     "'norm', 'mnorm')")

            # For debugging
            if algorithm == 'pca':
                decomp_means = np.load(op.join(save_dir, '%s_decomp_means.npy' % mod))
                if whitened:
                    decomp_explvar = np.load(op.join(save_dir, '%s_decomp_explvar.npy' % mod))
                    resids_inv = np.dot(random_data, np.sqrt(decomp_explvar[:, np.newaxis]) *
                                        decomp_comps) + decomp_means
                else:
                    resids_inv = random_data.dot(decomp_comps) + decomp_means
            elif algorithm == 'ica':
                resids_inv = random_data.dot(decomp_comps.T)
                resid_means = np.load(op.join(save_dir, '%s_residual_means.npy' % mod))
                resid_stds = np.load(op.join(save_dir, '%s_residual_stds.npy' % mod))
                resids_inv *= resid_stds
                resids_inv += resid_means
            elif algorithm == 'nmf':
                resids_inv = random_data.dot(decomp_comps)
                resid_mins = np.load(op.join(save_dir, '%s_residual_mins.npy' % mod))
                resid_scale = np.load(op.join(save_dir, '%s_residual_scale.npy' % mod))
                resids_inv -= resid_mins
                resids_inv /= resid_scale

            norm_vec = self._generate_design_vector(gender, age, ethn)
            norm = norm_vec.dot(betas)
            final_face_data = norm + resids_inv
            for i in range(N):
                tmp = np.zeros(DATA_SHAPES[self.version][mod])
                tmp[nz_mask] = final_face_data[i, :]
                tmp = tmp.reshape(DATA_SHAPES[self.version][mod])
                to_write[i][mod] = tmp

        to_return = []
        for key, value in to_write.items():
            name = 'id-g%i_gen-%s_age-%i_eth-%s.mat' % (key, gender, age, ethn)
            outname = op.join(save_dir, name)
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

        if not isinstance(scode, np.ndarray):
            scode = np.array(scode)

        if save_dir is None:
            save_dir = self.save_dir

        all_scodes = np.load(op.join(save_dir, 'scodes.npy'))
        return np.isin(all_scodes, scode)

    def _generate_design_vector(self, gender, age, ethn):
        """ Generates a 'design vector' (for lack of a better word). """
        mapping = dict(WC=[1, 0, 0], BA=[0, 1, 0], EA=[0, 0, 1])
        gender = [0, 1] if gender == 'F' else [1, 0]
        des_vec = np.array([1] + gender + mapping[ethn] + [age])[np.newaxis, :]
        des_vec = self._add_interactions(des_vec)

        return des_vec

    def _add_interactions(self, X):
        """ Adds interaction terms to X. """
        pnf = PolynomialFeatures(interaction_only=True)
        return pnf.fit_transform(X)
