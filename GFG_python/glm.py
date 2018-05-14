import numpy as np
import os.path as op
import h5py
from scipy.io import loadmat, savemat
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from utils import load_cinfo

data_path = op.join(op.dirname(__file__), 'data')
DATA_LOCS = dict(
    v1=dict(vertices=op.join(data_path, 'vertices_v1.mat'),
            textures=op.join(data_path, 'textures_v1.mat')),
    v2=dict(vertices=op.join(data_path, 'vertices_v2.mat')),
    v2dense=dict(vertices=op.join(data_path, 'vertices_v2dense.mat'))
)


class FaceGenerator:
    """ Class to generate new faces.

    Parameters
    ----------
    version : str
        Version of the face-database (possibel: 'v1', 'v2', 'v2dense')

    Attributes
    ----------

    """

    def __init__(self, version='v1'):
        """ Initializes FaceGenerator object. """
        self.version = version
        self.cinfo = None
        self.fdata = dict(data=dict(),
                          shape=dict(),
                          nz_mask=dict())

    def load(self):
        """ Loads all necessary data (cinfo, shapes, textures). """
        self.cinfo = load_cinfo(version=self.version)
        v = loadmat(DATA_LOCS[self.version]['vertices'])
        v = v['vertices_%s' % self.version]
        nz_mask = v.sum(axis=-1) != 0
        self.fdata['data']['vertices'] = v[:, nz_mask]
        self.fdata['nz_mask']['vertices'] = nz_mask
        self.fdata['shape']['vertices'] = v.shape

        if self.version == 'v1':
            with h5py.File(DATA_LOCS[self.version]['textures'], 'r') as f:
                t = np.array(f.get('textures_v1'))
                nz_mask = t.sum(axis=-1) != 0
                self.fdata['data']['textures'] = t[:, nz_mask]
                self.fdata['nz_mask']['textures'] = nz_mask
                self.fdata['shape']['vertices'] = v.shape

            self.fdata['textures']
    def fit_GLM(age=True, ethn=True, gender=True):
        """ Fits a GLM to the shape/texture data. """

        # TO DO: FILTER MODEL WHEN NO ETHN!
        data_orig_shape = data.shape
        y = data.reshape(-1, data.shape[-1]).T
        y_orig_shape = y.shape
        nonzero_y = y.sum(axis=0) != 0
        y = y[:, nonzero_y]

        ohe = OneHotEncoder(sparse=False)
        gender = ohe.fit_transform(cinfo.fm.values[:, np.newaxis])
        ethn = cinfo[['WC', 'BA', 'EA']].values
        age = cinfo.age.values[:, np.newaxis]
        icept = np.ones((age.size, 1))
        X = np.hstack((icept, gender, ethn, age))

        # Fit model
        betas = np.linalg.lstsq(X, y, rcond=None)[0]
        yhat = X.dot(betas)
        SStot = np.sum((y - y.mean(axis=0))**2, axis=0)
        SSres = np.sum((y - yhat)**2, axis=0)
        r2scores = 1 - SSres / SStot
        residuals = y - yhat
        print(r2scores.max())
        print(r2scores.mean())

        pca = PCA()
        residuals_pca = pca.fit_transform(residuals)
        inverted_pca = pca.inverse_transform(residuals_pca)
        #mean_pca, std_pca = residuals_pca.mean(axis=0), residuals_pca.std(axis=0)
        #random_v = np.random.normal(mean_pca, std_pca)
        #new_V = pca.inverse_transform(random_v)

        to_save = np.zeros(y_orig_shape[1])
        to_save[nonzero_y] = new_V
        to_save = to_save.reshape(data_orig_shape[:2])

        # icept, gender, ethn, age
        norm = np.zeros(y_orig_shape[1])
        norm_cfg = np.array([1, 1, 0, 1, 0, 0, 70])[np.newaxis, :]
        norm[nonzero_y] = np.squeeze(norm_cfg.dot(betas))
        norm = norm.reshape(data_orig_shape[:2])
        to_save += norm
        savemat('/Users/lukas/desktop/test.mat', dict(V=to_save))


fg = FaceGenerator(version='v1')
fg.load()
