from .basedataset import BaseDataset

import pandas as pd
import numpy as np
import torch
import nibabel as nb

from typing import List, Dict
from pathlib import Path
from scipy import ndimage
from nilearn.image import smooth_img


class fBIRNFAsMRI(BaseDataset):
    def __init__(self,
                 info_df: pd.DataFrame,
                 target_names: List[str],
                 main_target: str,
                 numpy_root: Path,
                 preprocess: bool,
                 seed: int,
                 num_folds: int,
                 batch_size: int):
        self._comp_ix = [0, 1]
        self._comp_names = ['FA_maps', 'sMRI']        
        self._fa_affine = np.array([[  -1.,    0.,    0.,   90.],
                                    [   0.,    1.,    0., -126.],
                                    [   0.,    0.,    1.,  -72.],
                                    [   0.,    0.,    0.,    1.]])
        
        self._data_shape = (1, 121, 145, 121)
        self._mask_fa = np.load('/data/users1/egeenjaar/atlases/fbirn_fa_mask.npy')
        self._mean_fa = np.load('/data/users1/egeenjaar/atlases/fbirn_fa_mean.npy')
        self._std_fa = np.load('/data/users1/egeenjaar/atlases/fbirn_fa_std.npy')
        self._mask_smri = np.load('/data/users1/egeenjaar/atlases/fbirn_smri_mask.npy')
        self._mean_smri = np.load('/data/users1/egeenjaar/atlases/fbirn_smri_mean.npy')
        self._std_smri = np.load('/data/users1/egeenjaar/atlases/fbirn_smri_std.npy')

        super().__init__(
            info_df=info_df,
            target_names=target_names,
            main_target=main_target,
            numpy_root=numpy_root,
            preprocess=preprocess,
            seed=seed,
            num_folds=num_folds,
            batch_size=batch_size)

    @property
    def data_shape(self):
        # (timesteps, dFNC values)
        return self._data_shape

    @property
    def tasks(self):
        return ['Training']

    @property
    def num_classes(self):
        return 1

    # FNC + sMRI
    def static_preprocess(self, data):
        fa_data, smri_data = data
        fa_factor = [target_size / float(fa_size) for
                     fa_size, target_size
                     in zip((182, 218, 182), self.data_shape[1:])]
        fa_data = ndimage.interpolation.zoom(
            fa_data,
            zoom=fa_factor)
        fa_data -= self._mean_fa
        fa_data /= self._std_fa
        fa_data[~self._mask_fa] = 0.0
        fa_data = fa_data[np.newaxis]
        smri_data -= self._mean_smri
        smri_data /= self._std_smri
        smri_data[~self._mask_smri] = 0.0
        smri_data = smri_data[np.newaxis]
        return smri_data, fa_data