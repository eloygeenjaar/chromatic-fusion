from .basedataset import BaseDataset

import pandas as pd
import numpy as np
import torch
import nibabel as nb

from typing import List, Dict
from pathlib import Path
from scipy import ndimage
from nilearn.image import smooth_img


class fBIRNFNCFA(BaseDataset):
    def __init__(self,
                 info_df: pd.DataFrame,
                 target_names: List[str],
                 main_target: str,
                 numpy_root: Path,
                 preprocess: bool,
                 seed: int,
                 num_folds: int,
                 batch_size: int):
        self._comp_ix = [
            68, 52, 97, 98, 44,

            20, 55,

            2, 8, 1, 10, 26, 53, 65, 79, 71,

            15, 4, 61, 14, 11, 92, 19, 7, 76,

            67, 32, 42, 69, 60, 54, 62, 78, 83, 95, 87, 47, 80, 36, 66, 37, 82,

            31, 39, 22, 70, 16, 50, 93,

            12, 17, 3, 6]
        self._comp_names = [
            'CAU1', 'SUB/HYPOT', 'PUT', 'CAU2', 'THA',

            'STG', 'MTG1',

            'PoCG1', 'L PoCG', 'ParaCL1', 'R PoCG', 'SPL1',
            'ParaCL2', 'PreCG', 'SPL', 'PoCG2',

            'CalcarineG', 'MOG', 'MTG2', 'CUN', 'R MOG',
            'FUG', 'IOG', 'LingualG', 'MTG3',

            'IPL1', 'INS', 'SMFG', 'IFG1', 'R IFG', 'MiFG1',
            'IPL2', 'R IPL', 'SMA', 'SFG', 'MiFG2', 'HiPP1',
            'L IPL', 'MCC', 'IFG2', 'MiFG3', 'HiPP2',

            'Pr1', 'Pr2', 'ACC1', 'PCC1', 'ACC2', 'Pr3', 'PCC2',

            'CB1', 'CB2', 'CB3', 'CB4']
        
        self._data_shape = (1, 121, 145, 121)
        self._mask_fa = np.load('/data/users1/egeenjaar/atlases/fbirn_fa_mask.npy')
        self._mean_fa = np.load('/data/users1/egeenjaar/atlases/fbirn_fa_mean.npy')
        self._std_fa = np.load('/data/users1/egeenjaar/atlases/fbirn_fa_std.npy')
        self._mean_fnc = np.load('/data/users1/egeenjaar/atlases/fbirn_fnc_mean.npy')
        self._std_fnc = np.load('/data/users1/egeenjaar/atlases/fbirn_fnc_std.npy')

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
        fnc, fa_data = data
        fa_factor = [target_size / float(fa_size) for
                     fa_size, target_size
                     in zip((182, 218, 182), self.data_shape[1:])]
        fa_data = ndimage.interpolation.zoom(
            fa_data,
            zoom=fa_factor)
        fa_data -= self._mean_fa
        fa_data /= self._std_fa
        fa_data[~self._mask_fa] = 0.0
        print(fa_data[self._mask_fa].min(), fa_data[self._mask_fa].max(), (fa_data == 0.0).sum(), np.product(fa_data.shape))
        fa_data = fa_data[np.newaxis]
        fnc = fnc[:, self._comp_ix].transpose(1, 0)
        triu_indices = np.triu_indices(53, 1)
        fnc = np.corrcoef(fnc)
        fnc = fnc[triu_indices]
        fnc -= self._mean_fnc
        fnc /= self._std_fnc
        return fa_data, fnc
