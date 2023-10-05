from .basedataset import BaseDataset

import pandas as pd
import numpy as np

from typing import List
from pathlib import Path
from scipy import ndimage


class fBIRNICAsMRI(BaseDataset):
    def __init__(self,
                 info_df: pd.DataFrame,
                 target_names: List[str],
                 main_target: str,
                 numpy_root: Path,
                 preprocess: bool,
                 seed: int,
                 num_folds: int,
                 batch_size: int):
        #self._comp_ix = [50, 53, 11, 5, 24, 6, 9, 41]
        #self._comp_names = ['CB1', 'CB2', 'SMR', 'THM', 'VIS',
        #                    'TL', 'SML', 'FRONT']
        self._comp_ix = [84, 45, 88, 96, 61, 62, 66, 67]
        self._comp_names = ['SMA', 'THA', 'MiFG2', 'SFG', 
                            'R IFG', 'MTG2', 'PreCG', 'IFG2']
        self._data_shape = (1, 121, 145, 121)
        self._mask_smri = np.load('/data/users1/egeenjaar/atlases/fbirn_smri_mask.npy')
        self._mean_smri = np.load('/data/users1/egeenjaar/atlases/fbirn_smri_mean.npy')
        self._std_smri = np.load('/data/users1/egeenjaar/atlases/fbirn_smri_std.npy')
        self._mask_ica = np.load('/data/users1/egeenjaar/atlases/fbirn_ica_mask.npy').astype(bool)
        self._mean_ica = np.load('/data/users1/egeenjaar/atlases/fbirn_ica_mean.npy')
        self._std_ica = np.load('/data/users1/egeenjaar/atlases/fbirn_ica_std.npy')

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

    # ICA + sMRI static preprocess
    def static_preprocess(self, data):
        ica_data, smri_data = data
        fnc, smri_data = data
        smri_data -= self._mean_smri
        smri_data /= self._std_smri
        smri_data[~self._mask_smri] = 0.0
        smri = smri_data[np.newaxis]
        ica_data = ica_data[..., self._comp_ix]
        ica_data = np.transpose(ica_data, (3, 0, 1, 2))
        ica_data -= self._mean_ica
        ica_data /= self._std_ica
        ica_data[~self._mask_ica] = 0.0
        return smri, ica_data
