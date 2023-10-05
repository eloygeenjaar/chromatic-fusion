from .basedataset import BaseDataset

import pandas as pd
import numpy as np

from typing import List
from pathlib import Path
from scipy import ndimage


class fBIRNICAFNC(BaseDataset):
    def __init__(self,
                 info_df: pd.DataFrame,
                 target_names: List[str],
                 main_target: str,
                 numpy_root: Path,
                 preprocess: bool,
                 seed: int,
                 num_folds: int,
                 batch_size: int):
        self._comp_ix = [84, 45, 88, 96, 61, 62, 66, 67]
        self._comp_names = ['SMA', 'THA', 'MiFG2', 'SFG', 
                            'R IFG', 'MTG2', 'PreCG', 'IFG2']
        self._comp_ix_fnc = [
            68, 52, 97, 98, 44,

            20, 55,

            2, 8, 1, 10, 26, 53, 65, 79, 71,

            15, 4, 61, 14, 11, 92, 19, 7, 76,

            67, 32, 42, 69, 60, 54, 62, 78, 83, 95, 87, 47, 80, 36, 66, 37, 82,

            31, 39, 22, 70, 16, 50, 93,

            12, 17, 3, 6]
        self._comp_names_fnc = [
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

        self._mean_fnc = np.load('/data/users1/egeenjaar/atlases/fbirn_fnc_mean.npy')
        self._std_fnc = np.load('/data/users1/egeenjaar/atlases/fbirn_fnc_std.npy')
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
        return (1, 53, 63, 52)

    @property
    def tasks(self):
        return ['Training']

    @property
    def num_classes(self):
        return 1

    # ICA + sFNC static preprocess
    def static_preprocess(self, data):
        ica_data, fnc = data
        ica_data = ica_data[..., self._comp_ix]
        ica_data = np.transpose(ica_data, (3, 0, 1, 2))
        ica_data -= self._mean_ica
        ica_data /= self._std_ica
        ica_data[~self._mask_ica] = 0.0
        fnc = fnc[:, self._comp_ix_fnc].transpose(1, 0)
        triu_indices = np.triu_indices(53, 1)
        fnc = np.corrcoef(fnc)
        fnc = fnc[triu_indices]
        fnc -= self._mean_fnc
        fnc /= self._std_fnc
        return ica_data, fnc
