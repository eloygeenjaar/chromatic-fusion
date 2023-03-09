from .generator import CatalystDALIGenericIterator

import nvidia.dali.fn as fn
import numpy as np
import pandas as pd
import nibabel as nb
import json
import torch
import ast

from nvidia.dali.plugin.pytorch import DALIGenericIterator as dgi
from pathlib import Path
from typing import List, Tuple, Dict
from nvidia import dali
from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from catalyst.contrib.utils.pandas import split_dataframe_on_stratified_folds

class DS(object):
    def __init__(self, num_subjects, mean=None):
        self._num_subjects = num_subjects
    
    def __len__(self):
        return self._num_subjects

class DALIGenericIterator(dgi):
    def __init__(self, pipe, num_subjects, *args, **kwargs):
        super().__init__(pipe, *args, **kwargs)
        self._dataset = DS(num_subjects)
    
    @property
    def dataset(self):
        return self._dataset
    
    @property
    def sampler(self):
        return None

class BaseDataset(object):
    def __init__(self,
                 info_df: pd.DataFrame, 
                 target_names: List[str],
                 main_target: str,
                 numpy_root: Path,
                 preprocess: bool,
                 seed: int,
                 num_folds: int,
                 batch_size: int,
                 *args, **kwargs):
        self._numpy_root = numpy_root
        self._info_df = info_df
        self._target_names = target_names
        self._main_target = main_target
        self._preprocess = preprocess
        self._seed = seed
        self.num_folds = num_folds
        self._batch_size = batch_size
        self._fold_means = [None] * num_folds
        self._fold_vars = [None] * num_folds
        
        self._dfs = []
        
        # Assume numbers are 32-bit/4 bytes
        self._num_bytes_data = np.product(self.data_shape) * 4
        
        
        self.generate_numpy_files()
        self._folds = self.create_folds()
        self._pipelines = self.create_pipelines()
    
    #TODO: 
    # - Think of a good way to integrate into Catalyst (train_step or something, or on_epoch_end for pipe.reset())
    # - Make sure multiple pipelines do not exist at once
    @staticmethod
    @dali.pipeline_def(batch_size=4, num_threads=4, device_id=0, set_affinity=True, seed=42)
    def dali_pipeline(m1_paths, m2_paths, target_paths, num_data_bytes):
        m1 = fn.readers.numpy(bytes_per_sample_hint=num_data_bytes, files=m1_paths, device='cpu',
                              prefetch_queue_depth=2, read_ahead=True, tensor_init_bytes=num_data_bytes,
                              name='Modality 1 reader').gpu()
        m2 = fn.readers.numpy(bytes_per_sample_hint=num_data_bytes, files=m2_paths, device='cpu',
                                prefetch_queue_depth=2, read_ahead=True, tensor_init_bytes=num_data_bytes,
                                name='Modality 2 reader').gpu()
        targets = fn.readers.numpy(bytes_per_sample_hint=4, files=target_paths, device='cpu',
                                   prefetch_queue_depth=2, read_ahead=True, tensor_init_bytes=4,
                                   name='Target reader').gpu()
        return m1, m2, targets
    
    def create_pipelines(self):
        pipelines = []
        for fold in self._folds:
            train_subjects, train_targets = fold['train']
            valid_subjects, valid_targets = fold['valid']
            test_subjects, test_targets = fold['test']
            train_m1 = [paths[0] for paths in train_subjects]
            train_m2 = [paths[1] for paths in train_subjects]
            valid_m1 = [paths[0] for paths in valid_subjects]
            valid_m2 = [paths[1] for paths in valid_subjects]
            
            
            train_pipe = self.dali_pipeline(batch_size=self._batch_size,
                                            seed=self._seed,
                                            m1_paths=train_m1,
                                            m2_paths=train_m2,
                                            target_paths=train_targets,
                                            num_data_bytes=self._num_bytes_data)
            valid_pipe = self.dali_pipeline(batch_size=self._batch_size,
                                            seed=self._seed,
                                            m1_paths=valid_m1,
                                            m2_paths=valid_m2,
                                            target_paths=valid_targets,
                                            num_data_bytes=self._num_bytes_data)
            if test_subjects and test_targets:
                test_m1 = [paths[0] for paths in test_subjects]
                test_m2 = [paths[1] for paths in test_subjects]
                test_pipe = self.dali_pipeline(batch_size=self._batch_size,
                                               seed=self._seed,
                                               m1_paths=test_m1,
                                               m2_paths=test_m2,
                                               target_paths=test_targets,
                                               num_data_bytes=self._num_bytes_data)
            else:
                test_pipe = None
            pipelines.append((train_pipe, valid_pipe, test_pipe))
        return pipelines
    
    def build_pipes(self, fold):
        train_pipe, valid_pipe, test_pipe = self._pipelines[fold]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        train_pipe.build()
        valid_pipe.build()
        if test_pipe is not None:
            test_pipe.build()
            test_loader = DALIGenericIterator(test_pipe, num_subjects=self._num_subjects[fold]['test'],
                                              output_map=['m1', 'm2', 'targets'], reader_name="Modality 1 reader",
                                              fill_last_batch=False)
        else:
            test_loader = None
        loaders = {
            'train': DALIGenericIterator(train_pipe, num_subjects=self._num_subjects[fold]['train'],
                                                 output_map=['m1', 'm2', 'targets'], reader_name="Modality 1 reader",
                                                 fill_last_batch=False),
            'valid': DALIGenericIterator(valid_pipe, num_subjects=self._num_subjects[fold]['valid'],
                                                 output_map=['m1', 'm2', 'targets'], reader_name="Modality 1 reader",
                                                 fill_last_batch=False),
            'test':  test_loader
        }
        return loaders, (train_pipe, valid_pipe, test_pipe)
    
    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def tasks(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError
    
    @property
    def folds(self):
        return self._folds
    
    @property
    def dataframes(self):
        return self._dfs
    
    @property
    def dataloaders(self):
        return self._dataloaders
    
    @property
    def fold_means(self):
        return self._fold_means
    
    @property
    def fold_vars(self):
        return self._fold_vars
    
    def static_preprocess(self, data):
        raise NotImplementedError
    
    # Add contents file that describes contents
    # Add symlink if file is already numpy
    def _file_check(self, p: Path):
        if not p.is_dir():
            return False
        elif not (p / Path('m1.npy')).is_file():
            return False
        elif not (p / Path('m2.npy')).is_file():
            return False
        elif not (p / Path('.file_contents.json')).is_file():
            return False
        else:
            checks = True
            with (p / Path('.file_contents.json')).open('r') as f:
                fc = json.load(f)
                checks &= fc['target_names'] == self._target_names
                checks &= tuple(fc['data_shape']) == self.data_shape
            return checks
        
    @staticmethod
    def _recast(data: np.ndarray):
        if issubclass(data.dtype.type, np.integer):
            # int32 is 4 bytes in size, if int64 -> recast to int32
            # the dataloader will recast the array anyways and this will save disk space
            if data.itemsize > 4:
                data = data.astype(np.int32)
        elif issubclass(data.dtype.type, np.floating):
            # float32 is 4 bytes in size, if float64 -> recast to float32
            # the dataloader will recast the array anyways and this will save disk space
            if data.itemsize > 4:
                data = data.astype(np.float32)
        else:
            raise NotImplementedError(f'Dtype {data.dtype} of numpy array {orig_path} is not supported')
        return data
    
    def _check_numpy_files(self, all_subjects):
        to_be_generated = [(p, idc) for (p, idc) in all_subjects if not self._file_check(p)]
        return to_be_generated
    
    def _create_data_file(self, orig_paths: Path, new_paths: Path):
        # Create symlink if file is already numpy file
        data_ls = []
        for (orig_path, new_path) in zip(orig_paths, new_paths):
            orig_path = Path(orig_path)
            if orig_path.suffix == '.npy' or orig_path.suffix == '.npz':
                if self._preprocess:
                    data = np.load(orig_path)
                    data = self._recast(data)
                    data_ls.append(data)
                else:
                    orig_path.symlink_to(new_path,
                                        target_is_directory=False)
                    return
            elif orig_path.suffix == '.nii' or orig_path.suffixes == ['.nii', '.gz']:
                data = nb.load(orig_path).get_fdata()
                data = self._recast(data)
                data_ls.append(data)
            elif orig_path.suffix == '.mat':
                data = loadmat(orig_path)['sFNC'].astype(np.float32)
                data_ls.append(data)
            else:
                raise NotImplementedError(f'File type {orig_path.suffixes} for path {orig_path} is not supported')
        if self._preprocess:
            m1, m2 = self.static_preprocess(data_ls)
        else:
            data = data_ls[0]
        np.save(new_paths[0], m1)
        np.save(new_paths[1], m2)

    def _create_target_file(self, new_path: Path, y):
        target_df = self._info_df[self._target_names].copy()
        # This command creates a Numpy array that can be indexed into like a pandas dataframe/dictionary
        # The first line creates a list of tuples for each of the rows in the dataframe and casts it to a numpy array
        #targets = np.array(list(target_df.itertuples(index=False)),
        #                   # This line makes sure that the dtypes are named after the columns in the df, which allows
        #                   # for the indexing based on the column names. It zips the column names + the dtype per column
        #                   # to make sure everything is cast correctly when the file is saved.
        #                   dtype=list(zip(list(target_df), target_df.dtypes)))
        np.save(new_path, y)
    def _create_content_file(self, orig_path: Path, new_path: Path, idc):
        file_content_dict = {
            'original_path': str(orig_path),
            'target_names': self._target_names,
            'data_shape': self.data_shape,
            'idc': str(idc)
        }
        
        with new_path.open('w') as f:
            json.dump(file_content_dict, f)
        
    def generate_numpy_files(self):
        all_subjects = [(self._numpy_root / Path(str(idc)), idc) for idc in self._info_df['idc']]
        to_be_generated = self._check_numpy_files(all_subjects)
        for (subject_path, idc) in to_be_generated:
            subject_path.mkdir(parents=True, exist_ok=True)
            m1_path = subject_path / Path('m1.npy')
            m2_path = subject_path / Path('m2.npy')
            target_path = subject_path / Path('target.npy')
            file_content_path = subject_path / Path('.file_contents.json')
            
            orig_files = self._info_df.loc[ (self._info_df['idc'] == idc), 'file_paths'].values[0]
            orig_files = orig_files.replace('(', '').replace(')', '').replace("'", '').split(', ')
            orig_files = [Path(str(p)) for p in orig_files]
            y = int(self._info_df.loc[ (self._info_df['idc'] == idc), self._main_target].values[0])
            self._create_data_file(orig_files, (m1_path, m2_path))
            self._create_target_file(target_path, y)
            self._create_content_file(orig_files, file_content_path, idc)
        
        # TODO: Clean this up, this can be done more elegantly
        for (subject_path, idc) in all_subjects:  
            m1_path = str((subject_path / Path('m1.npy')).resolve())
            m2_path = str((subject_path / Path('m2.npy')).resolve())
            target_path = subject_path / Path('target.npy')
            self._info_df.loc[ (self._info_df['idc'] == idc), 'data_path'] = str((m1_path, m2_path))
            self._info_df.loc[ (self._info_df['idc'] == idc), 'target_path'] = target_path
            
    def create_folds(self) -> List[Dict[str, Tuple[List, List]]]:
        self._num_subjects = []
        
        if self.num_folds == 0:
            self._info_df['fold'] = 0
        else:
            self._info_df = split_dataframe_on_stratified_folds(self._info_df,
                                                                class_column=self._main_target,
                                                                random_state=self._seed, 
                                                                n_folds=self.num_folds)
            train_df = self._info_df.loc[self._info_df['fold'] != 0, :].copy()
            test_df = self._info_df.loc[self._info_df['fold'] == 0, :].copy()
        self._info_df['targets'] = self._info_df[self._main_target]
        
        folds = []
        
        if self.num_folds == 0:
            train_df = self._info_df.loc[self._info_df['fold'] == 0, :].copy()
            x_train, x_val, _, _ = train_test_split(train_df, train_df['targets'], train_size=0.9, 
                                                    random_state=self._seed, stratify=True)
            self._dfs.append({
                'train': x_train,
                'valid': x_val,
                'test': x_test
            })
            cur_fold = {'train': (x_train['data_path'].tolist(), x_train['target_path'].tolist()),
                        'valid': (x_val['data_path'].tolist(), x_val['target_path'].tolist()),
                        'test': ([], [])}
            folds.append(cur_fold)
            num_subject = {'train': len(cur_fold['train'][0]),
                           'valid': len(cur_fold['valid'][0]),
                           'test': len(cur_fold['test'][0])}
            self._num_subjects.append(num_subject)
            
            
        for fold in range(self.num_folds):
            train_df = self._info_df.loc[self._info_df['fold'] != fold, :].copy()
            x_test = self._info_df.loc[self._info_df['fold'] == fold, :].copy()
            x_train, x_val, _, _ = train_test_split(train_df, train_df['targets'], train_size=0.9, 
                                                    random_state=self._seed, stratify=train_df['targets'])
            self._dfs.append({
                'train': x_train,
                'valid': x_val,
                'test': x_test
            })
            train_paths = [eval(tuple_path) for tuple_path in x_train['data_path'].tolist()]
            val_paths = [eval(tuple_path) for tuple_path in x_val['data_path'].tolist()]
            test_paths = [eval(tuple_path) for tuple_path in x_test['data_path'].tolist()]
            cur_fold = {'train': (train_paths, x_train['target_path'].tolist()),
                        'valid': (val_paths, x_val['target_path'].tolist()),
                        'test':  (test_paths, x_test['target_path'].tolist())}
            folds.append(cur_fold)
            num_subject = {'train': len(cur_fold['train'][0]),
                           'valid': len(cur_fold['valid'][0]),
                           'test': len(cur_fold['test'][0])}
            self._num_subjects.append(num_subject)
        
        if hasattr(self, '_calc_fold_statistics'):
            self._calc_fold_statistics(folds)
        
        return folds
