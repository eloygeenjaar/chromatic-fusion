from .basetask import BaseTask
import torch
import time
import importlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import seaborn as sns
import pandas as pd
import nibabel as nb
import numpy as np
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from catalyst.data.loader import BatchPrefetchLoaderWrapper
from catalyst import dl
#from catalyst.contrib.nn import optimizers as opt
from catalyst.callbacks.misc import TqdmCallback
from torch import nn
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from chromatic.runners import DALICallback
from catalyst.contrib.nn.schedulers import OneCycleLRWithWarmup
from torch import optim as opt


class Training(BaseTask):
    def __init__(self,
                 model,
                 batch_size: int,
                 dataset_generator: List[Tuple[Dataset, Dataset, Dataset]],
                 criterion,
                 learning_rate: float,
                 epochs: int,
                 checkpoint_path,
                 num_folds: int,
                 device,
                 logdir,
                 current_fold):
        super().__init__(model=model,
                         dataset_generator=dataset_generator,
                         logdir=logdir / Path(f'fold_{current_fold}'))

        self._criterion = criterion
        self._batch_size = batch_size
        self._device = device
        self._num_folds = num_folds
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._checkpoint_path = checkpoint_path
        self._current_fold = current_fold

        self._callbacks = {
            "checkpoint": dl.CheckpointCallback(
                self._logdir, loader_key="valid",
                metric_key="loss", minimize=True),
            "_verbose": TqdmCallback(),
            "dali": DALICallback()}
        tasks_module = importlib.import_module('chromatic.runners')
        self._runner_class = getattr(tasks_module, f'{model}Runner')

    def run(self):

        if self._current_fold != 0:
            time.sleep(4)

        torch.backends.cudnn.benchmark = False
        if not self._logdir.is_dir():
            self._logdir.mkdir(parents=True, exist_ok=True)

        if self._checkpoint_path is None:
            checkpoint_path = self._logdir.parent / Path('init_state.pth')
        else:
            checkpoint_path = self._checkpoint_path

        scores = []

        loaders, pipes = self._dataset_generator.build_pipes(
            self._current_fold)

        if self._current_fold == 0 and (self._checkpoint_path is None):
            torch.save(self._model.state_dict(), checkpoint_path)
        elif self._checkpoint_path is None:
            self._model.load_state_dict(
                torch.load(checkpoint_path, map_location=self._device))
        else:
            self._model.load_state_dict(
                torch.load(
                    checkpoint_path,
                    map_location=self._device)['model_state_dict'])

        self._model.train()

        optimizer = opt.Adam(
            self._model.parameters(),
            lr=self._learning_rate, weight_decay=1E-5,
            amsgrad=True)
        scheduler = None

        train_loaders = {"train": loaders['train'],
                         "valid": loaders['test']}
        runner = self._runner_class(
            model=self._model,
            loaders=train_loaders,
            optimizer=optimizer,
            criterion=self._criterion,
            epochs=self._epochs,
            callbacks=self._callbacks,
            scheduler=scheduler,
            logdir=self._logdir,
            device=self._device)
        runner.run()
