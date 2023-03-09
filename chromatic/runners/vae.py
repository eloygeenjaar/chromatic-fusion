import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from torch import nn, optim
from torch.utils.data import DataLoader
from torch import distributions as D
from catalyst import dl, metrics, callbacks
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


class DMVAERunner(dl.Runner):
    def __init__(self,
                 model: nn.Module,
                 loaders: Dict[str, DataLoader],
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 scheduler,
                 callbacks: Dict[str, callbacks.Callback],
                 epochs: int,
                 logdir: Path,
                 device: torch.device):
        super().__init__()
        self.model = model
        self._loaders = loaders
        self._optimizer = optimizer
        self._criterion = criterion
        self._epochs = epochs
        self._callbacks = callbacks
        self._logdir = logdir
        self._device = device
        self._scheduler = scheduler

    @property
    def stages(self):
        return ["train"]
    
    def get_scheduler(self, stage, optimizer):
        return self._scheduler

    def get_engine(self):
        return dl.DeviceEngine(self._device)

    def get_stage_len(self, stage):
        return self._epochs

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }

    def get_loaders(self, stage):
        return self._loaders

    def get_model(self, stage: str):
        return self.model

    def get_optimizer(self, stage: str, model):
        return self._optimizer

    def get_callbacks(self, stage: str):
        return self._callbacks

    def on_loader_start(self, runner):
        """Event handler."""
        assert self.loader is not None
        self.is_train_loader: bool = self.loader_key.startswith("train")
        self.is_valid_loader: bool = self.loader_key.startswith("valid")
        self.is_infer_loader: bool = self.loader_key.startswith("infer")
        assert self.is_train_loader or self.is_valid_loader or self.is_infer_loader
        self.loader_batch_size: int = self.loader.batch_size
        self.loader_batch_len: int = len(self.loader)
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0
        self.loader_metrics: Dict = defaultdict(None)

        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in self._criterion.keys
        }

    def handle_batch(self, batch):
        m1 = batch[0]['m1'].float()
        m2 = batch[0]['m2'].float()
        y = batch[0]['targets']

        model_results = self.model(m1, m2)

        model_results['targets'] = y.long()
        loss, losses = self._criterion(m1, m2, model_results)

        self.batch_metrics.update(losses)
        for key in self._criterion.keys:
            self.meters[key].update(
                self.batch_metrics[key].detach().cpu(), m1.size(0))

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def on_loader_end(self, runner):
        for key in self._criterion.keys:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        #if self.is_train_loader:
        #    self.optimizer.param_groups[0]['lr'] *= 0.98
        #    print(f'New learning rate: {self.optimizer.param_groups[0]["lr"]}')
        super().on_loader_end(runner)

    def predict_batch(self, batch):
        m1 = batch[0]['m1'].float()
        m2 = batch[0]['m2'].float()
        y = batch[0]['targets']

        model_results = self.model(m1, m2)
        model_results['targets'] = y.long()
        return model_results

    def evaluate_loader(self, loader: DataLoader):
        priv_m1 = torch.zeros(
            (len(loader.dataset), 2, self.model.private_size))
        priv_m2 = torch.zeros(
            (len(loader.dataset), 2, self.model.private_size))
        shared = torch.zeros(
            (len(loader.dataset), 2, self.model.shared_size))
        shared_m1 = torch.zeros(
            (len(loader.dataset), 2, self.model.shared_size))
        shared_m2 = torch.zeros(
            (len(loader.dataset), 2, self.model.shared_size))
        y_true = torch.zeros((len(loader.dataset),))
        ix_start = 0
        for (i, batch) in enumerate(loader):
            with torch.no_grad():
                prediction = self.predict_batch(batch)
            ix_end = ix_start + prediction['targets'].size(0)

            shared[ix_start:ix_end, 0] = prediction['dists'][0].mean.detach()
            shared[ix_start:ix_end, 1] = prediction['dists'][0].stddev.detach()

            shared_m1[ix_start:ix_end, 0] = prediction['dists'][1].mean.detach()
            shared_m1[ix_start:ix_end, 1] = prediction['dists'][1].stddev.detach()

            shared_m2[ix_start:ix_end, 0] = prediction['dists'][2].mean.detach()
            shared_m2[ix_start:ix_end, 1] = prediction['dists'][2].stddev.detach()

            priv_m1[ix_start:ix_end, 0] = prediction['dists'][3].mean.detach()
            priv_m1[ix_start:ix_end, 1] = prediction['dists'][3].stddev.detach()

            priv_m2[ix_start:ix_end, 0] = prediction['dists'][4].mean.detach()
            priv_m2[ix_start:ix_end, 1] = prediction['dists'][4].stddev.detach()

            y_true[ix_start:ix_end] = prediction['targets'].detach()
            ix_start = ix_end

        return (priv_m1[:, 0], priv_m2[:, 0], shared[:, 0], shared_m1[:, 0], shared_m2[:, 0]), \
            y_true, (priv_m1, priv_m2, shared)

    def on_batch_end(self, runner: "IRunner"):
        self.log_metrics(metrics=self.batch_metrics, scope="batch")
