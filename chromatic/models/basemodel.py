import torch

from torch import nn
from typing import Dict


class BaseModel(nn.Module):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__()
        self._latent_space = None
        pass

    def forward(self, x):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_keys(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def _get_encoder(self):
        raise NotImplementedError

    def get_latent_space(self) -> Dict[str, torch.tensor]:
        raise NotImplementedError
