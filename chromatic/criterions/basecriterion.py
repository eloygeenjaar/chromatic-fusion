import torch
from torch import nn
from typing import Dict, List


class BaseCriterion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self._keys = []

    def forward(self, 
                model_dict: Dict[str, torch.tensor]):
        raise NotImplementedError

    @property
    def keys(self) -> List[str]:
        return self._keys
