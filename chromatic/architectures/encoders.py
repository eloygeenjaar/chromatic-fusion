import torch

from .basearchitecture import BaseArchitecture
from typing import Union, Tuple
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm  # noqa


class MLP2DEncoder(BaseArchitecture):
    def __init__(self,
                 input_shape: Union[Tuple[int, int, int],
                                    Tuple[int, int, int, int]],
                 priv_size: int,
                 shared_size: int,
                 num_features: int):
        super().__init__(input_shape=input_shape,
                         latent_dimension=priv_size + shared_size,
                         num_features=num_features)
        self._priv_size = priv_size
        self._shared_size = shared_size
        self._lin1 = nn.Sequential(nn.Linear(1378, 512, bias=False), nn.GroupNorm(8, 512), nn.ELU(), nn.Dropout(0.05))
        self._lin2 = nn.Sequential(nn.Linear(512, 256, bias=False), nn.GroupNorm(8, 256), nn.ELU(), nn.Dropout(0.05))
        self._lin3 = nn.Sequential(nn.Linear(256, 128, bias=False), nn.GroupNorm(8, 128), nn.ELU(), nn.Dropout(0.05))
        self._lin4 = nn.Sequential(nn.Linear(128, (self._shared_size + self._priv_size), bias=False), nn.Dropout(0.05))
        self._feature_map_size = self._shared_size + self._priv_size
        self.apply(self._init_weights)

    @property
    def private_size(self):
        return self._priv_size
    
    @property
    def shared_size(self):
        return self._shared_size

    @property
    def output_size(self):
        return self._feature_map_size

    def forward(self, x):
        batch_size = x.size(0)
        x = self._lin1(x)
        x = self._lin2(x)
        x = self._lin3(x)
        x = self._lin4(x)
        return x

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('Norm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m.bias, 'data'):  
                nn.init.constant_(m.bias.data, 0)
    
    def __str__(self):
        return 'MLP2DEncoder'

class DCGANEncoder(BaseArchitecture):
    def __init__(self,
                 input_shape: Union[Tuple[int, int, int],
                                    Tuple[int, int, int, int]],
                 priv_size: int,
                 shared_size: int,
                 num_features: int):
        super().__init__(input_shape=input_shape,
                         latent_dimension=priv_size + shared_size,
                         num_features=num_features)
    
        self._priv_size = priv_size
        self._shared_size = shared_size
        self.layers = nn.Sequential(
            nn.Conv3d(1, self._num_features, 4, 2, 1, bias=False),
            nn.ELU(inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv3d(self._num_features, self._num_features * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(8, self._num_features * 2),
            nn.ELU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv3d(self._num_features * 2, self._num_features * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(8, self._num_features * 4),
            nn.ELU(inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv3d(self._num_features * 4, self._num_features * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(8, self._num_features * 8),
            nn.ELU(inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv3d(self._num_features * 8, self._num_features * 16, 4, 2, 1, bias=False),
            nn.GroupNorm(8, self._num_features * 16),
            nn.ELU(inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv3d(self._num_features * 16, priv_size + shared_size, (3, 4, 3), 1, 0, bias=False)
        )
        self.apply(self._init_weights)
        
    @property
    def private_size(self):
        return self._priv_size
    
    @property
    def shared_size(self):
        return self._shared_size

    @property
    def output_size(self):
        return self._feature_map_size

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), x.size(1))
        return x

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('Norm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m.bias, 'data'):  
                nn.init.constant_(m.bias.data, 0)
            
    def __str__(self):
        return 'DCGANEncoder'


class DCGANEncoderICA(BaseArchitecture):
    def __init__(self,
                 input_shape: Union[Tuple[int, int, int],
                                    Tuple[int, int, int, int]],
                 priv_size: int,
                 shared_size: int,
                 num_features: int):
        super().__init__(input_shape=input_shape,
                         latent_dimension=priv_size + shared_size,
                         num_features=num_features)
    
        self._priv_size = priv_size
        self._shared_size = shared_size
    
        self.layers = nn.Sequential(
                nn.Conv3d(8, self._num_features * 2, 4, 2, 1, bias=False),
                nn.ELU(inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv3d(self._num_features * 2, self._num_features * 4, 4, 2, 1, bias=False),
                nn.GroupNorm(8, self._num_features * 4),
                nn.ELU(inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv3d(self._num_features * 4, self._num_features * 8, 4, 2, 1, bias=False),
                nn.GroupNorm(8, self._num_features * 8),
                nn.ELU(inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv3d(self._num_features * 8, self._num_features * 16, 4, 2, 1, bias=False),
                nn.GroupNorm(8, self._num_features * 16),
                nn.ELU(inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv3d(self._num_features * 16, priv_size + shared_size, (3, 3, 3), 1, 0, bias=False)
        ) 
        self.apply(self._init_weights)
        
    @property
    def private_size(self):
        return self._priv_size
    
    @property
    def shared_size(self):
        return self._shared_size

    @property
    def output_size(self):
        return self._feature_map_size

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), x.size(1))
        return x

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('Norm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m.bias, 'data'):  
                nn.init.constant_(m.bias.data, 0)
            
    def __str__(self):
        return 'DCGANEncoderICA'