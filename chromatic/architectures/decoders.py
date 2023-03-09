import torch  # noqa

from .basearchitecture import BaseArchitecture
from typing import Union, Tuple
from torch import nn  # noqa
from torch.nn import functional as F  # noqa
from torch.nn.utils import weight_norm  # noqa


class MLP2DDecoder(BaseArchitecture):
    def __init__(self,
                 input_shape: Union[Tuple[int, int, int],
                                    Tuple[int, int, int, int]],
                 private_dim: int,
                 shared_dim: int,
                 num_features: int):
        super().__init__(input_shape=input_shape,
                         latent_dimension=private_dim + shared_dim,
                         num_features=num_features)
        self._input_size = private_dim + shared_dim
        self._lin1 = nn.Sequential(nn.Linear(self._input_size, 128, bias=False), nn.GroupNorm(8, 128), nn.ELU(), nn.Dropout(0.05))
        self._lin2 = nn.Sequential(nn.Linear(128, 256, bias=False), nn.GroupNorm(8, 256), nn.ELU(), nn.Dropout(0.05))
        self._lin3 = nn.Sequential(nn.Linear(256, 512, bias=False), nn.GroupNorm(8, 512), nn.ELU(), nn.Dropout(0.05))
        self._lin4 = nn.Linear(512, 1378, bias=False)
        self.apply(self._init_weights)

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
        return 'MLP2DDecoder'


class DCGANDecoder(BaseArchitecture):
    def __init__(self,
                 input_shape: Union[Tuple[int, int, int],
                                    Tuple[int, int, int, int]],
                 private_dim: int,
                 shared_dim: int,
                 num_features: int):
        super().__init__(input_shape=input_shape,
                         latent_dimension=private_dim + shared_dim,
                         num_features=num_features)
        _, self._conv_type = self._get_conv_type()
        # Feature map size for first layer:
        # spatial_size * num_dimensions * num_channels in first layer
        self._first_layer_features = 2 ** (self._num_layers - 1) \
            * self._num_features
        self._feature_map_size = (2 ** self._input_dimensionality) \
            * self._first_layer_features
        self._input_size = private_dim + shared_dim
        
        self._output_shapes = [
            (3, 4, 3),
            (7, 9, 7),
            (15, 18, 15),
            (30, 36, 30),
            (60, 72, 60),
            (121, 145, 121)]
        
        self._layers = nn.ModuleList([
            nn.ConvTranspose3d(self._input_size, self._num_features * 16, (3, 4, 3), 1, 0, bias=False),
            nn.GroupNorm(8, self._num_features * 16),
            nn.ELU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose3d(self._num_features * 16, self._num_features * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(8, self._num_features * 8),
            nn.ELU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose3d(self._num_features * 8, self._num_features * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(8, self._num_features * 4),
            nn.ELU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose3d(self._num_features * 4, self._num_features * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(8, self._num_features * 2),
            nn.ELU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose3d(self._num_features * 2, self._num_features, 4, 2, 1, bias=False),
            # state size. (nc) x 64 x 64
            nn.GroupNorm(8, self._num_features),
            nn.ELU(True),
            nn.ConvTranspose3d(self._num_features, 1, 4, 2, 1, bias=False)]
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        i = 0
        for layer in self._layers:
            if isinstance(layer, self._conv_type):
                x = layer(x, output_size=self._output_shapes[i])
                i += 1
        x = x.view(x.size(0), -1)
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

    @property
    def feature_map_shape(self):
        return self._feature_map_shape
    
    def __str__(self):
        return 'DCGANDecoder'


class DCGANDecoderICA(BaseArchitecture):
    def __init__(self,
                 input_shape: Union[Tuple[int, int, int],
                                    Tuple[int, int, int, int]],
                 private_dim: int,
                 shared_dim: int,
                 num_features: int):
        super().__init__(input_shape=input_shape,
                         latent_dimension=private_dim + shared_dim,
                         num_features=num_features)
        _, self._conv_type = self._get_conv_type()
        # Feature map size for first layer:
        # spatial_size * num_dimensions * num_channels in first layer
        self._first_layer_features = 2 ** (self._num_layers - 1) \
            * self._num_features
        self._feature_map_size = (2 ** self._input_dimensionality) \
            * self._first_layer_features
        self._priv_size = private_dim
        self._shared_size = shared_dim
        self._input_size = private_dim + shared_dim
        
        self._output_shapes = [
            (3, 3, 3),
            (6, 7, 6),
            (13, 15, 13),
            (26, 31, 26),
            (53, 63, 52)]

        self._layers = nn.ModuleList([
            nn.ConvTranspose3d(self._input_size, self._num_features * 16, (3, 3, 3), 1, 0, bias=False),
            nn.GroupNorm(8, self._num_features * 16),
            nn.ELU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose3d(self._num_features * 16, self._num_features * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(8, self._num_features * 8),
            nn.ELU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose3d(self._num_features * 8, self._num_features * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(8, self._num_features * 4),
            nn.ELU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose3d(self._num_features * 4, self._num_features * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(8, self._num_features * 2),
            nn.ELU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose3d(self._num_features * 2, 8, 4, 2, 1, bias=False)]
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        i = 0
        for layer in self._layers:
            if isinstance(layer, self._conv_type):
                x = layer(x, output_size=self._output_shapes[i])
                i += 1
        x = x.view(x.size(0), -1)
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
    
    @property
    def feature_map_shape(self):
        return self._feature_map_shape
    
    def __str__(self):
        return 'DCGANDecoderICA'
