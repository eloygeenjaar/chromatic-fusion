from torch import nn
from typing import Union, Tuple


class BaseArchitecture(nn.Module):
    def __init__(self,
                 input_shape: Union[Tuple[int, int, int],
                                    Tuple[int, int, int, int]],
                 latent_dimension: int,
                 num_features: int,
                 *args,
                 **kwargs) -> None:
        """ This function initializes the base architecture that 
            each of the encoder and decoder architectures are based on.

        Args:
            input_shape (Union[Tuple[int, int, int], Tuple[int, int, int, int]]): The input shape of the data
                        for this particular encoder/decoder architecture
            latent_dimension (int): The size of the latent dimension
            num_features (int): The number of features used within the architecture itself
        """
        super().__init__()
        # Input shape:
        # 2D: (num_channels, height, width)
        # 3D: (num_channels, height, width, depth)
        self._input_shape = input_shape
        self._latent_dimension = latent_dimension
        self._num_features = num_features
        # Initialize the layers of the architecture as a list
        self._layers = nn.ModuleList()
        self._input_dimensionality = len(self._input_shape) - 2
        self._num_layers = self._get_num_layers()

    def _get_num_layers(self) -> Union[int, None]:
        """ This function calculates the number of layers that are necessary
            inside of the architecture. For larger inputs there are more layers
            that are needed to make sure their final feature vector is the same size.

        Raises:
            NotImplementedError: Currently this function expects the inputs to be
            the sizes we use in our paper.

        Returns:
            Union[int, None]: Returns the number of layers needed to reach roughly
            equivalent feature vector sizes at the end of the architecture
        """
        if len(self._input_shape) == 4:
            return 3
        elif len(self._input_shape) == 5 and self._input_shape[1] == 53:
            return 4
        elif len(self._input_shape) == 5 and self._input_shape[1] == 121:
            return 5
        elif len(self._input_shape) == 5 and self._input_shape[1] == 160:
            return 5
        elif len(self._input_shape) == 1:
            return None
        else:
            raise NotImplementedError(f'{len(self._input_shape)} input '
                                      f'dimensions are not supported: '
                                      f'{self._input_shape}')

    def _get_conv_type(self) -> Union[Tuple[nn.Module, nn.Module],
                                      Tuple[None, None]]:
        """ This function returns the type of convolution/transposed convolution
           for the architecture to use. Input shapes that have 3 dimensions
           (including number of channels) use 2D convolutions, input shapes
           with 4 dimensions use 3D convolutions.

        Raises:
            NotImplementedError: Currently our model only supports 2D and 3D
            convolutions.

        Returns:
            Union[Tuple[nn.Module, nn.Module], Tuple[None, None]]: The
            two modules that are appropraite for the convolutions of the given
            data.
        """
        if len(self._input_shape) == 3:
            return nn.Conv2d, nn.ConvTranspose2d
        elif len(self._input_shape) == 4:
            return nn.Conv3d, nn.ConvTranspose3d
        elif len(self._input_shape) == 1:
            return None, None
        else:
            raise NotImplementedError(f'{len(self._input_shape)} input '
                                      f'dimensions are not supported')
