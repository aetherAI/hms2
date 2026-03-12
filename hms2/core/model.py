from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .loader_modules import BaseLoaderModule, EmbeddingLoaderModule


class Hms2Model(nn.Module):
    """
    A torch module implementing HMS2.

    Args:
        loader_module (BaseLoaderModule):
            A module handling region reading, augmentation, and CPU-GPU transfer. Please refer to the description
            in BaseLoaderModule.
        conv_module (torch.nn.Module):
            A module implements the convolutional part of the model. It should be on CUDA.
        pre_pooling_module (torch.nn.Module):
            A module implements the pre-pooling part of the model. It should be on
            CUDA.
        local_pooling_module (torch.nn.Module):
            A module applies a pooling operation right after each tile of the embedding
            feature map is produced. The default value None disables local pooling. It
            should be on CUDA.
        dense_module (torch.nn.Module):
            A module implements the dense part of the model. It should be on CUDA.
        tile_size (int):
            The tile size for HMS2. Decrease this value if GPU OOM happens.
        emb_crop_size (int):
            The cropping size for embedding. The default value 7 is for ResNet 50.
        emb_stride_size (int):
            The striding size of the receptive fields of two neighboring embedding
            vectors. The default value 32 is for ResNet 50.
        skip_no_grad (bool):
            Skip backward computations of a tile when the gradients w.r.t. the tile
            are zero. The default is True.
        cache_background_forward (bool):
            Cache forward results of background tiles to skip re-computations. The
            default is True.
        cache_background_backward (bool):
            Cache backward results of background tiles to skip re-computations.
            The default is True.
        use_cpu_for_dense (bool):
            Whether to perform dense layer computation in CPU.
    """

    def __init__(
        self,
        loader_module: nn.Module,
        conv_module: nn.Module,
        pre_pooling_module: Optional[nn.Module],
        local_pooling_module: Optional[nn.Module],
        dense_module: nn.Module,
        tile_size: int = 4096,
        emb_crop_size: int = 7,
        emb_stride_size: int = 32,
        skip_no_grad: bool = True,
        cache_background_forward: bool = True,
        cache_background_backward: bool = True,
        use_cpu_for_dense: bool = False,
    ):
        super().__init__()

        if not isinstance(loader_module, BaseLoaderModule):
            raise ValueError('loader_module should be an instance of BaseLoaderModule.')

        self.loader_module = loader_module
        self.conv_module = conv_module
        self.pre_pooling_module = pre_pooling_module
        self.local_pooling_module = local_pooling_module
        self.dense_module = dense_module
        self.tile_size = tile_size
        self.emb_crop_size = emb_crop_size
        self.emb_stride_size = emb_stride_size
        self.skip_no_grad = skip_no_grad
        self.cache_background_forward = cache_background_forward
        self.cache_background_backward = cache_background_backward
        self.use_cpu_for_dense = use_cpu_for_dense

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Implement how tensors flow in a HMS2 model.

        Args:
            image_batch (torch.Tensor): An image batch in NHWC and uint8 dtype.

        Returns:
            output (torch.Tensor): The output of dense_module.
        """
        if not isinstance(self.loader_module, BaseLoaderModule):
            raise ValueError('loader_module should be an instance of BaseLoaderModule.')

        self.loader_module.randomize()

        parameters = list(self.conv_module.parameters())
        if self.pre_pooling_module is not None:
            parameters += list(self.pre_pooling_module.parameters())

        conv_output = _Hms2Convolutional.apply(
            image_batch,
            _Hms2ConvolutionalArguments(
                loader_module=self.loader_module,
                conv_module=self.conv_module,
                pre_pooling_module=self.pre_pooling_module,
                local_pooling_module=self.local_pooling_module,
                tile_size=self.tile_size,
                emb_crop_size=self.emb_crop_size,
                emb_stride_size=self.emb_stride_size,
                skip_no_grad=self.skip_no_grad,
                cache_background_forward=self.cache_background_forward,
                cache_background_backward=self.cache_background_backward,
                use_cpu_for_dense=self.use_cpu_for_dense,
            ),
            *parameters,
        )
        output = self.dense_module(conv_output)

        return output

    def forward_embedding(
        self,
        image_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Implement how tensors flow in a HMS2 model without the backbone.

        Args:
            image_batch (Dict[str, torch.Tensor]):
                The dictionary should contain the key 'embed', which is the major data array with shape. Currently,
                the shape of the 'embed' should be in NCHW format due to the limitation of EmbeddingLoaderModule.

        Returns:
            output (torch.Tensor): The output of dense_module.
        """
        if not isinstance(self.loader_module, EmbeddingLoaderModule):
            raise ValueError('loader_module should be an instance of EmbeddingLoaderModule.')

        if self.pre_pooling_module is None:
            raise ValueError('pre_pooling_module should be provided.')

        if not isinstance(image_batch, dict) or 'embed' not in image_batch:
            raise ValueError("image_batch should be a dictionary containing key 'embed' in embedding mode.")

        conv_output = _Hms2Convolutional.apply(
            image_batch,
            _Hms2ConvolutionalArguments(
                loader_module=self.loader_module,
                conv_module=nn.Identity(),
                pre_pooling_module=self.pre_pooling_module,
                local_pooling_module=self.local_pooling_module,
                tile_size=self.tile_size,
                emb_crop_size=self.emb_crop_size,
                emb_stride_size=self.emb_stride_size,
                skip_no_grad=self.skip_no_grad,
                cache_background_forward=self.cache_background_forward,
                cache_background_backward=self.cache_background_backward,
                use_cpu_for_dense=self.use_cpu_for_dense,
            ),
            *self.pre_pooling_module.parameters(),
        )
        output = self.dense_module(conv_output)

        return output


@dataclass
class _Hms2ConvolutionalArguments:
    """
    Arguments for `_Hms2Convolutional`.

    Args:
        See descriptions in `Hms2Model`.
    """

    loader_module: BaseLoaderModule
    conv_module: nn.Module
    pre_pooling_module: Optional[nn.Module]
    local_pooling_module: Optional[nn.Module]
    tile_size: int
    emb_crop_size: int
    emb_stride_size: int
    skip_no_grad: bool
    cache_background_forward: bool
    cache_background_backward: bool
    use_cpu_for_dense: bool


class _Hms2Convolutional(torch.autograd.Function):
    """
    The core part of HMS2 that implements tiling in the convolutional part and backward
    re-computations. Using torch.autograd.Function instead of torch.nn.Module is
    because only torch.autograd.Function can rewrite custom backward operations.
    """

    @staticmethod
    def forward(
        ctx: Any,
        image_batch: Union[Dict[str, torch.Tensor], torch.Tensor],
        arguments: _Hms2ConvolutionalArguments,
        *conv_parameters: torch.Tensor,
    ) -> torch.Tensor:
        """
        HMS2 forward-convolutional.

        Args:
            ctx (Any):
                See PyTorch documentations.
            image_batch (torch.Tensor or Dict of torch.Tensor):
                if `image_batch` is a dictionary, it should contain the keys 'embed', which is the major data
                array. Currently, the shape of the 'embed' shoule be in NCHW format due to the limitation of
                EmbeddingLoaderModule.

                if `image_batch` is a tensor, it should be in NHWC format and uint8.
            arguments (_Hms2ConvolutionalArguments):
                See descriptions in `_Hms2ConvolutionalArguments`.
            conv_parameters (list of torch.Tensor):
                A list retrieved by calling `conv_module.parameters()`.

        Returns:
            emb (torch.Tensor): The resulting embedding feature map.
        """
        # Save parameters
        ctx.image_batch = image_batch
        ctx.arguments = arguments
        ctx.conv_parameters = conv_parameters

        # Load arguments
        loader_module = arguments.loader_module

        # Create a background tile cache if required
        if arguments.cache_background_forward or arguments.cache_background_backward:
            ctx.tile_signature_factory = _TileSignatureFactory()

        if arguments.cache_background_forward:
            tile_signature_factory = ctx.tile_signature_factory
            background_tile_cache_forward = _BackgroundTileCache()
        else:
            tile_signature_factory = None
            background_tile_cache_forward = None

        # Calculate the tile number
        tile_dimensions = _Hms2Convolutional._compute_tile_dimensions(
            image_batch,
            arguments,
        )

        # Hint loader module the future accesses.
        _Hms2Convolutional._hint_loader_module(
            image_batch,
            tile_dimensions,
            arguments,
        )

        # Forward convolutional
        with torch.no_grad():  # Do no store any feature maps
            # Iterate tiles
            emb_tiles = []
            for tile_y in range(tile_dimensions[1]):
                emb_tiles_row = []
                for tile_x in range(tile_dimensions[0]):
                    # Load image tile
                    tile_coord, tile_size = _Hms2Convolutional._compute_image_tile_coord(
                        image_batch,
                        arguments,
                        (tile_x, tile_y),
                    )
                    image_tile_batch = loader_module(
                        image_batch,
                        tile_coord,
                        tile_size,
                    )

                    # Do forward
                    if arguments.cache_background_forward:
                        emb_tile = _Hms2Convolutional._forward_tile(
                            image_tile_batch,
                            arguments,
                            (tile_x, tile_y),
                            tile_dimensions,
                            tile_signature_factory,
                            background_tile_cache_forward,
                        )
                    else:
                        emb_tile = _Hms2Convolutional._forward_tile(
                            image_tile_batch,
                            arguments,
                            (tile_x, tile_y),
                            tile_dimensions,
                        )

                    emb_tiles_row.append(emb_tile)
                emb_tiles.append(emb_tiles_row)

            # Compute the look-up table for the coordinates of embedding tiles
            emb_tile_coord_lut = _Hms2Convolutional._compute_emb_tile_coord_lut(emb_tiles)

            # Concatenate tiles to get the embedding feature map
            emb_rows = [torch.cat(emb_tiles_row, dim=3) for emb_tiles_row in emb_tiles]
            emb = torch.cat(emb_rows, dim=2)

        # Save the look-up table
        ctx.emb_tile_coord_lut = emb_tile_coord_lut

        return emb

    @staticmethod
    def backward(
        ctx: Any,
        grad_emb: torch.Tensor,
    ) -> Sequence[Optional[torch.Tensor]]:
        """
        HMS2 backward-convolutional.

        Args:
            ctx (Any):
                See PyTorch documentations.
            grad_emb (torch.Tensor): The gradients w.r.t. the embedding feature map.

        Returns:
            grad_image_batch (NoneType): Remain None.
            grad_arguments (NoneType): Remain None.
            grad_conv_parameters (tuple):
                A tuple of the gradients w.r.t. parameters in the convolutional module.
        """
        # Load saved parameters
        image_batch = ctx.image_batch
        arguments = ctx.arguments
        conv_parameters = ctx.conv_parameters
        emb_tile_coord_lut = ctx.emb_tile_coord_lut

        # Load arguments
        loader_module = arguments.loader_module
        cache_background_backward = arguments.cache_background_backward

        # Create a background tile cache if required
        if arguments.cache_background_backward:
            tile_signature_factory = ctx.tile_signature_factory
            background_tile_cache_backward = _BackgroundTileCache()
        else:
            tile_signature_factory = None
            background_tile_cache_backward = None

        # Calculate the tile number
        tile_dimensions = _Hms2Convolutional._compute_tile_dimensions(
            image_batch,
            arguments,
        )

        # Hint loader module the future accesses.
        _Hms2Convolutional._hint_loader_module(
            image_batch,
            tile_dimensions,
            arguments,
        )

        # Iterate tiles
        trainable_conv_parameters = [parameter for parameter in conv_parameters if parameter.requires_grad]
        indices_trainable_conv_parameters = [
            index for index, parameter in enumerate(conv_parameters) if parameter.requires_grad
        ]
        grad_trainable_conv_parameters = [
            torch.zeros_like(parameter, device=parameter.device) for parameter in trainable_conv_parameters
        ]
        for tile_y in range(tile_dimensions[1]):
            for tile_x in range(tile_dimensions[0]):
                with torch.enable_grad():
                    # Get the gradients w.r.t. the embedding tile
                    grad_emb_tile_coord, grad_emb_tile_size = _Hms2Convolutional._use_emb_tile_coord_lut(
                        emb_tile_coord_lut,
                        (tile_x, tile_y),
                    )
                    grad_emb_tile = grad_emb[
                        :,
                        :,
                        grad_emb_tile_coord[1]: grad_emb_tile_coord[1] + grad_emb_tile_size[1],
                        grad_emb_tile_coord[0]: grad_emb_tile_coord[0] + grad_emb_tile_size[0],
                    ]
                    grad_emb_tile = grad_emb_tile.contiguous()

                    # Skip this tile if all the gradients are 0
                    if arguments.skip_no_grad and not grad_emb_tile.bool().any():
                        _Hms2Convolutional._prefetch_next(arguments)
                        continue

                    # Load image tile
                    tile_coord, tile_size = _Hms2Convolutional._compute_image_tile_coord(
                        image_batch,
                        arguments,
                        (tile_x, tile_y),
                    )
                    image_tile_batch = loader_module(
                        image_batch,
                        tile_coord,
                        tile_size,
                    )

                    # If caching is enabled and the tile is not on the
                    # edge, look up background_tile_cache_backward to get gradients.
                    # If not found, return None.
                    partial_grad_trainable_conv_parameters = None
                    tile_signature = None
                    if (
                        cache_background_backward
                        and tile_y not in [0, tile_dimensions[1] - 1]
                        and tile_x not in [0, tile_dimensions[0] - 1]
                    ):
                        if tile_signature_factory is None:
                            raise RuntimeError
                        if background_tile_cache_backward is None:
                            raise RuntimeError
                        tile_signature = tile_signature_factory(
                            (tile_x, tile_y),
                            image_tile_batch,
                        )
                        partial_grad_trainable_conv_parameters = background_tile_cache_backward[tile_signature]

                    if partial_grad_trainable_conv_parameters is None:
                        # Re-compute forward convolutional. Background tile cache
                        # should be always disabled because we need gradients.
                        emb_tile = _Hms2Convolutional._forward_tile(
                            image_tile_batch,
                            arguments,
                            (tile_x, tile_y),
                            tile_dimensions,
                            tile_signature_factory=None,
                            background_tile_cache_forward=None,
                        )

                        # Compute the partial gradients w.r.t. the parameters in the
                        # convolutional module
                        partial_grad_trainable_conv_parameters = torch.autograd.grad(
                            [emb_tile],
                            trainable_conv_parameters,
                            [grad_emb_tile],
                            allow_unused=True,
                        )

                        # Update the cache
                        if (
                            cache_background_backward
                            and tile_y not in [0, tile_dimensions[1] - 1]
                            and tile_x not in [0, tile_dimensions[0] - 1]
                        ):
                            if background_tile_cache_backward is None:
                                raise RuntimeError
                            if tile_signature is None:
                                raise RuntimeError
                            background_tile_cache_backward[tile_signature] = partial_grad_trainable_conv_parameters

                with torch.no_grad():
                    # Accumulate partial gradients
                    for idx, partial_grad in enumerate(partial_grad_trainable_conv_parameters):
                        if partial_grad is None:
                            continue
                        grad_trainable_conv_parameters[idx] += partial_grad

        grad_conv_parameters: Optional[torch.Tensor] = [None for _ in conv_parameters]
        for index, grad in zip(indices_trainable_conv_parameters, grad_trainable_conv_parameters):
            grad_conv_parameters[index] = grad

        return (None, None) + tuple(grad_conv_parameters)

    @staticmethod
    def _forward_tile(
        image_tile_batch: torch.Tensor,
        arguments: _Hms2ConvolutionalArguments,
        tile_indices: Tuple[int, int],
        tile_dimensions: Tuple[int, int],
        tile_signature_factory: Optional['_TileSignatureFactory'] = None,
        background_tile_cache_forward: Optional['_BackgroundTileCache'] = None,
    ) -> torch.Tensor:
        # Get arguments
        conv_module = arguments.conv_module
        pre_pooling_module = arguments.pre_pooling_module
        local_pooling_module = arguments.local_pooling_module
        emb_crop_size = arguments.emb_crop_size

        # Look up background_tile_cache_forward to get emb_tile. If not found, return
        # None
        emb_tile = None
        tile_signature = None
        if background_tile_cache_forward is not None:
            if tile_signature_factory is None:
                raise RuntimeError
            tile_signature = tile_signature_factory(tile_indices, image_tile_batch)
            emb_tile = background_tile_cache_forward[tile_signature]

        # Do convolutions when cache miss
        if emb_tile is None:
            emb_tile = conv_module(image_tile_batch)

            if background_tile_cache_forward is not None:
                if tile_signature is None:
                    raise RuntimeError
                background_tile_cache_forward[tile_signature] = emb_tile

        # Crop invalid borders
        tile_x, tile_y = tile_indices
        _, _, emb_tile_height, emb_tile_width = emb_tile.shape
        left, right, top, bottom = 0, emb_tile_width, 0, emb_tile_height

        if emb_crop_size > 0:
            left = emb_crop_size if tile_x != 0 else 0
            right = -emb_crop_size if tile_x != tile_dimensions[0] - 1 else emb_tile_width
            top = emb_crop_size if tile_y != 0 else 0
            bottom = -emb_crop_size if tile_y != tile_dimensions[1] - 1 else emb_tile_height

        emb_tile = emb_tile[
            :,
            :,
            top:bottom,
            left:right,
        ]

        if pre_pooling_module is not None:
            emb_tile = pre_pooling_module(emb_tile)

        # Local pooling
        if local_pooling_module is not None:
            emb_tile = local_pooling_module(emb_tile)

        # To CPU if required
        if arguments.use_cpu_for_dense:
            emb_tile = emb_tile.to('cpu')

        return emb_tile

    @staticmethod
    def _compute_tile_dimensions(
        image_batch: Union[Dict[str, torch.Tensor], torch.Tensor],
        arguments: _Hms2ConvolutionalArguments,
    ) -> Tuple[int, int]:
        # Get arguments
        tile_size = arguments.tile_size
        emb_crop_size = arguments.emb_crop_size
        emb_stride_size = arguments.emb_stride_size

        # Compute tile dimensions
        if isinstance(image_batch, dict):
            if 'embed' in image_batch:
                _, _, height, width = image_batch['embed'].shape
            else:
                raise ValueError("image_batch should contain key 'embed' in embedding mode.")
        else:
            _, height, width, _ = image_batch.shape

        overlapping_size = emb_crop_size * emb_stride_size * 2
        tile_width = max(0, int(np.ceil((width - tile_size) / (tile_size - overlapping_size)))) + 1
        tile_height = max(0, int(np.ceil((height - tile_size) / (tile_size - overlapping_size)))) + 1

        return (tile_width, tile_height)

    @staticmethod
    def _hint_loader_module(
        image_batch: Union[Dict[str, torch.Tensor], torch.Tensor],
        tile_dimensions: Tuple[int, int],
        arguments: _Hms2ConvolutionalArguments,
    ) -> None:
        # Get arguments
        loader_module = arguments.loader_module

        # Calculate tile coordinates and sizes that will be accessed, and hint the
        # loader.
        tile_coords = []
        tile_sizes = []
        for tile_y in range(tile_dimensions[1]):
            for tile_x in range(tile_dimensions[0]):
                tile_coord, tile_size = _Hms2Convolutional._compute_image_tile_coord(
                    image_batch,
                    arguments,
                    (tile_x, tile_y),
                )
                tile_coords.append(tile_coord)
                tile_sizes.append(tile_size)

        loader_module.hint_future_accesses(image_batch, tile_coords, tile_sizes)

    @staticmethod
    def _prefetch_next(arguments: _Hms2ConvolutionalArguments) -> None:
        loader_module = arguments.loader_module
        loader_module.prefetch_next()

    @staticmethod
    def _compute_image_tile_coord(
        image_batch: Union[Dict[str, torch.Tensor], torch.Tensor],
        arguments: _Hms2ConvolutionalArguments,
        tile_indices: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        # Get arguments
        tile_size = arguments.tile_size
        emb_crop_size = arguments.emb_crop_size
        emb_stride_size = arguments.emb_stride_size

        # Compute coord and size
        if isinstance(image_batch, dict):
            if 'embed' in image_batch:
                _, _, height, width = image_batch['embed'].shape
            else:
                raise ValueError("image_batch should contain key 'embed' in embedding mode.")
        else:
            _, height, width, _ = image_batch.shape
        overlapping_size = emb_crop_size * emb_stride_size * 2
        tile_x, tile_y = tile_indices
        coord_x = tile_x * (tile_size - overlapping_size)
        coord_y = tile_y * (tile_size - overlapping_size)
        size_x = min(tile_size, width - coord_x)
        size_y = min(tile_size, height - coord_y)

        return (coord_x, coord_y), (size_x, size_y)

    @staticmethod
    def _compute_emb_tile_coord_lut(
        emb_tiles: Sequence[Sequence[torch.Tensor]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        widths = np.array([emb_tile.shape[3] for emb_tile in emb_tiles[0]])
        cum_widths = np.cumsum(widths)

        heights = np.array([row_emb_tiles[0].shape[2] for row_emb_tiles in emb_tiles])
        cum_heights = np.cumsum(heights)

        return widths, cum_widths, heights, cum_heights

    @staticmethod
    def _use_emb_tile_coord_lut(
        lut: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tile_indices: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        widths, cum_widths, heights, cum_heights = lut
        tile_x, tile_y = tile_indices

        coord_x = cum_widths[tile_x] - widths[tile_x]
        coord_y = cum_heights[tile_y] - heights[tile_y]
        size_x = widths[tile_x]
        size_y = heights[tile_y]

        return (coord_x, coord_y), (size_x, size_y)


@dataclass
class _TileSignature:
    background: bool
    shape: Optional[torch.Size] = None
    pixel_values: Optional[NDArray[np.float_]] = None

    @torch.no_grad()
    def __eq__(self, other: '_TileSignature') -> bool:
        return (
            self.background == other.background
            and self.shape == other.shape
            and bool(np.all(self.pixel_values == other.pixel_values))
        )


class _TileSignatureFactory:
    def __init__(self):
        self.cache = {}

    @torch.no_grad()
    def __call__(
        self,
        tile_indices: Tuple[int, int],
        tile: torch.Tensor,
    ) -> '_TileSignature':
        # Cache hit.
        if tile_indices in self.cache:
            return self.cache[tile_indices]

        # Get basic info of the tile.
        shape = tile.shape
        pixel_values = tile[0, :, 0, 0].detach().contiguous()

        # Check if the tile is background tile.
        if torch.all(tile == pixel_values[np.newaxis, :, np.newaxis, np.newaxis]):
            pixel_values_np = pixel_values.cpu().numpy()
            tile_signature = _TileSignature(
                background=True,
                shape=shape,
                pixel_values=pixel_values_np,
            )
        else:
            tile_signature = _TileSignature(background=False)

        # Update the cache and return
        self.cache[tile_indices] = tile_signature
        return tile_signature


class _BackgroundTileCache:
    def __init__(self):
        self.cache = []

    @torch.no_grad()
    def __getitem__(
        self,
        tile_signature: _TileSignature,
    ) -> Any:
        # If not background, return nothing.
        if not tile_signature.background:
            return None

        # Cache look-up.
        for entry in self.cache:
            if tile_signature == entry['tile_signature']:
                return entry['result']

        # Cache miss
        return None

    @torch.no_grad()
    def __setitem__(
        self,
        tile_signature: _TileSignature,
        result: Any,
    ) -> None:
        # If not background, return.
        if not tile_signature.background:
            return

        # The tile should not be already in the cache.
        for entry in self.cache:
            if tile_signature == entry['tile_signature']:
                raise ValueError('The _BackgroundTileCache already stores the same entry.')

        # Update the cache
        self.cache.append(
            {
                'tile_signature': tile_signature,
                'result': result,
            },
        )
