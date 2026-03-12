import abc
import typing as t

import torch
import torch.nn.functional as F  # noqa


class BaseCompressor(abc.ABC):
    @abc.abstractproperty
    def preloadable_keys(self) -> t.Set[str]:
        """Key set that can be preloaded to GPU memory before patch decompression."""
        ...

    @abc.abstractmethod
    def compress(
        self,
        data: t.Dict[str, torch.Tensor],
    ) -> t.Dict[str, torch.Tensor]:
        ...

    @abc.abstractmethod
    def decompress(
        self,
        data: t.Dict[str, torch.Tensor],
    ) -> t.Dict[str, torch.Tensor]:
        ...


class FP16Compressor(BaseCompressor):
    @property
    def preloadable_keys(self) -> t.Set[str]:
        return set()

    def compress(
        self,
        data: t.Dict[str, torch.Tensor],
    ) -> t.Dict[str, torch.Tensor]:
        """Convert the input tensor to float16 for compression.
        Args:
            data (Dict[str, torch.Tensor]): Input dictionary containing 'embed' key. 'embed' is a tensor of shape
            (C, H, W) representing the embeddings.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the compressed 'embed' tensor.  The 'embed' tensor is
            converted to float16 for compression.
        """
        data['embed'] = data['embed'].to(torch.float16)
        return data

    def decompress(
        self,
        data: t.Dict[str, torch.Tensor],
    ) -> t.Dict[str, torch.Tensor]:
        data['embed'] = data['embed'].to(torch.float32)
        return data


class UniqueVectorCompressor(BaseCompressor):
    @property
    def preloadable_keys(self) -> t.Set[str]:
        return set()

    def compress(
        self,
        data: t.Dict[str, torch.Tensor],
    ) -> t.Dict[str, torch.Tensor]:
        """Compress the input tensor by keeping only unique vectors.

        Args:
            data (Dict[str, torch.Tensor]): Input dictionary containing 'embed' key. 'embed' is a tensor of shape
            (C, H, W) representing the embeddings.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the compressed 'embed' tensor with unique vectors.
        """
        embed = data['embed']
        c, h, w = embed.shape
        embed = embed.reshape(c, -1)  # C, H, W -> C, N
        vectors = torch.unique(embed, dim=1)
        vectors = vectors.reshape(c, 1, -1)  # C, N -> C, 1, N

        data['embed'] = vectors
        return data

    def decompress(self, data: t.Dict[str, torch.Tensor]) -> t.Dict[str, torch.Tensor]:
        return data


class Int8Compressor(BaseCompressor):
    def __init__(self, num_bits: int = 8):
        self.num_bits = num_bits
        self.qmin = 0
        self.qmax = 2**num_bits - 1

    @property
    def preloadable_keys(self) -> t.Set[str]:
        return {'scale', 'zp'}

    def _quantize_tensor(self, tensor: torch.Tensor):
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / float(self.qmax - self.qmin)
        zero_point = self.qmin - torch.round(min_val / scale)
        zero_point = zero_point.to(torch.int32)

        quantized = torch.clamp(torch.round(tensor / scale + zero_point), self.qmin, self.qmax).to(torch.uint8)
        return quantized, scale, zero_point

    def _dequantize_tensor(self, quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
        return (quantized.to(torch.float32) - zero_point) * scale

    def compress(self, data: t.Dict[str, torch.Tensor]) -> t.Dict[str, torch.Tensor]:
        """Compress the input tensor using quantization.
        Args:
            data (Dict[str, torch.Tensor]): Input dictionary containing 'embed' key. 'embed' is a tensor of shape
            (C, H, W) representing the embeddings.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the compressed 'embed' tensor, scale, and zero point.
            The 'embed' tensor is quantized to uint8 format for compression.
        """
        q, s, zp = self._quantize_tensor(data['embed'])
        data['embed'] = q
        data['scale'] = s
        data['zp'] = zp
        return data

    def decompress(self, data: t.Dict[str, torch.Tensor]) -> t.Dict[str, torch.Tensor]:
        data['embed'] = self._dequantize_tensor(data['embed'], data['scale'], data['zp'])
        return data


class AvgPoolCompressor(BaseCompressor):
    """
    Compressor using average pooling. Due to GPU memory limitations, we need to chunk the input tensor.
    Args:
        kernel_size (int): Kernel size for average pooling.
        chunk_size (int): Maximum number of elements per GPU chunk, default is 500_000_000.
    """

    def __init__(self, kernel_size: int = 3, chunk_size: int = 500_000_000) -> None:
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.chunk_size = chunk_size  # max elements per GPU chunk

    @property
    def preloadable_keys(self) -> t.Set[str]:
        return set()

    def compress(self, data: t.Dict[str, torch.Tensor]) -> t.Dict[str, torch.Tensor]:
        """
        Compress the input tensor using average pooling.
        Args:
            data (Dict[str, torch.Tensor]): Input dictionary containing 'embed' key. 'embed' is a tensor
            of shape (C, H, W) representing the embeddings.
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the compressed 'embed' tensor. The 'embed' tensor is
            compressed using average pooling.
        """
        data['embed'] = F.avg_pool2d(
            data['embed'].unsqueeze(0),
            kernel_size=self.kernel_size,
            stride=self.stride,
            ceil_mode=True,
        ).squeeze(0)
        return data

    def decompress(self, data: t.Dict[str, torch.Tensor]) -> t.Dict[str, torch.Tensor]:
        return data


def get_compressor(
    compressor_name: str,
) -> BaseCompressor:
    compressor = None
    if compressor_name == 'fp16':
        compressor = FP16Compressor()
    elif compressor_name == 'int8':
        compressor = Int8Compressor()
    elif compressor_name == 'avg_pool_3x3':
        compressor = AvgPoolCompressor(kernel_size=3)
    elif compressor_name == 'avg_pool_5x5':
        compressor = AvgPoolCompressor(kernel_size=5)
    elif compressor_name == 'avg_pool_7x7':
        compressor = AvgPoolCompressor(kernel_size=7)
    elif compressor_name == 'unique_vector':
        compressor = UniqueVectorCompressor()
    else:
        raise ValueError(f'{compressor_name} is not a supported compressor.')
    return compressor
