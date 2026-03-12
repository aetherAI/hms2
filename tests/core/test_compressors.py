import os

import pytest
import torch

from hms2.core.compressors import (
    AvgPoolCompressor, FP16Compressor, Int8Compressor, UniqueVectorCompressor,
)


@pytest.fixture(autouse=True, scope='session')
def set_up() -> None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True


@pytest.fixture(scope='session')
def feature_map() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.7, 0.8, 0.9]],
            [[1.1, 1.2, 1.3], [1.5, 1.5, 1.5], [1.7, 1.8, 1.9]],
        ],
        dtype=torch.float32,
    )


def test_fp16_compressor(
    feature_map: torch.Tensor,
):
    data = {'embed': feature_map}
    compressor = FP16Compressor()
    compressed_data = compressor.compress(data)
    assert isinstance(compressed_data['embed'], torch.Tensor)
    assert compressed_data['embed'].dtype == torch.float16

    decompressed_data = compressor.decompress(compressed_data)
    assert decompressed_data['embed'].dtype == torch.float32
    assert torch.allclose(data['embed'], decompressed_data['embed'], atol=1e-4)


def test_int8_compressor(
    feature_map: torch.Tensor,
):
    data = {'embed': feature_map}
    compressor = Int8Compressor()
    compressed_data = compressor.compress(data)

    assert isinstance(compressed_data['embed'], torch.Tensor)
    assert compressed_data['embed'].dtype == torch.uint8
    assert torch.allclose(compressed_data['scale'], torch.tensor(0.0071), atol=1e-4)
    assert compressed_data['zp'] == torch.tensor(-14, dtype=torch.int32)

    decompressed_data = compressor.decompress(compressed_data)
    assert decompressed_data['embed'].dtype == torch.float32
    assert torch.allclose(data['embed'], decompressed_data['embed'], atol=1e-2)


def test_unique_vector_compressor(
    feature_map: torch.Tensor,
):
    data = {'embed': feature_map}
    compressor = UniqueVectorCompressor()
    compressed_data = compressor.compress(data)

    embed_answer = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]],
            [[1.1, 1.2, 1.3, 1.5, 1.7, 1.8, 1.9]],
        ],
        dtype=torch.float32,
    )

    assert compressed_data['embed'].shape == (2, 1, 7)
    assert torch.allclose(compressed_data['embed'], embed_answer, atol=1e-4)


def test_avg_pool_compressor(
    feature_map: torch.Tensor,
):
    data = {'embed': feature_map}
    compressor = AvgPoolCompressor(kernel_size=3)
    compressed_data = compressor.compress(data)

    embed_answer = torch.tensor(
        [
            [[0.5]],
            [[1.5]],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(compressed_data['embed'], embed_answer, atol=1e-4)
