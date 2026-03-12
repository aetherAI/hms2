import os
import typing as t

import numpy as np
import numpy.typing as npt
import pytest
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from hms2.core.loader_modules import (
    BaseAugmentorModule, EmbeddingLoaderModule,
    GPUAugmentationLoaderModule, NoLoaderModule, PlainLoaderModule,
)


@pytest.fixture(autouse=True, scope='session')
def set_up() -> None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True


@pytest.fixture(
    scope='session',
    params=[
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='No CUDA is available.'),
        ),
    ],
)
def use_cuda(request) -> bool:
    return request.param


@pytest.fixture(scope='session')
def image() -> npt.NDArray[np.uint8]:
    image = Image.open('misc/lena_color.gif').convert('RGB')

    width = 2000
    height = 3000
    image = image.resize((width, height))

    image = np.array(image)
    return image


@pytest.fixture(scope='session')
def image_batch(image: npt.NDArray[np.uint8]) -> torch.Tensor:
    image_batch = torch.tensor(image, dtype=torch.uint8)
    image_batch = image_batch[np.newaxis, :, :, :]

    return image_batch


@pytest.fixture(scope='session')
def embedding() -> npt.NDArray[np.float32]:
    embedding = np.random.rand(2048, 128, 128).astype(np.float32)
    return embedding


@pytest.fixture(scope='session')
def embedding_batch(embedding: npt.NDArray[np.float32]) -> torch.Tensor:
    embedding_torch = torch.tensor(embedding, dtype=torch.float32)
    embedding_torch = embedding_torch[np.newaxis, :, :, :]
    return embedding_torch


@pytest.fixture(scope='session')
def imagenet_normalization() -> t.Tuple[t.Sequence[float], t.Sequence[float]]:
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]
    return {'mean': normalization_mean, 'std': normalization_std}


@pytest.fixture(scope='session')
def same_normalization() -> t.Tuple[t.Sequence[float], t.Sequence[float]]:
    normalization_mean = [0.5, 0.5, 0.5]
    normalization_std = [0.5, 0.5, 0.5]
    return {'mean': normalization_mean, 'std': normalization_std}


@pytest.mark.parametrize('do_hint', [True, False])
@pytest.mark.parametrize('normalization_str', ['imagenet_normalization', 'same_normalization'])
@pytest.mark.parametrize('loader_module_use', ['plain', 'gpu_aug_disable_aug', 'no'])
def test_loader_module_forward_with_no_aug(
    image: npt.NDArray[np.uint8],
    image_batch: torch.Tensor,
    normalization_str: str,
    use_cuda: bool,
    do_hint: bool,
    loader_module_use: str,
    request: pytest.FixtureRequest,
) -> None:
    normalization = request.getfixturevalue(normalization_str)
    if loader_module_use == 'plain':
        loader_module = PlainLoaderModule(normalization['mean'], normalization['std'])
    elif loader_module_use == 'gpu_aug_disable_aug':
        loader_module = GPUAugmentationLoaderModule(
            normalization_mean=normalization['mean'],
            normalization_std=normalization['std'],
            random_flip=False,
            random_rotation=False,
            random_translation=None,
        )
    elif loader_module_use == 'no':
        loader_module = NoLoaderModule(
            normalization_mean=normalization['mean'],
            normalization_std=normalization['std'],
        )
    else:
        raise ValueError

    if use_cuda:
        loader_module = loader_module.cuda()

    coord = (0, 1000)
    size = (1000, 2000)
    if loader_module_use in ['plain', 'gpu_aug_disable_aug']:
        if do_hint:
            loader_module.hint_future_accesses(image_batch, [coord], [size])
        output = loader_module(image_batch, coord, size)
        assert isinstance(output, torch.Tensor)
        if use_cuda:
            assert output.is_cuda
    elif loader_module_use == 'no':
        partial_image_batch = image_batch[
            :,
            coord[1]: coord[1] + size[1],
            coord[0]: coord[0] + size[0],
            :,
        ]
        output = loader_module(partial_image_batch)
        assert isinstance(output, torch.Tensor)
    else:
        raise ValueError

    output = output.cpu().numpy()
    output = output[0, ...]
    output = np.transpose(output, [1, 2, 0])
    output *= np.array(normalization['std'], dtype=np.float32)
    output += np.array(normalization['mean'], dtype=np.float32)
    output = np.minimum(np.maximum(output * 255.0, 0.0), 255.0).astype(np.uint8)

    ground_truth = image[
        coord[1]: coord[1] + size[1],
        coord[0]: coord[0] + size[0],
        :,
    ]
    ssim = structural_similarity(output, ground_truth, channel_axis=-1)
    assert ssim > 0.99


def test_gpu_augmentation_loader_module_forward_with_aug(
    image: npt.NDArray[np.uint8],
    image_batch: torch.Tensor,
    use_cuda: bool,
) -> None:
    # Augmentation arguments
    rotation_angle = 8.7
    translation_pixels = [9, -8]
    do_flip = True

    class AddBias(BaseAugmentorModule):
        def __init__(self, bias):
            super().__init__()
            self.bias = bias

        def randomize(self):
            pass

        def forward(self, inputs, is_background_tile=False):
            return inputs + self.bias

    other_augmentations = [AddBias(0.1)]

    # Get the loader module
    loader_module = GPUAugmentationLoaderModule(other_augmentations=other_augmentations)
    if use_cuda:
        loader_module = loader_module.cuda()
    loader_module.do_flip = do_flip
    loader_module.rotation_angle = rotation_angle
    loader_module.translation_pixels = translation_pixels
    loader_module.affine_matrix = loader_module._calculate_affine_matrix()

    # Do forward
    coord = (0, 1000)
    size = (1000, 2000)
    output = loader_module(image_batch, coord, size)
    assert isinstance(output, torch.Tensor)
    if use_cuda:
        assert output.is_cuda

    output = output.cpu().numpy()
    output = output[0, ...]
    output = np.transpose(output, [1, 2, 0])
    output *= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    output += np.array([0.485, 0.456, 0.406], dtype=np.float32)
    output -= 0.1  # Inverse of AddBias(0.1)
    output = np.minimum(np.maximum(output * 255.0, 0.0), 255.0).astype(np.uint8)

    # Get ground truth
    img_aug = Image.fromarray(image)
    img_aug = img_aug.rotate(
        angle=rotation_angle,
        resample=Image.Resampling.BILINEAR,
        translate=translation_pixels,
        fillcolor=(255, 255, 255),
    )
    if do_flip:
        img_aug = img_aug.transpose(method=Image.FLIP_LEFT_RIGHT)
    ground_truth = np.array(img_aug)[
        coord[1]: coord[1] + size[1],
        coord[0]: coord[0] + size[0],
        :,
    ]
    ssim = structural_similarity(output, ground_truth, channel_axis=-1)
    assert np.min(ground_truth) < 128  # The selected tile should be meaningful
    assert ssim > 0.99


def test_gpu_augmentation_loader_module_forward_with_randomness(image_batch: torch.Tensor, use_cuda: bool) -> None:
    # Augmentation arguments
    class AddBias(BaseAugmentorModule):
        def __init__(self, bias):
            super().__init__()
            self.bias = bias

        def randomize(self):
            pass

        def forward(self, inputs, is_background_tile=False):
            return inputs + self.bias

    other_augmentations = [AddBias(0.1)]

    # Get the loader module
    loader_module = GPUAugmentationLoaderModule(other_augmentations=other_augmentations)
    if use_cuda:
        loader_module = loader_module.cuda()

    # Do two forward operations in training model
    coord = (0, 1000)
    size = (1000, 2000)

    loader_module.train()
    loader_module.randomize()
    output_0 = loader_module(image_batch, coord, size)
    loader_module.randomize()
    output_1 = loader_module(image_batch, coord, size)
    assert torch.any(output_0 != output_1).item()

    # Do two forward operations in evaluation model
    loader_module.eval()
    loader_module.randomize()
    output_0 = loader_module(image_batch, coord, size)
    loader_module.randomize()
    output_1 = loader_module(image_batch, coord, size)
    assert torch.all(output_0 == output_1).item()


@pytest.mark.parametrize('do_hint', [True, False])
def test_embedding_loader_module_forward(
    embedding: npt.NDArray[np.uint8],
    embedding_batch: torch.Tensor,
    use_cuda: bool,
    do_hint: bool,
) -> None:
    loader_module = EmbeddingLoaderModule()

    if use_cuda:
        loader_module = loader_module.cuda()

    coord = (0, 10)
    size = (64, 64)

    data = {'embed': embedding_batch}
    if do_hint:
        loader_module.hint_future_accesses(data, [coord], [size])
    output = loader_module(data, coord, size)
    assert isinstance(output, torch.Tensor)
    if use_cuda:
        assert output.is_cuda

    output = output.cpu().numpy()
    output = output[0, ...]

    ground_truth = embedding[
        :,
        coord[1]: coord[1] + size[1],
        coord[0]: coord[0] + size[0],
    ]
    ssim = structural_similarity(output, ground_truth, channel_axis=-1)
    assert ssim > 0.99
