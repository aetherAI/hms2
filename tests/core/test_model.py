import os
import typing as t
import warnings
from time import time

import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from hms2.core.loader_modules import EmbeddingLoaderModule, PlainLoaderModule
from hms2.core.model import Hms2Model


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
def conv_module(use_cuda: bool) -> nn.Module:
    resnet = torchvision.models.resnet18(pretrained=True).eval()
    conv_module = nn.Sequential(*list(resnet.children())[:-2])
    if use_cuda:
        conv_module = conv_module.cuda()
    return conv_module


@pytest.fixture(scope='session')
def pre_pooling_module(use_cuda: bool) -> nn.Module:
    resnet = torchvision.models.resnet18(pretrained=True).eval()
    channels = list(resnet.children())[-1].in_features
    pre_pooling_module = nn.Sequential(
        nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
        ),
        nn.ReLU(),
    )
    if use_cuda:
        pre_pooling_module = pre_pooling_module.cuda()
    return pre_pooling_module


@pytest.fixture(scope='session')
def dense_module(use_cuda: bool) -> nn.Module:
    resnet = torchvision.models.resnet18(pretrained=True).eval()
    dense_module = nn.Sequential(
        nn.AdaptiveMaxPool2d((1, 1)),
        nn.Flatten(),
        list(resnet.children())[-1],
    )
    if use_cuda:
        dense_module = dense_module.cuda()
    return dense_module


@pytest.fixture(scope='session')
def local_pooling_module() -> t.Optional[nn.Module]:
    local_pooling_module = nn.AdaptiveMaxPool2d((1, 1))
    return local_pooling_module


@pytest.fixture(scope='session', params=[3072, 4096])
def hms2_model(
    conv_module: nn.Module,
    pre_pooling_module: nn.Module,
    dense_module: nn.Module,
    local_pooling_module: t.Optional[nn.Module],
    use_cuda: bool,
    request: pytest.FixtureRequest,
) -> nn.Module:
    tile_size = request.param

    loader_module = PlainLoaderModule()
    if use_cuda:
        loader_module = loader_module.cuda()
    hms2_model = Hms2Model(
        loader_module=loader_module,
        conv_module=conv_module,
        pre_pooling_module=pre_pooling_module,
        dense_module=dense_module,
        local_pooling_module=local_pooling_module,
        tile_size=tile_size,
        emb_crop_size=7,
        emb_stride_size=32,
    )
    return hms2_model


@pytest.fixture(scope='session', params=[3136])
def hms2_feature_extraction_model(
    conv_module: nn.Module,
    use_cuda: bool,
    request: pytest.FixtureRequest,
):
    tile_size = request.param

    loader_module = PlainLoaderModule()
    if use_cuda:
        loader_module = loader_module.cuda()

    hms2_model = Hms2Model(
        loader_module=loader_module,
        conv_module=conv_module,
        pre_pooling_module=nn.Identity(),
        dense_module=nn.Identity(),
        local_pooling_module=None,
        skip_no_grad=False,
        cache_background_forward=False,
        cache_background_backward=False,
        tile_size=tile_size,
        emb_crop_size=0,
        emb_stride_size=1,
    )
    return hms2_model


@pytest.fixture(scope='session')
def hms2_embedding_model(
    conv_module: nn.Module,
    pre_pooling_module: nn.Module,
    local_pooling_module: t.Optional[nn.Module],
    dense_module: nn.Module,
    use_cuda: bool,
):
    loader_module = EmbeddingLoaderModule()
    if use_cuda:
        loader_module = loader_module.cuda()

    hms2_model = Hms2Model(
        loader_module=loader_module,
        conv_module=conv_module,
        pre_pooling_module=pre_pooling_module,
        dense_module=dense_module,
        local_pooling_module=local_pooling_module,
        skip_no_grad=False,
        cache_background_forward=False,
        cache_background_backward=False,
        tile_size=128 * 128,
        emb_crop_size=0,
        emb_stride_size=1,
    )
    return hms2_model


@pytest.fixture(scope='session')
def plain_model(
    conv_module: nn.Module, pre_pooling_module: nn.Module, dense_module: nn.Module, use_cuda: bool,
) -> nn.Module:
    class PlainModel(nn.Module):
        def __init__(self, conv_module, dense_module):
            super().__init__()
            self.conv_module = conv_module
            self.pre_pooling_module = pre_pooling_module
            self.dense_module = dense_module

        def forward(self, image_batch):
            if use_cuda:
                image_batch = image_batch.cuda()
            image_batch = image_batch.permute(0, 3, 1, 2).contiguous()
            image_batch = image_batch.float().div(255.0)
            image_batch = transforms.functional.normalize(
                tensor=image_batch,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            conv_output = self.conv_module(image_batch)
            conv_output = self.pre_pooling_module(conv_output)
            output = self.dense_module(conv_output)
            return output

    plain_model = PlainModel(conv_module, dense_module)
    return plain_model


def test_hms2_model_forward(hms2_model: nn.Module, plain_model: nn.Module, image_batch: torch.Tensor) -> None:
    hms2_output = hms2_model(image_batch)
    hms2_output = hms2_output.detach().cpu().numpy()

    plain_output = plain_model(image_batch)
    plain_output = plain_output.detach().cpu().numpy()

    np.testing.assert_allclose(hms2_output, plain_output)


def test_hms2_embedding_model_forward(
    hms2_feature_extraction_model: nn.Module,
    hms2_embedding_model: nn.Module,
    plain_model: nn.Module,
    image_batch: torch.Tensor,
) -> None:
    feature_map = hms2_feature_extraction_model(image_batch)
    print(feature_map.shape)

    input_dict = {'embed': feature_map}
    hms2_output = hms2_embedding_model.forward_embedding(input_dict)
    hms2_output = hms2_output.detach().cpu().numpy()

    plain_output = plain_model(image_batch)
    plain_output = plain_output.detach().cpu().numpy()

    np.testing.assert_allclose(hms2_output, plain_output)


def test_hms2_model_backward(
    hms2_model: nn.Module,
    plain_model: nn.Module,
    image_batch: torch.Tensor,
    use_cuda: bool,
) -> None:
    target_batch = torch.tensor(np.array([100]), dtype=torch.long)
    if use_cuda:
        target_batch = target_batch.cuda()

    hms2_output = hms2_model(image_batch)
    hms2_loss = nn.CrossEntropyLoss()(hms2_output, target_batch)
    hms2_model.zero_grad()
    hms2_loss.backward()
    hms2_grads = [parameter.grad.cpu().numpy() for parameter in hms2_model.parameters()]

    plain_output = plain_model(image_batch)
    plain_loss = nn.CrossEntropyLoss()(plain_output, target_batch)
    plain_model.zero_grad()
    plain_loss.backward()
    plain_grads = [parameter.grad.cpu().numpy() for parameter in plain_model.parameters()]

    assert len(hms2_grads) == len(plain_grads)
    for idx, _ in enumerate(hms2_grads):
        np.testing.assert_allclose(hms2_grads[idx], plain_grads[idx])


def test_hms2_embedding_model_backward(
    hms2_feature_extraction_model: nn.Module,
    hms2_embedding_model: nn.Module,
    image_batch: torch.Tensor,
    use_cuda: bool,
) -> None:
    target_batch = torch.tensor(np.array([100]), dtype=torch.long)
    if use_cuda:
        target_batch = target_batch.cuda()

    feature_map = hms2_feature_extraction_model(image_batch)
    input_dict = {'embed': feature_map}

    hms2_output = hms2_embedding_model.forward_embedding(input_dict)
    hms2_loss = nn.CrossEntropyLoss()(hms2_output, target_batch)
    hms2_embedding_model.zero_grad()
    hms2_loss.backward()
    for key, arr in hms2_embedding_model.named_parameters():
        if key.startswith('conv_module'):
            assert arr.grad is None
        else:
            assert arr.grad is not None


def test_hms2_model_backward_with_no_grad(hms2_model: nn.Module, image_batch: torch.Tensor, use_cuda: bool) -> None:
    target_batch = torch.tensor(np.array([100]), dtype=torch.long)
    if use_cuda:
        target_batch = target_batch.cuda()

    optimizer = torch.optim.SGD(hms2_model.parameters(), lr=0.01)

    optimizer.zero_grad()
    hms2_output = hms2_model(image_batch)
    hms2_output = torch.min(hms2_output, torch.tensor(-999.9, device=hms2_output.device))
    hms2_loss = nn.CrossEntropyLoss()(hms2_output, target_batch)
    hms2_loss.backward()
    hms2_grads = [parameter.grad for parameter in hms2_model.parameters()]
    optimizer.step()

    for grad in hms2_grads:
        assert grad is None or torch.count_nonzero(grad).item() == 0


def test_hms2_model_backward_with_not_requires_grad(
    hms2_model: nn.Module,
    image_batch: torch.Tensor,
    use_cuda: bool,
) -> None:
    list(hms2_model.parameters())[0].requires_grad_(False)

    target_batch = torch.tensor(np.array([100]), dtype=torch.long)
    if use_cuda:
        target_batch = target_batch.cuda()

    optimizer = torch.optim.SGD(hms2_model.parameters(), lr=0.01)

    optimizer.zero_grad()
    hms2_output = hms2_model(image_batch)
    hms2_output = torch.min(hms2_output, torch.tensor(0.0, device=hms2_output.device))
    hms2_loss = nn.CrossEntropyLoss()(hms2_output, target_batch)
    hms2_loss.backward()
    hms2_grads = [parameter.grad for parameter in hms2_model.parameters()]
    optimizer.step()

    assert hms2_grads[0] is None

    list(hms2_model.parameters())[0].requires_grad_(True)


def test_hms2_model_with_cache_background_forward(
    conv_module: nn.Module,
    dense_module: nn.Module,
    local_pooling_module: t.Optional[nn.Module],
    use_cuda: bool,
) -> None:
    # Create a huge white image
    height = 5000
    width = 5000
    image = np.full(shape=(height, width, 3), fill_value=255, dtype=np.uint8)
    image_batch = torch.tensor(image, dtype=torch.uint8)
    image_batch = image_batch[np.newaxis, :, :, :]

    # Create models
    tile_size = 3072
    loader_module = PlainLoaderModule()
    if use_cuda:
        loader_module = loader_module.cuda()
    hms2_model_use = Hms2Model(
        loader_module=loader_module,
        conv_module=conv_module,
        pre_pooling_module=None,
        dense_module=dense_module,
        local_pooling_module=local_pooling_module,
        tile_size=tile_size,
        emb_crop_size=7,
        emb_stride_size=32,
        cache_background_forward=True,
    )
    hms2_model_nouse = Hms2Model(
        loader_module=loader_module,
        conv_module=conv_module,
        pre_pooling_module=None,
        dense_module=dense_module,
        local_pooling_module=local_pooling_module,
        tile_size=tile_size,
        emb_crop_size=7,
        emb_stride_size=32,
        cache_background_forward=False,
    )

    # Test forward
    time_1 = time()
    use_output = hms2_model_use(image_batch)
    use_output = use_output.detach().cpu().numpy()
    time_2 = time()
    use_time = time_2 - time_1

    time_1 = time()
    nouse_output = hms2_model_nouse(image_batch)
    nouse_output = nouse_output.detach().cpu().numpy()
    time_2 = time()
    nouse_time = time_2 - time_1

    np.testing.assert_allclose(use_output, nouse_output)
    if use_time > nouse_time:
        warnings.warn(f'use_time {use_time} is longer than no_use_time {nouse_time}')


def test_hms2_model_with_cache_background_backward(
    conv_module: nn.Module,
    dense_module: nn.Module,
    local_pooling_module: t.Optional[nn.Module],
    use_cuda: bool,
) -> None:
    # Create a huge white image
    height = 5000
    width = 5000
    image = np.full(shape=(height, width, 3), fill_value=255, dtype=np.uint8)
    image_batch = torch.tensor(image, dtype=torch.uint8)
    image_batch = image_batch[np.newaxis, :, :, :]
    target_batch = torch.tensor(np.array([100]), dtype=torch.long)
    if use_cuda:
        target_batch = target_batch.cuda()

    # Create models
    tile_size = 3072
    loader_module = PlainLoaderModule()
    if use_cuda:
        loader_module = loader_module.cuda()
    hms2_model_use = Hms2Model(
        loader_module=loader_module,
        conv_module=conv_module,
        pre_pooling_module=None,
        dense_module=dense_module,
        local_pooling_module=local_pooling_module,
        tile_size=tile_size,
        emb_crop_size=7,
        emb_stride_size=32,
        skip_no_grad=False,
        cache_background_backward=True,
    )
    hms2_model_nouse = Hms2Model(
        loader_module=loader_module,
        conv_module=conv_module,
        pre_pooling_module=None,
        dense_module=dense_module,
        local_pooling_module=local_pooling_module,
        tile_size=tile_size,
        emb_crop_size=7,
        emb_stride_size=32,
        skip_no_grad=False,
        cache_background_backward=False,
    )

    # Test backward
    hms2_model_use.zero_grad()
    use_output = hms2_model_use(image_batch)
    loss = nn.CrossEntropyLoss()(use_output, target_batch)
    time_1 = time()
    loss.backward()
    time_2 = time()
    use_grads = [parameter.grad.cpu().numpy() for parameter in hms2_model_use.parameters()]
    use_time = time_2 - time_1

    hms2_model_nouse.zero_grad()
    nouse_output = hms2_model_nouse(image_batch)
    loss = nn.CrossEntropyLoss()(nouse_output, target_batch)
    time_1 = time()
    loss.backward()
    time_2 = time()
    nouse_grads = [parameter.grad.cpu().numpy() for parameter in hms2_model_nouse.parameters()]
    nouse_time = time_2 - time_1

    for use_grad, nouse_grad in zip(use_grads, nouse_grads):
        np.testing.assert_allclose(use_grad, nouse_grad)
    if use_time > nouse_time:
        warnings.warn(f'use_time {use_time} is longer than no_use_time {nouse_time}')
