import os
from io import BytesIO
from time import time

import numpy as np
import pytest
import requests
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity

from hms2.core.loader_modules import (
    GPUAugmentationLoaderModule,
    NoLoaderModule,
    PlainLoaderModule,
)
from hms2.core.model import Hms2Model


@pytest.fixture(autouse=True, scope="session")
def set_up():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True


@pytest.fixture(scope="session")
def image():
    url = "https://upload.wikimedia.org/wikipedia/zh/3/34/Lenna.jpg"
    with requests.get(url) as req:
        buff = BytesIO(req.content)
    image = Image.open(buff)

    width = 2000
    height = 3000
    image = image.resize([width, height])

    image = np.array(image)
    return image


@pytest.fixture(scope="session")
def image_batch(image):
    image_batch = torch.tensor(image, dtype=torch.uint8)
    image_batch = image_batch[np.newaxis, :, :, :]

    return image_batch


@pytest.mark.parametrize("do_hint", [True, False])
@pytest.mark.parametrize("loader_module_use", ["plain", "gpu_aug_disable_aug", "no"])
def test_loader_module_forward_with_no_aug(
    image, image_batch, do_hint, loader_module_use
):
    if loader_module_use == "plain":
        loader_module = PlainLoaderModule()
    elif loader_module_use == "gpu_aug_disable_aug":
        loader_module = GPUAugmentationLoaderModule(
            random_flip=False,
            random_rotation=False,
            random_translation=None,
        )
    elif loader_module_use == "no":
        loader_module = NoLoaderModule()
    else:
        assert False

    loader_module = loader_module.cuda()

    coord = (0, 1000)
    size = (1000, 2000)
    if loader_module_use in ["plain", "gpu_aug_disable_aug"]:
        if do_hint:
            loader_module.hint_future_accesses(image_batch, [coord], [size])
        output = loader_module(image_batch, coord, size)
        assert isinstance(output, torch.Tensor)
        assert output.is_cuda
    elif loader_module_use == "no":
        partial_image_batch = image_batch[
            :,
            coord[1] : coord[1] + size[1],
            coord[0] : coord[0] + size[0],
            :,
        ]
        output = loader_module(partial_image_batch)
        assert isinstance(output, torch.Tensor)
    else:
        assert False

    output = output.cpu().numpy()
    output = output[0, ...]
    output = np.transpose(output, [1, 2, 0])
    output *= np.float32([0.229, 0.224, 0.225])
    output += np.float32([0.485, 0.456, 0.406])
    output = np.minimum(np.maximum(output * 255.0, 0.0), 255.0).astype(np.uint8)

    ground_truth = image[
        coord[1] : coord[1] + size[1],
        coord[0] : coord[0] + size[0],
        :,
    ]
    ssim = structural_similarity(output, ground_truth, channel_axis=-1)
    assert ssim > 0.99


def test_gpu_augmentation_loader_module_forward_with_aug(image, image_batch):
    # Augmentation arguments
    rotation_angle = 8.7
    translation_pixels = [9, -8]
    do_flip = True

    class AddBias(nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.bias = bias

        def randomize(self):
            pass

        def forward(self, inputs):
            return inputs + self.bias

    other_augmentations = [AddBias(0.1)]

    # Get the loader module
    loader_module = GPUAugmentationLoaderModule(other_augmentations=other_augmentations)
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
    assert output.is_cuda

    output = output.cpu().numpy()
    output = output[0, ...]
    output = np.transpose(output, [1, 2, 0])
    output *= np.float32([0.229, 0.224, 0.225])
    output += np.float32([0.485, 0.456, 0.406])
    output -= 0.1  # Inverse of AddBias(0.1)
    output = np.minimum(np.maximum(output * 255.0, 0.0), 255.0).astype(np.uint8)

    # Get ground truth
    img_aug = Image.fromarray(image)
    img_aug = img_aug.rotate(
        angle=rotation_angle,
        resample=Image.BILINEAR,
        translate=translation_pixels,
        fillcolor=(255, 255, 255),
    )
    if do_flip:
        img_aug = img_aug.transpose(method=Image.FLIP_LEFT_RIGHT)
    ground_truth = np.array(img_aug)[
        coord[1] : coord[1] + size[1],
        coord[0] : coord[0] + size[0],
        :,
    ]
    ssim = structural_similarity(output, ground_truth, channel_axis=-1)
    assert np.min(ground_truth) < 128  # The selected tile should be meaningful
    assert ssim > 0.99


def test_gpu_augmentation_loader_module_forward_with_randomness(image_batch):
    # Augmentation arguments
    class AddBias(nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.bias = bias

        def randomize(self):
            pass

        def forward(self, inputs):
            return inputs + self.bias

    other_augmentations = [AddBias(0.1)]

    # Get the loader module
    loader_module = GPUAugmentationLoaderModule(other_augmentations=other_augmentations)
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


@pytest.fixture(scope="session")
def conv_module():
    resnet50 = torchvision.models.resnet50(pretrained=True).eval()
    conv_module = nn.Sequential(*list(resnet50.children())[:-2])
    conv_module = conv_module.cuda()
    return conv_module


@pytest.fixture(scope="session")
def dense_module():
    resnet50 = torchvision.models.resnet50(pretrained=True).eval()
    dense_module = nn.Sequential(
        nn.AdaptiveMaxPool2d((1, 1)),
        nn.Flatten(),
        list(resnet50.children())[-1],
    )
    dense_module = dense_module.cuda()
    return dense_module


@pytest.fixture(scope="session", params=["max", "none"])
def local_pooling_module(request):
    if request.param == "max":
        local_pooling_module = nn.AdaptiveMaxPool2d((1, 1))
    else:
        local_pooling_module = None
    return local_pooling_module


@pytest.fixture(scope="session", params=[3072, 4096])
def hms2_model(conv_module, dense_module, local_pooling_module, request):
    tile_size = request.param

    hms2_model = Hms2Model(
        loader_module=PlainLoaderModule().cuda(),
        conv_module=conv_module,
        dense_module=dense_module,
        local_pooling_module=local_pooling_module,
        tile_size=tile_size,
        emb_crop_size=7,
        emb_stride_size=32,
    )
    return hms2_model


@pytest.fixture(scope="session")
def plain_model(conv_module, dense_module):
    class PlainModel(nn.Module):
        def __init__(self, conv_module, dense_module):
            super().__init__()
            self.conv_module = conv_module
            self.dense_module = dense_module

        def forward(self, image_batch):
            image_batch = image_batch.cuda()
            image_batch = image_batch.permute(0, 3, 1, 2).contiguous()
            image_batch = image_batch.float().div(255.0)
            image_batch = transforms.functional.normalize(
                tensor=image_batch,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            conv_output = self.conv_module(image_batch)
            output = self.dense_module(conv_output)
            return output

    plain_model = PlainModel(conv_module, dense_module)
    return plain_model


def test_hms2_model_forward(hms2_model, plain_model, image_batch):
    hms2_output = hms2_model(image_batch)
    hms2_output = hms2_output.detach().cpu().numpy()

    plain_output = plain_model(image_batch)
    plain_output = plain_output.detach().cpu().numpy()

    assert hms2_output == pytest.approx(plain_output)


def test_hms2_model_backward(hms2_model, plain_model, image_batch):
    target_batch = torch.tensor(np.array([100]), dtype=torch.long).cuda()

    hms2_output = hms2_model(image_batch)
    hms2_loss = nn.CrossEntropyLoss()(hms2_output, target_batch)
    hms2_model.zero_grad()
    hms2_loss.backward()
    hms2_grads = [parameter.grad.cpu().numpy() for parameter in hms2_model.parameters()]

    plain_output = plain_model(image_batch)
    plain_loss = nn.CrossEntropyLoss()(plain_output, target_batch)
    plain_model.zero_grad()
    plain_loss.backward()
    plain_grads = [
        parameter.grad.cpu().numpy() for parameter in plain_model.parameters()
    ]

    assert len(hms2_grads) == len(plain_grads)
    for idx, _ in enumerate(hms2_grads):
        assert hms2_grads[idx] == pytest.approx(plain_grads[idx], abs=1e-4)


def test_hms2_model_backward_with_no_grad(hms2_model, image_batch):
    target_batch = torch.tensor(np.array([100]), dtype=torch.long).cuda()

    optimizer = torch.optim.SGD(hms2_model.parameters(), lr=0.01)

    optimizer.zero_grad()
    hms2_output = hms2_model(image_batch)
    hms2_output = torch.min(hms2_output, torch.tensor(-999.9).cuda())
    hms2_loss = nn.CrossEntropyLoss()(hms2_output, target_batch)
    hms2_loss.backward()
    hms2_grads = [parameter.grad for parameter in hms2_model.parameters()]
    optimizer.step()

    for grad in hms2_grads:
        assert grad is None or torch.count_nonzero(grad).item() == 0


def test_hms2_model_with_cache_background_forward(
    conv_module,
    dense_module,
    local_pooling_module,
):
    # Create a huge white image
    height = 10000
    width = 10000
    image = np.full(shape=(height, width, 3), fill_value=255, dtype=np.uint8)
    image_batch = torch.tensor(image, dtype=torch.uint8)
    image_batch = image_batch[np.newaxis, :, :, :]

    # Create models
    tile_size = 3072
    hms2_model_use = Hms2Model(
        loader_module=PlainLoaderModule().cuda(),
        conv_module=conv_module,
        dense_module=dense_module,
        local_pooling_module=local_pooling_module,
        tile_size=tile_size,
        emb_crop_size=7,
        emb_stride_size=32,
        cache_background_forward=True,
    )
    hms2_model_nouse = Hms2Model(
        loader_module=PlainLoaderModule().cuda(),
        conv_module=conv_module,
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

    assert use_output == pytest.approx(nouse_output)
    assert use_time < nouse_time


def test_hms2_model_with_cache_background_backward(
    conv_module,
    dense_module,
    local_pooling_module,
):
    # Create a huge white image
    height = 10000
    width = 10000
    image = np.full(shape=(height, width, 3), fill_value=255, dtype=np.uint8)
    image_batch = torch.tensor(image, dtype=torch.uint8)
    image_batch = image_batch[np.newaxis, :, :, :]
    target_batch = torch.tensor(np.array([100]), dtype=torch.long).cuda()

    # Create models
    tile_size = 3072
    hms2_model_use = Hms2Model(
        loader_module=PlainLoaderModule().cuda(),
        conv_module=conv_module,
        dense_module=dense_module,
        local_pooling_module=local_pooling_module,
        tile_size=tile_size,
        emb_crop_size=7,
        emb_stride_size=32,
        skip_no_grad=False,
        cache_background_backward=True,
    )
    hms2_model_nouse = Hms2Model(
        loader_module=PlainLoaderModule().cuda(),
        conv_module=conv_module,
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
    use_grads = [
        parameter.grad.cpu().numpy() for parameter in hms2_model_use.parameters()
    ]
    use_time = time_2 - time_1

    hms2_model_nouse.zero_grad()
    nouse_output = hms2_model_nouse(image_batch)
    loss = nn.CrossEntropyLoss()(nouse_output, target_batch)
    time_1 = time()
    loss.backward()
    time_2 = time()
    nouse_grads = [
        parameter.grad.cpu().numpy() for parameter in hms2_model_nouse.parameters()
    ]
    nouse_time = time_2 - time_1

    for use_grad, nouse_grad in zip(use_grads, nouse_grads):
        assert use_grad == pytest.approx(nouse_grad, abs=1e-4)
    assert use_time < nouse_time
