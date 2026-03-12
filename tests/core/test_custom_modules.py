import typing as t

import cv2
import numpy as np
import numpy.typing as npt
import pytest
import torch
import torchvision
from PIL import Image

from hms2.core.custom_modules import (
    EuclideanFarthestPointPrePooling, FrozenBatchNorm2d,
    GaussianBlurAugmentorModule, LogSumExpPool2d, ReCAM, ReLSEPool2d, SoftplusInv,
)


@pytest.fixture(scope='session')
def image() -> npt.NDArray[np.uint8]:
    image = Image.open('misc/lena_color.gif').convert('RGB')

    width = 224
    height = 224
    image = image.resize((width, height))

    image = np.array(image)
    return image


def test_gaussian_blur_augmentor_module(image: npt.NDArray[np.uint8]) -> None:
    kernel_size = 3
    sigma_range = (0.0, 1.0)

    augmentor = GaussianBlurAugmentorModule(
        kernel_size=kernel_size,
        sigma_range=sigma_range,
    )
    augmentor.randomize()
    image_batch = torch.tensor(image).permute((2, 0, 1))[np.newaxis, ...] / 255.0  # Shape: [1, C, H, W]. Value: [0, 1]
    actual_image_batch = augmentor.forward(
        image_batch=image_batch,
        is_background_tile=False,
    )
    actual = (actual_image_batch * 255.0)[0, ...].permute((1, 2, 0)).to(torch.uint8).numpy()  # [H, W, C]

    desired = cv2.GaussianBlur(
        src=image,
        ksize=[kernel_size, kernel_size],
        sigmaX=augmentor.sigma,
        sigmaY=augmentor.sigma,
    )  # [H, W, C]
    np.testing.assert_allclose(actual, desired, atol=2)


def test_frozen_batch_norm_2d(image):
    original_model = torchvision.models.resnet50(pretrained=True).eval()
    image = torchvision.transforms.ToTensor()(image)
    image = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
    image_batch = image[np.newaxis, :, :, :]
    label_batch = torch.zeros([1], dtype=torch.int64)

    original_output = original_model(image_batch)
    loss = torch.nn.CrossEntropyLoss()(original_output, label_batch)
    loss.backward()
    original_grads = []
    for parameter in original_model.parameters():
        assert parameter.grad is not None
        original_grads.append(parameter.grad.numpy())
    original_model.zero_grad()

    frozen_bn_model = FrozenBatchNorm2d.convert_frozen_batchnorm(original_model)
    frozen_bn_model.train()
    frozen_bn_output = frozen_bn_model(image_batch)
    loss = torch.nn.CrossEntropyLoss()(frozen_bn_output, label_batch)
    loss.backward()
    frozen_bn_grads = []
    for parameter in frozen_bn_model.parameters():
        assert parameter.grad is not None
        frozen_bn_grads.append(parameter.grad.numpy())
    frozen_bn_model.zero_grad()

    # Check the integrity of parameters
    original_parameters = [parameter.detach().numpy() for parameter in original_model.parameters()]
    frozen_bn_parameters = [parameter.detach().numpy() for parameter in frozen_bn_model.parameters()]
    assert len(original_parameters) == len(frozen_bn_parameters)
    for idx, _ in enumerate(original_parameters):
        np.testing.assert_allclose(frozen_bn_parameters[idx], original_parameters[idx])

    # Check the integrities of outputs and gradients
    np.testing.assert_allclose(frozen_bn_output.detach().numpy(), original_output.detach().numpy())
    for idx, _ in enumerate(original_grads):
        np.testing.assert_allclose(frozen_bn_grads[idx], original_grads[idx])


def test_log_sum_exp_pool_2d() -> None:
    input = torch.tensor([[[[2.0, 0.0], [-np.inf, 1.0]]]])
    output = LogSumExpPool2d(factor=1)(input)[0, 0, 0, 0]
    assert pytest.approx(output) == 1.02131160332
    input = torch.tensor([[[[-np.inf, -np.inf], [-np.inf, -np.inf]]]])
    output = LogSumExpPool2d(factor=1)(input)[0, 0, 0, 0]
    assert pytest.approx(output) == -np.inf


def _test_output_and_grad(
    function: t.Callable[[torch.Tensor], torch.Tensor],
    input: t.Union[float, npt.NDArray[np.float_]],
    expected_output: t.Union[float, npt.NDArray[np.float_]],
    expected_grad: t.Optional[t.Union[float, npt.NDArray[np.float_]]] = None,
) -> None:
    input_tensor = torch.tensor(input)
    input_tensor.requires_grad_(True)
    output_tensor = function(input_tensor)
    output = output_tensor.detach().cpu().numpy()
    np.testing.assert_allclose(output, expected_output, rtol=1e-3)
    if expected_grad is not None:
        grad = torch.autograd.grad(output_tensor, [input_tensor])[0].cpu().numpy()
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-3)


def test_softplus_inv() -> None:
    softplus_inv = SoftplusInv()
    _test_output_and_grad(
        function=softplus_inv,
        input=0.5,
        expected_output=-0.43275212956,
        expected_grad=2.5414941,
    )
    _test_output_and_grad(
        function=softplus_inv,
        input=0.0,
        expected_output=-np.inf,
        expected_grad=None,
    )
    _test_output_and_grad(
        function=softplus_inv,
        input=1000.0,
        expected_output=1000.0,
        expected_grad=1.0,
    )


def test_re_lse_pool_2d() -> None:
    module = ReLSEPool2d()

    _test_output_and_grad(
        function=module,
        input=np.array([[[[0.0, 0.0], [0.0, 0.0]]]]),
        expected_output=np.array([[[[0.0]]]]),
        expected_grad=np.array([[[[1.0, 1.0], [1.0, 1.0]]]]),
    )
    _test_output_and_grad(
        function=module,
        input=np.array([[[[1.0, 1.0], [1.0, 1.0]]]]),
        expected_output=np.array([[[[2.06345536]]]]),
        expected_grad=np.array([[[[0.34526074837, 0.34526074837], [0.34526074837, 0.34526074837]]]]),
    )
    _test_output_and_grad(
        function=module,
        input=np.array([[[[100.0, 100.0], [100.0, 100.0]]]]),
        expected_output=np.array([[[[101.38629]]]]),
        expected_grad=np.array([[[[0.25, 0.25], [0.25, 0.25]]]]),
    )
    _test_output_and_grad(
        function=module,
        input=np.array([[[[0.0, 1.0], [0.0, 0.0]]]]),
        expected_output=np.array([[[[1.0]]]]),
        expected_grad=np.array([[[[0.36787944117, 1.0], [0.36787944117, 0.36787944117]]]]),
    )
    _test_output_and_grad(
        function=module,
        input=np.array([[[[0.0, 100.0], [0.0, 0.0]]]]),
        expected_output=np.array([[[[100.0]]]]),
        expected_grad=np.array([[[[3.720076e-44, 1.0], [3.720076e-44, 3.720076e-44]]]]),
    )


def test_re_cam() -> None:
    module = ReCAM(n=1e6)

    _test_output_and_grad(
        function=module,
        input=np.array([[[[0.0, 1.0], [10.0, 100.0]]]]),
        expected_output=np.array([[[[0.0, 14.3568359946], [23.815465157, 113.815510558]]]]),
    )


def test_euclidean_farthest_point_pre_pooling() -> None:
    module = EuclideanFarthestPointPrePooling(side_sample_ratio=0.5)
    feature_map = torch.tensor(
        [
            [[0.0, 0.1, 0.0, 0.1], [1.0, 1.1, 0.9, 1.1], [0.1, 0.0, 0.0, 0.2], [1.0, 1.1, 1.2, 0.9]],
            [[0.0, 0.0, 0.2, 0.1], [1.1, 0.9, 0.9, 1.0], [1.0, 1.1, 1.2, 0.9], [0.1, 0.0, 0.0, 0.2]],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)  # Shape: [1, 2, 4, 4]
    expected_output = torch.tensor(
        [
            [[0.0, 1.0], [1.2, 0.0]],
            [[0.0, 1.1], [0.0, 1.2]],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)  # Shape: [1, 2, 2, 2]

    output = module(feature_map)
    np.testing.assert_allclose(output.detach().cpu().numpy(), expected_output.detach().cpu().numpy(), rtol=1e-3)
