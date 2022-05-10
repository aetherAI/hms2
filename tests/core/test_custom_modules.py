from io import BytesIO

import numpy as np
import pytest
import requests
import torch
import torchvision
from PIL import Image

from hms2.core.custom_modules import FrozenBatchNorm2d


@pytest.fixture(scope="session")
def image():
    url = "https://upload.wikimedia.org/wikipedia/zh/3/34/Lenna.jpg"
    with requests.get(url) as req:
        buff = BytesIO(req.content)
    image = Image.open(buff)

    width = 224
    height = 224
    image = image.resize([width, height])

    image = np.array(image)
    return image


def test_frozen_batch_norm_2d(image):
    original_model = torchvision.models.resnet50(pretrained=True).cuda().eval()
    image = torchvision.transforms.ToTensor()(image)
    image = torchvision.transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )(image)
    image_batch = image[np.newaxis, :, :, :].cuda()
    label_batch = torch.zeros([1], dtype=torch.int64).cuda()

    original_output = original_model(image_batch)
    loss = torch.nn.CrossEntropyLoss()(original_output, label_batch)
    loss.backward()
    original_grads = [
        parameter.grad.cpu().numpy() for parameter in original_model.parameters()
    ]
    original_model.zero_grad()

    frozen_bn_model = FrozenBatchNorm2d.convert_frozen_batchnorm(original_model)
    frozen_bn_model.train()
    frozen_bn_output = frozen_bn_model(image_batch)
    loss = torch.nn.CrossEntropyLoss()(frozen_bn_output, label_batch)
    loss.backward()
    frozen_bn_grads = [
        parameter.grad.cpu().numpy() for parameter in frozen_bn_model.parameters()
    ]
    frozen_bn_model.zero_grad()

    # Check the integrity of parameters
    original_parameters = [
        parameter.detach().cpu().numpy() for parameter in original_model.parameters()
    ]
    frozen_bn_parameters = [
        parameter.detach().cpu().numpy() for parameter in frozen_bn_model.parameters()
    ]
    assert len(original_parameters) == len(frozen_bn_parameters)
    for idx, _ in enumerate(original_parameters):
        assert original_parameters[idx] == pytest.approx(frozen_bn_parameters[idx])

    # Check the integrities of outputs and gradients
    assert original_output.detach().cpu().numpy() == pytest.approx(
        frozen_bn_output.detach().cpu().numpy()
    )
    for idx, _ in enumerate(original_grads):
        assert original_grads[idx] == pytest.approx(frozen_bn_grads[idx], abs=1e-4)
