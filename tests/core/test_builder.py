import os

import numpy as np
import pytest
import torch

from hms2.core.builder import Hms2ModelBuilder


@pytest.fixture(autouse=True, scope="session")
def set_up():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True


@pytest.fixture(scope="session")
def n_classes():
    return 10


@pytest.fixture(scope="session", params=["resnet50_frozenbn"])
def backbone(request):
    return request.param


@pytest.fixture(scope="session", params=["gmp", "gap"])
def pooling(request):
    return request.param


@pytest.fixture(scope="session", params=[False, True])
def use_hms2(request):
    return request.param


@pytest.fixture(scope="session", params=[None, ["flip", "rigid", "hed_perturb"]])
def augmentation_list(request):
    return request.param


def test_hms2_model_builder_with_dry_run(
    n_classes,
    backbone,
    pooling,
    use_hms2,
    augmentation_list,
):
    # Skip the situation that use_hms2 == False and augmentation_list is not None
    if not use_hms2 and augmentation_list is not None:
        return

    # Set larger image size for HMS2
    if use_hms2:
        image_size = (5000, 5000)
    else:
        image_size = (2000, 2000)

    # Build a model
    model = Hms2ModelBuilder().build(
        n_classes=n_classes,
        backbone=backbone,
        pooling=pooling,
        use_hms2=use_hms2,
        augmentation_list=augmentation_list,
    )

    # Dry-run backpropagation
    optimizer = torch.optim.AdamW(model.parameters())
    loss = torch.nn.CrossEntropyLoss()

    for _ in range(2):
        input_batch = np.random.randint(
            low=0, high=255, size=((1,) + image_size + (3,)), dtype=np.uint8
        )
        y_true_batch = np.random.randint(0, n_classes - 1, size=(1,), dtype=np.int64)
        input_batch = torch.tensor(input_batch)
        if not use_hms2:
            # If HMS2 is disabled, the input should be manually moved to GPU.
            input_batch = input_batch.cuda()

        y_pred_batch = model(input_batch)
        assert y_pred_batch.size() == (1, n_classes)

        y_true_batch = torch.tensor(y_true_batch).cuda()
        loss_batch = loss(y_pred_batch, y_true_batch)
        loss_batch.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_hms2_model_builder_with_use_less_gpu_memory_budget(
    n_classes,
    backbone,
):
    image_size = (5000, 5000)

    # Build two models with different GPU memory budgets
    model_rich = Hms2ModelBuilder().build(
        n_classes=n_classes,
        backbone=backbone,
        pooling="gap",
        use_hms2=True,
        gpu_memory_budget=32.0,
    )
    model_poor = Hms2ModelBuilder().build(
        n_classes=n_classes,
        backbone=backbone,
        pooling="gap",
        use_hms2=True,
        gpu_memory_budget=16.0,
    )

    # Run forward
    input_batch = np.random.randint(
        low=0, high=255, size=((1,) + image_size + (3,)), dtype=np.uint8
    )
    input_batch = torch.tensor(input_batch)

    y_pred_batch_rich = model_rich(input_batch)
    y_pred_batch_poor = model_poor(input_batch)

    assert y_pred_batch_poor.detach().cpu().numpy() == pytest.approx(
        y_pred_batch_rich.detach().cpu().numpy(), abs=1.0
    )


def test_hms2_model_builder_with_cam(
    n_classes,
    backbone,
    use_hms2,
):
    # Set larger image size for HMS2
    if use_hms2:
        image_size = (5000, 5000)
    else:
        image_size = (2000, 2000)

    # Build a model
    model = Hms2ModelBuilder().build(
        n_classes=n_classes,
        backbone=backbone,
        pooling="cam",
        use_hms2=use_hms2,
    )

    # Dry-run
    for _ in range(2):
        input_batch = np.random.randint(
            low=0, high=255, size=((1,) + image_size + (3,)), dtype=np.uint8
        )
        input_batch = torch.tensor(input_batch)
        if not use_hms2:
            # If HMS2 is disabled, the input should be manually moved to GPU.
            input_batch = input_batch.cuda()

        cam = model(input_batch)
        assert cam.size()[0] == 1
        assert cam.size()[1] > 1
        assert cam.size()[2] > 1
        assert cam.size()[3] == n_classes


def test_hms2_model_builder_with_emb(
    n_classes,
    backbone,
    use_hms2,
):
    # Set larger image size for HMS2
    if use_hms2:
        image_size = (5000, 5000)
    else:
        image_size = (2000, 2000)

    # Build a model
    model = Hms2ModelBuilder().build(
        n_classes=n_classes,
        backbone=backbone,
        pooling="no",
        custom_dense="no",
        use_hms2=use_hms2,
    )

    # Dry-run
    for _ in range(2):
        input_batch = np.random.randint(
            low=0, high=255, size=((1,) + image_size + (3,)), dtype=np.uint8
        )
        input_batch = torch.tensor(input_batch)
        if not use_hms2:
            # If HMS2 is disabled, the input should be manually moved to GPU.
            input_batch = input_batch.cuda()

        emb = model(input_batch)
        assert emb.size()[0] == 1
        assert emb.size()[1] == 2048
        assert emb.size()[2] > 1
        assert emb.size()[3] > 1
