"""
This module provides a model building tool.
"""
from collections import namedtuple
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torchvision

from .custom_modules import (
    FrozenBatchNorm2d,
    HEDPerturbAugmentorModule,
    LogSumExpPool2d,
    PermuteLayer,
    ScaleAndShift,
    ToDevice,
)
from .loader_modules import (
    GPUAugmentationLoaderModule,
    NoLoaderModule,
    PlainLoaderModule,
)
from .model import Hms2Model


class Hms2ModelBuilder:
    def __init__(self):
        self.Augmentation = namedtuple(
            "Augmentation",
            ["build_func"],
        )
        self.Backbone = namedtuple(
            "Backbone",
            ["build_func", "output_channels", "get_hms2_parameters"],
        )
        self.Pooling = namedtuple(
            "Pooling",
            ["local_pooling_build_func", "pooling_build_func"],
        )
        self.CustomDense = namedtuple(
            "CustomDense",
            ["custom_dense_build_func"],
        )

        self.augmentation_registry = {}
        self.backbone_registry = {}
        self.pooling_registry = {}
        self.custom_dense_registry = {}

        self._register_builtins()

    def build(
        self,
        n_classes: int,
        augmentation_list: Optional[Sequence[str]] = None,
        backbone: str = "resnet50_frozenbn",
        pretrained: bool = True,
        pooling: str = "gmp",
        custom_dense: Optional[str] = None,
        use_hms2: bool = True,
        device: Optional[Union[torch.device, str, int]] = None,
        use_cpu_for_dense: bool = False,
        gpu_memory_budget: float = 32.0,
    ) -> nn.Module:
        """
        Build a model given parameters.

        Args:
            n_classes (int): The number of classes.
            augmentation_list (list or NoneType):
                A list of str, each of which specify an augmentation process, including
                "flip", "rigid", and "hed_perturb". The default is None that disables
                GPU augmentations.
            backbone (str):
                Specify the backbone structure. One of "resnet50_frozenbn" (default).
            pretrained (bool):
                Whether to load pretrained weights for the backbone (default: True).
            pooling (str or NoneType):
                Specify the pooling function. One of "gmp", "gmp_scaled", "gap", "lse",
                "cam", and "no".
            custom_dense (Optional[str]):
                Specify the module after pooling if not using the standard single dense
                layer. One of "no".
            use_hms2 (bool): Whether to enable HMS2. The default is True.
            device (torch.device):
                The device to place modules. If None (default), it calls
                torch.cuda.current_device() to get the device.
            use_cpu_for_dense: Whether to compute dense layers using CPU.
            gpu_memory_budget (float):
                The GPU memory capacity to let the builder determine the parameters of
                HMS2.
        """
        # Default arguments
        if augmentation_list is None:
            augmentation_list = []
        if device is None:
            device = torch.cuda.current_device()

        # Build components
        loader_module = self._build_loader_module(
            use_hms2=use_hms2,
            augmentation_list=augmentation_list,
            device=device,
        )
        backbone_module = self._build_backbone_module(
            backbone=backbone,
            pretrained=pretrained,
            device=device,
        )
        local_pooling_module = self._build_local_pooling_module(
            pooling=pooling,
            device=device,
            use_cpu_for_dense=use_cpu_for_dense,
        )
        dense_module = self._build_dense_module(
            backbone=backbone,
            pooling=pooling,
            custom_dense=custom_dense,
            n_classes=n_classes,
            device=device,
            use_cpu_for_dense=use_cpu_for_dense,
        )

        # Build the model
        model: nn.Module
        if use_hms2:
            hms2_parameters = self.backbone_registry[backbone].get_hms2_parameters(
                gpu_memory_budget=gpu_memory_budget,
            )

            model = Hms2Model(
                loader_module=loader_module,
                conv_module=backbone_module,
                dense_module=dense_module,
                local_pooling_module=local_pooling_module,
                **hms2_parameters,
            )
        else:
            model = _PlainModel(
                loader_module=loader_module,
                conv_module=backbone_module,
                dense_module=dense_module,
                local_pooling_module=local_pooling_module,
            )

        return model

    def register_augmentation(
        self,
        signature: str,
        build_func: Callable,
    ) -> None:
        """Register an augmentation.

        Args:
            signature: The name of the augmentation.
            build_func: Calling build_func() will yield an nn.Module.
        """
        self.augmentation_registry[signature] = self.Augmentation(
            build_func=build_func,
        )

    def register_backbone(
        self,
        signature: str,
        build_func: Callable,
        output_channels: int,
        get_hms2_parameters: Callable,
    ) -> None:
        """Register a backbone.

        Args:
            signature: The name of the backbone.
            build_func:
                Calling build_func(pretrained=xxx) will yield an nn.Module.
                "pretrained: bool" must be included as an argument.
            output_channels: The number of the output channels.
            get_hms2_parameters:
                A callable with a parameter `gpu_memory_budget`. It returns a dict
                with the keys "tile_size", "emb_crop_size", and "emb_stride_size".
        """
        self.backbone_registry[signature] = self.Backbone(
            build_func=build_func,
            output_channels=output_channels,
            get_hms2_parameters=get_hms2_parameters,
        )

    def register_pooling(
        self,
        signature: str,
        local_pooling_build_func: Optional[Callable],
        pooling_build_func: Callable,
    ) -> None:
        """Register a pooling.

        Args:
            signature: The name of the pooling.
            local_pooling_build_func:
                Calling local_pooling_build_func() will yield an nn.Module. This
                pooling will be applied before HMS2 aggregation.
            pooling_build_func:
                Calling pooling_build_func() will yield an nn.Module. This pooling will
                be applied before linear layers.
        """
        self.pooling_registry[signature] = self.Pooling(
            local_pooling_build_func=local_pooling_build_func,
            pooling_build_func=pooling_build_func,
        )

    def register_custom_dense(
        self,
        signature: str,
        custom_dense_build_func: Callable[[int], nn.Module],
    ) -> None:
        """Register a custom dense.

        Args:
            signature: The name of the custom dense.
            custom_dense_build_func:
                Calling custom_dense_build_func(num_classes) will yeild an nn.Module.
        """
        self.custom_dense_registry[signature] = self.CustomDense(
            custom_dense_build_func=custom_dense_build_func,
        )

    def _register_builtins(self):
        # Augmentations
        self.register_augmentation("hed_perturb", HEDPerturbAugmentorModule)

        # Backbones: ResNet50 with frozen BN layers.
        def resnet50_frozenbn_build_func(pretrained: bool) -> nn.Module:
            module = torchvision.models.resnet50(pretrained=pretrained)
            module = FrozenBatchNorm2d.convert_frozen_batchnorm(module)
            module = nn.Sequential(*list(module.children())[:-2])
            return module

        def resnet50_frozenbn_get_hms2_parameters(gpu_memory_budget: float) -> dict:
            if gpu_memory_budget >= 32:
                parameters = {
                    "tile_size": 3072,
                    "emb_crop_size": 7,
                    "emb_stride_size": 32,
                }
            else:
                parameters = {
                    "tile_size": 2048,
                    "emb_crop_size": 7,
                    "emb_stride_size": 32,
                }
            return parameters

        self.register_backbone(
            signature="resnet50_frozenbn",
            build_func=resnet50_frozenbn_build_func,
            output_channels=2048,
            get_hms2_parameters=resnet50_frozenbn_get_hms2_parameters,
        )

        # Poolings
        self.register_pooling(
            "gmp",
            local_pooling_build_func=(lambda: nn.AdaptiveMaxPool2d((1, 1))),
            pooling_build_func=(
                lambda: nn.Sequential(
                    nn.AdaptiveMaxPool2d((1, 1)),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            "gmp_scaled",
            local_pooling_build_func=(
                lambda: nn.Sequential(
                    ScaleAndShift(scale=3.79, bias=(-17.7)),
                    nn.AdaptiveMaxPool2d((1, 1)),
                )
            ),
            pooling_build_func=(
                lambda: nn.Sequential(
                    nn.AdaptiveMaxPool2d((1, 1)),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            "gmp_scaled_1k",
            local_pooling_build_func=(
                lambda: nn.Sequential(
                    ScaleAndShift(scale=3.933, bias=(-19.14)),
                    nn.AdaptiveMaxPool2d((1, 1)),
                )
            ),
            pooling_build_func=(
                lambda: nn.Sequential(
                    nn.AdaptiveMaxPool2d((1, 1)),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            "gmp_scaled_2k",
            local_pooling_build_func=(
                lambda: nn.Sequential(
                    ScaleAndShift(scale=4.135, bias=(-21.23)),
                    nn.AdaptiveMaxPool2d((1, 1)),
                )
            ),
            pooling_build_func=(
                lambda: nn.Sequential(
                    nn.AdaptiveMaxPool2d((1, 1)),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            "gap",
            local_pooling_build_func=(lambda: nn.AdaptiveAvgPool2d((1, 1))),
            pooling_build_func=(
                lambda: nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            "lse",
            local_pooling_build_func=(lambda: LogSumExpPool2d(factor=1.0)),
            pooling_build_func=(
                lambda: nn.Sequential(
                    LogSumExpPool2d(factor=1.0),
                    nn.Flatten(),
                )
            ),
        )
        self.register_pooling(
            "cam",
            local_pooling_build_func=None,
            pooling_build_func=(lambda: PermuteLayer(dims=(0, 2, 3, 1))),
        )
        self.register_pooling(
            "no",
            local_pooling_build_func=None,
            pooling_build_func=(lambda: nn.Identity()),
        )

        # Custom Dense
        self.register_custom_dense(
            "no",
            custom_dense_build_func=(lambda: nn.Identity()),
        )

    def _build_loader_module(
        self,
        use_hms2: bool,
        augmentation_list: Sequence[str],
        device: Union[torch.device, str, int],
    ) -> nn.Module:
        # Translate the augmentation_list
        augmentation_modules = []
        for augmentation in augmentation_list:
            if use_hms2 and augmentation in ["flip", "rigid"]:
                # "flip" and "rigid" are built-in of HMS2 to enable patch-based
                # affine transformation. Skip initiating a module.
                pass
            elif augmentation in self.augmentation_registry:
                module = self.augmentation_registry[augmentation].build_func()
                augmentation_modules.append(module)

            else:
                raise RuntimeError(
                    f"{augmentation} has not yet been registered as an augmentation."
                )

        # Build the loader module
        loader_module: nn.Module
        if use_hms2:
            if augmentation_list is None:
                loader_module = PlainLoaderModule()
            else:
                random_rotation = "rigid" in augmentation_list
                random_translation = (
                    (-32.0, 32.0) if "rigid" in augmentation_list else None
                )
                random_flip = "flip" in augmentation_list

                loader_module = GPUAugmentationLoaderModule(
                    random_rotation=random_rotation,
                    random_translation=random_translation,
                    random_flip=random_flip,
                    other_augmentations=augmentation_modules,
                )
        else:
            loader_module = NoLoaderModule(augmentations=augmentation_modules)

        loader_module = loader_module.to(device)
        return loader_module

    def _build_backbone_module(
        self,
        backbone: str,
        pretrained: bool,
        device: Union[torch.device, str, int],
    ) -> nn.Module:
        if backbone not in self.backbone_registry:
            raise RuntimeError(f"{backbone} has not yet registered as a backbone.")

        backbone_module = self.backbone_registry[backbone].build_func(
            pretrained=pretrained
        )
        backbone_module = backbone_module.to(device)
        return backbone_module

    def _build_local_pooling_module(
        self,
        pooling: str,
        device: Union[torch.device, str, int],
        use_cpu_for_dense: bool,
    ) -> Optional[nn.Module]:
        if pooling not in self.pooling_registry:
            raise RuntimeError(f"{pooling} has not yet registered as a pooling.")

        local_pooling_build_func = self.pooling_registry[
            pooling
        ].local_pooling_build_func
        if local_pooling_build_func is None:
            if use_cpu_for_dense:
                return ToDevice("cpu")
            else:
                return None

        local_pooling_module = local_pooling_build_func()
        if use_cpu_for_dense:
            local_pooling_module = nn.Sequential(
                ToDevice("cpu"),
                local_pooling_module,
            )
        else:
            local_pooling_module = local_pooling_module.to(device)

        return local_pooling_module

    def _build_dense_module(
        self,
        backbone: str,
        pooling: str,
        custom_dense: Optional[str],
        n_classes: int,
        device: Union[torch.device, str, int],
        use_cpu_for_dense: bool,
    ) -> nn.Module:
        if backbone not in self.backbone_registry:
            raise RuntimeError(f"{backbone} has not yet registered as a backbone.")

        output_channels = self.backbone_registry[backbone].output_channels

        if pooling not in self.pooling_registry:
            raise RuntimeError(f"{pooling} has not yet registered as a pooling.")

        pooling_module = self.pooling_registry[pooling].pooling_build_func()

        if custom_dense is None:
            dense_layer = nn.Linear(output_channels, n_classes, bias=True)
            with torch.no_grad():
                dense_layer.weight.div_(10.0)
                dense_layer.bias.div_(10.0)
        else:
            dense_layer = self.custom_dense_registry[
                custom_dense
            ].custom_dense_build_func()

        dense_module = nn.Sequential(
            pooling_module,
            dense_layer,
        )
        if not use_cpu_for_dense:
            dense_module = dense_module.to(device)

        return dense_module


class _PlainModel(nn.Module):
    """
    Plain model with a similar interface as Hms2Model.

    Args:
        See the descriptions in `Hms2Model`.
    """

    def __init__(
        self,
        loader_module: nn.Module,
        conv_module: nn.Module,
        dense_module: nn.Module,
        local_pooling_module: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.loader_module = loader_module
        self.conv_module = conv_module
        self.dense_module = dense_module
        self.local_pooling_module = local_pooling_module

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        """
        Implementation of a plain model.
        """
        if isinstance(img_batch, torch.Tensor):
            if len(img_batch.size()) != 4:
                raise ValueError("img_batch should have 4 dimensions")
        else:
            raise ValueError("img_batch should be torch.Tensor")

        loaded = self.loader_module(img_batch)
        conved = self.conv_module(loaded)
        if self.local_pooling_module is not None:
            local_pooled = self.local_pooling_module(conved)
        else:
            local_pooled = conved
        output = self.dense_module(local_pooled)

        return output
